# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Gemma model implementation."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, List

from . import config as gemma_config

from jetstream_pt import layers
import jax


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
  """Precomputes the frequency cis."""
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)
  freqs = torch.outer(t, freqs).float()
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


class RMSNorm(torch.nn.Module):

  def __init__(
      self,
      dim: int,
      eps: float = 1e-6,
      add_unit_offset: bool = True,
      device: str = "meta",
  ):
    super().__init__()
    self.eps = eps
    self.add_unit_offset = add_unit_offset
    self.weight = nn.Parameter(torch.zeros(dim, device=device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x):
    x = self._norm(x.float()).type_as(x)
    if self.add_unit_offset:
      output = x * (1 + self.weight)
    else:
      output = x * self.weight
    return output


class GemmaMLP(nn.Module):

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: int,
      device,
      env,
  ):
    super().__init__()
    Linear = (
        layers.WeightOnlyInt8Linear
        if env.enable_weight_quantization
        else torch.nn.Linear
    )
    self.gate_proj = Linear(hidden_size, intermediate_size, bias=False, device=device)
    self.up_proj = Linear(hidden_size, intermediate_size, bias=False, device=device)
    self.down_proj = Linear(intermediate_size, hidden_size, bias=False, device=device)

  def forward(self, x):
    gate = self.gate_proj(x)
    gate = F.gelu(gate, approximate="tanh")
    up = self.up_proj(x)
    fuse = gate * up
    outputs = self.down_proj(fuse)
    return outputs


class GemmaDecoderLayer(nn.Module):

  def __init__(self, config: gemma_config.GemmaConfig, env):
    super().__init__()
    self.self_attn = layers.Attention(
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.hidden_size,
        config.device,
        env,
    )
    self.mlp = GemmaMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        env=env,
        device=config.device,
    )
    self.input_layernorm = RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, device=config.device
    )
    self.post_attention_layernorm = RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, device=config.device
    )

  def forward(
      self,
      hidden_states: torch.Tensor,
      freqs_cis: torch.Tensor,
      cache: Any,
      mask: torch.Tensor,
  ) -> torch.Tensor:
    # Self Attention
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states,
        freqs_cis=freqs_cis,
        mask=mask,
        cache=cache,
    )
    hidden_states = residual + hidden_states

    # MLP
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states


class GemmaModel(nn.Module):

  def __init__(self, config: gemma_config.GemmaConfig, env):
    super().__init__()
    self.config = config
    self.vocab_size = config.vocab_size
    self.env = env

    self.layers = nn.ModuleList()
    for _ in range(config.num_hidden_layers):
      self.layers.append(GemmaDecoderLayer(config, env))
    self.norm = RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, device=config.device
    )
    Embedding = (
        layers.Int8Embedding
        if env.enable_weight_quantization
        else torch.nn.Embedding
    )

    self.embedder = Embedding(
        config.vocab_size, config.hidden_size, device=config.device
    )
    rope_theta = getattr(config, "rope_theta", 10000)
    freqs_cis = precompute_freqs_cis(
        config.head_dim, config.max_position_embeddings * 2, theta=rope_theta
    )
    self.register_buffer("freqs_cis", freqs_cis)

  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
  ):
    with jax.named_scope("transformer_freq"):
      bsz, seqlen = tokens.shape
      freqs_cis = self.freqs_cis[input_pos]
      freqs_cis = freqs_cis.reshape(bsz, seqlen, -1)

    hidden_states = self.embedder(tokens)
    hidden_states = hidden_states * (self.config.hidden_size**0.5)

    for i in range(len(self.layers)):
      layer = self.layers[i]
      hidden_states = layer(
          hidden_states=hidden_states,
          freqs_cis=freqs_cis,
          cache=caches[i],
          mask=mask,
      )
    hidden_states = self.norm(hidden_states)

    embedder_weight = self.embedder.weight
    if self.config.quant:
      embedder_weight = embedder_weight * self.embedder.weight_scaler.unsqueeze(
          -1
      )
    logits = torch.matmul(hidden_states, embedder_weight.t())
    return logits
