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


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (
      x.shape[0],
      x.shape[2],
      x.shape[3],
  ), f"freqs_cis: {freqs_cis.shape }, x: {x.shape}"
  shape = [d if i != 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
  """Applies the rotary embedding to the query and key tensors."""
  x_ = torch.view_as_complex(
      torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
  )
  freqs_cis = reshape_for_broadcast(freqs_cis, x_)
  x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
  x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
  x_out = x_out.reshape(
      x_out.shape[0], x_out.shape[1], x_out.shape[2], -1
  ).transpose(1, 2)
  return x_out


class GemmaAttention(nn.Module):

  def __init__(
      self,
      hidden_size: int,
      num_heads: int,
      num_kv_heads: int,
      head_dim: int,
      device,
      env,
  ):
    super().__init__()

    self.env = env

    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads

    assert self.num_heads % self.num_kv_heads == 0
    self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    self.hidden_size = hidden_size
    self.head_dim = head_dim

    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim

    self.scaling = self.head_dim**-0.5

    Linear = (
        layers.WeightOnlyInt8Linear
        if env.enable_weight_quantization
        else torch.nn.Linear
    )
    self.wq = Linear(
        hidden_size,
        num_heads * self.head_dim,
        bias=False,
        device=device,
    )
    self.wk = Linear(
        hidden_size,
        self.num_kv_heads * self.head_dim,
        bias=False,
        device=device,
    )
    self.wv = Linear(
        hidden_size,
        self.num_kv_heads * self.head_dim,
        bias=False,
        device=device,
    )
    self.o_proj = Linear(
        self.num_heads * self.head_dim,
        self.hidden_size,
        bias=False,
        device=device,
    )

    Kernel = (
        layers.Int8KVAttentionKernel
        if env.enable_kv_quantization
        else layers.AttentionKernel
    )
    self.attention_kernel = Kernel(env)

  def forward(
      self,
      hidden_states,
      freqs_cis,
      mask,
      cache,
  ) -> torch.Tensor:
    hidden_states_shape = hidden_states.shape
    assert len(hidden_states_shape) == 3
    batch_size, input_len, _ = hidden_states_shape

    xq = self.wq(hidden_states)
    xk = self.wk(hidden_states)
    xv = self.wv(hidden_states)

    xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
    xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
    xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

    if self.num_kv_heads > 1:
      self.env.apply_sharding(xq, axis=2)
      self.env.apply_sharding(xk, axis=2)
      self.env.apply_sharding(xv, axis=2)
    else:
      # Gemma 2B
      self.env.apply_sharding(xq, axis=3)
      self.env.apply_sharding(xk, axis=3)
      self.env.apply_sharding(xv, axis=3)

    # Positional embedding.
    xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
    xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

    # Write new kv cache.
    # [batch_size, input_len, n_local_kv_heads, head_dim]

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    xq = xq.transpose(1, 2)

    output = self.attention_kernel(xq, xk, xv, mask, cache)

    # [batch_size, input_len, hidden_dim]
    output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
    output = self.o_proj(output)
    return output


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
    self.gate_proj = Linear(
        hidden_size, intermediate_size, bias=False, device=device
    )
    self.up_proj = Linear(
        hidden_size, intermediate_size, bias=False, device=device
    )
    self.down_proj = Linear(
        intermediate_size, hidden_size, bias=False, device=device
    )

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
    self.self_attn = GemmaAttention(
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
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
