# pylint: disable-all
# # Copyright 2024 Google LLC
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

from dataclasses import dataclass
from typing import Optional, List, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from .config import ModelArgs, find_multiple
from jetstream_pt.layers import Attention, get_quantized_linear_layer, get_quantized_embedding_layer

import jax


class Transformer(nn.Module):

  def __init__(self, config: ModelArgs, env) -> None:
    super().__init__()
    self.config = config
    self.env = env

    Embedding = get_quantized_embedding_layer(env.quant_config)
    self.tok_embeddings = Embedding(
        config.vocab_size, config.dim, device=config.device
    )
    self.layers = nn.ModuleList(
        TransformerBlock(config, env, layer_id) for layer_id, _ in enumerate(range(config.n_layer))
    )
    self.norm = RMSNorm(config.dim, eps=config.norm_eps)
    LinearLayer = get_quantized_linear_layer(env.quant_config)
    self.output = LinearLayer(
        config.dim, config.vocab_size, bias=False, device=config.device
    )

    self.max_batch_size = -1
    self.max_seq_length = -1

    # TODO(Consider refactor with other models)
    freqs_cis = precompute_freqs_cis(
        self.config.block_size,
        self.config.dim // self.config.n_head,
        self.config.rope_base,
    )
    self.register_buffer("freqs_cis", freqs_cis)

  @torch.no_grad()
  def forward(
      self,
      idx: Tensor,
      input_pos: Optional[Tensor],
      caches: List[Any],
      mask,
      start: Optional[Tensor] = None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ) -> Tensor:
    assert self.freqs_cis is not None, "Caches must be initialized first"
    end = None if start is None else (start + input_pos) % self.env.cache_len
    with jax.named_scope("transformer_tok"):
      x = self.tok_embeddings(idx)
    with jax.named_scope("transformer_freq"):
      bsz, seqlen = idx.shape
      freqs_cis = self.freqs_cis[input_pos]
      freqs_cis = freqs_cis.reshape(bsz, seqlen, -1)
    assert len(caches) == len(
        self.layers
    ), f"Number of caches ({len(caches)}) and layers ({len(self.layers)}) dont match"
    for layer, cache in zip(self.layers, caches):
      with jax.named_scope("TransformerBlock"):
        x = layer(
            x,
            freqs_cis,
            mask,
            cache,
            start,
            end,
            ragged_batch_index,
            ragged_block_index,
        )

    with jax.named_scope("transformer_norm"):
      x = self.norm(x)
      logits = self.output(x)
    return logits

  @staticmethod
  def get_quantized_linear_weight_to_scaler_map():
    return {
        "attention.wq.weight": "attention.wq.weight_scaler",
        "attention.wk.weight": "attention.wk.weight_scaler",
        "attention.wv.weight": "attention.wv.weight_scaler",
        "attention.wo.weight": "attention.wo.weight_scaler",
        "output.weight": "output.weight_scaler",
        "block_sparse_moe.gate.weight": "block_sparse_moe.gate.weight_scaler",
        "block_sparse_moe.cond_ffn.w1": "block_sparse_moe.cond_ffn.w1_scaler",
        "block_sparse_moe.cond_ffn.w2": "block_sparse_moe.cond_ffn.w2_scaler",
        "block_sparse_moe.cond_ffn.w3": "block_sparse_moe.cond_ffn.w3_scaler",
    }

  @staticmethod
  def get_quantized_embedding_weight_to_scaler_map():
    return {
        "tok_embeddings.weight": "tok_embeddings.weight_scaler",
    }

  @staticmethod
  def get_weight_sharding_type():
    # ParallelEmbedding is col partitioned across the shards.
    # ColumnParallelLinear is row partitioned across shards due to transpose.
    # RowParallelLinear is col partitioned across shards due to transpose.
    # None is no partitioning and tensor should be identical across shards
    return {
        "tok_embeddings.weight": "ParallelEmbedding",
        "rope.freqs": None,
        "attention.wq.weight": "ColumnParallelLinear",
        "attention.wk.weight": "ColumnParallelLinear",
        "attention.wv.weight": "ColumnParallelLinear",
        "attention.wo.weight": "RowParallelLinear",
        "feed_forward.w1.weight": "ColumnParallelLinear",
        "feed_forward.w2.weight": "RowParallelLinear",
        "feed_forward.w3.weight": "ColumnParallelLinear",
        "attention_norm.weight": None,
        "ffn_norm.weight": None,
        "norm.weight": None,
        "output.weight": "ColumnParallelLinear",
    }


class TransformerBlock(nn.Module):

  def __init__(self, config: ModelArgs, env, layer_id) -> None:
    super().__init__()
    self.attention = Attention(
        config.n_head,
        config.n_local_heads,
        config.head_dim,
        config.dim,
        env=env,
        device=config.device,
        layer_id=layer_id
    )
    self.block_sparse_moe = MOEFeedForward(config, config.device, env)
    self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
    self.attention_norm = RMSNorm(config.dim, config.norm_eps)

  def forward(
      self,
      x: Tensor,
      freqs_cis: Tensor,
      mask: Tensor,
      caches: List[Tensor],
      start=None,
      end=None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ) -> Tensor:
    with jax.named_scope("Attention"):
      attn = self.attention(
          self.attention_norm(x),
          freqs_cis,
          mask,
          caches,
          start,
          end,
          ragged_batch_index,
          ragged_block_index,
      )
    with jax.named_scope("ffn_norm"):
      h = x + attn
      ffns = self.ffn_norm(h)
    with jax.named_scope("ffn"):
      moe = self.block_sparse_moe(ffns)
      out = h + moe
    return out


class Int8ConditionalFeedForward(nn.Module):

  def __init__(self, config):
    super().__init__()
    w1 = torch.empty(
        config.num_experts,
        config.intermediate_size,
        config.dim,
        dtype=torch.int8,
    )
    w2 = torch.empty(
        config.num_experts,
        config.dim,
        config.intermediate_size,
        dtype=torch.int8,
    )
    w3 = torch.empty(
        config.num_experts,
        config.intermediate_size,
        config.dim,
        dtype=torch.int8,
    )
    self.register_buffer("w1", w1)
    self.register_buffer("w2", w2)
    self.register_buffer("w3", w3)

    w1_scaler = torch.empty(config.num_experts, config.intermediate_size)
    w2_scaler = torch.empty(config.num_experts, config.dim)
    w3_scaler = torch.empty(config.num_experts, config.intermediate_size)
    self.register_buffer("w1_scaler", w1_scaler)
    self.register_buffer("w2_scaler", w2_scaler)
    self.register_buffer("w3_scaler", w3_scaler)

  def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
    seq_len = x.shape[0]
    if seq_len >= 4:
      return self.forward_for_long_seq_len(x, expert_indices)
    else:
      return self.forward_for_short_seq_len(x, expert_indices)

  def forward_for_short_seq_len(
      self, x: Tensor, expert_indices: Tensor
  ) -> Tensor:
    with jax.named_scope("conditional_ff"):
      w1_weights = self.w1[expert_indices]  # [T, A, D, D]
      w3_weights = self.w3[expert_indices]  # [T, A, D, D]
      w2_weights = self.w2[expert_indices]  # [T, A, D, D]
      w1_scaler = self.w1_scaler[expert_indices]
      w2_scaler = self.w2_scaler[expert_indices]
      w3_scaler = self.w3_scaler[expert_indices]

      x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights) * w1_scaler)
      x3 = torch.einsum("ti, taoi -> tao", x, w3_weights) * w3_scaler
      expert_outs = (
          torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights) * w2_scaler
      )
    return expert_outs

  def forward_for_long_seq_len(self, x, expert_indices):
    seqlen = x.shape[0]
    num_experts = self.w1.shape[0]

    # e = total num of exp = 8
    # t = seqlen
    # o = config.imtermediate size
    # i = config.dim
    with jax.named_scope("conditional_ff"):
      x1 = F.silu(torch.einsum("ti,eoi -> teo", x, self.w1) * self.w1_scaler)
      x3 = torch.einsum("ti, eoi-> teo", x, self.w3) * self.w3_scaler
      expert_outs = (
          torch.einsum("teo, eio -> tei", (x1 * x3), self.w2) * self.w2_scaler
      )
      # e = 8; need to reduce to 2
      seq_indexes = torch.arange(seqlen).unsqueeze(1)
      return expert_outs[seq_indexes, expert_indices]


class ConditionalFeedForward(nn.Module):

  def __init__(self, config):
    super().__init__()
    # TODO(How to enable quantization?)
    self.w1 = nn.Parameter(
        torch.empty(config.num_experts, config.intermediate_size, config.dim)
    )
    self.w2 = nn.Parameter(
        torch.empty(config.num_experts, config.dim, config.intermediate_size)
    )
    self.w3 = nn.Parameter(
        torch.empty(config.num_experts, config.intermediate_size, config.dim)
    )

  def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
    seq_len = x.shape[0]
    if seq_len >= 4:
      return self.forward_for_long_seq_len(x, expert_indices)
    else:
      return self.forward_for_short_seq_len(x, expert_indices)

  def forward_for_short_seq_len(
      self, x: Tensor, expert_indices: Tensor
  ) -> Tensor:
    with jax.named_scope("conditional_ff"):
      w1_weights = self.w1[expert_indices]  # [T, A, D, D]
      w3_weights = self.w3[expert_indices]  # [T, A, D, D]
      w2_weights = self.w2[expert_indices]  # [T, A, D, D]

      x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
      x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
      expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)
    return expert_outs

  def forward_for_long_seq_len(self, x, expert_indices):
    seqlen = x.shape[0]
    num_experts = self.w1.shape[0]

    # e = total num of exp = 8
    # t = seqlen
    # o = config.imtermediate size
    # i = config.dim
    with jax.named_scope("conditional_ff"):
      x1 = F.silu(torch.einsum("ti,eoi -> teo", x, self.w1))
      x3 = torch.einsum("ti, eoi-> teo", x, self.w3)
      expert_outs = torch.einsum("teo, eio -> tei", (x1 * x3), self.w2)
      # e = 8; need to reduce to 2
      seq_indexes = torch.arange(seqlen).unsqueeze(1)
      return expert_outs[seq_indexes, expert_indices]


class MOEFeedForward(nn.Module):

  def __init__(self, config, device, env) -> None:
    super().__init__()
    LinearLayer = get_quantized_linear_layer(env.quant_config)
    self.gate = LinearLayer(config.dim, config.num_experts, bias=False)
    CondLayer = (
        Int8ConditionalFeedForward
        if env.quant_config.enable_weight_quantization
        else ConditionalFeedForward
    )
    self.cond_ffn = CondLayer(config)
    self.dim = config.dim
    self.num_activated_experts = config.num_activated_experts

  def forward(self, x: Tensor) -> Tensor:
    bsz, seq, hidden = x.shape
    # [B, T, D], combine BT, for prefill B = 1, for decode, T = 1
    x = x.view(-1, self.dim)
    # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
    # x: [T, D]
    scores = self.gate(x)  # [T, E]
    expert_weights = F.softmax(scores, dim=-1)
    expert_weights, expert_indices = torch.topk(
        expert_weights, self.num_activated_experts, dim=-1
    )  # [T, A], [T, A]
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True)  # [T, A]
    expert_outs = self.cond_ffn(x, expert_indices)
    expert_outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)
    # Changes back to [B, T, D]
    expert_outs = expert_outs.reshape(bsz, seq, hidden)
    return expert_outs


class RMSNorm(nn.Module):

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def _norm(self, x):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

  def forward(self, x: Tensor) -> Tensor:
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
  freqs = 1.0 / (
      base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
  )
  t = torch.arange(seq_len)
  freqs = torch.outer(t, freqs)
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
  return freqs_cis
