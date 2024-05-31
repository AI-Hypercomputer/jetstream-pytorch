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

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from .config import ModelArgs, find_multiple
from jetstream_pt.layers import Attention, get_quantized_linear_layer, get_quantized_enbedding_layer

import jax

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs, env=None) -> None:
        super().__init__()
        self.config = config
        self.env = env

        Embedding = get_quantized_enbedding_layer(env.quant_config)
        self.tok_embeddings = Embedding(config.vocab_size, config.dim, device=env.device)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        LinearLayer = get_quantized_linear_layer(env.quant_config)
        self.output = LinearLayer(config.dim, config.vocab_size, bias=False, device=env.device)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

        # TODO(Consider refactor with other models)
        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)

    # def setup_caches(self, max_batch_size, max_seq_length):
    #     if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
    #         return
    #     head_dim = self.config.dim // self.config.n_head
    #     max_seq_length = find_multiple(max_seq_length, 8)
    #     self.max_seq_length = max_seq_length
    #     self.max_batch_size = max_batch_size
    #     for b in self.layers:
    #         b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim)

    #     self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
    #     self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor], caches: List[Any], mask, start: Optional[Tensor] = None, ragged_batch_index=None, ragged_block_index=None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        end = None if start is None else (start + input_pos) % self.env.cache_len       
        # mask = self.causal_mask[None, None, input_pos]
        with jax.named_scope("transformer_tok"):
            x = self.tok_embeddings(idx)
        with jax.named_scope("transformer_freq"):
            freqs_cis = self.freqs_cis[input_pos]
        
        assert len(caches) == len(
            self.layers
        ), f"Number of caches ({len(caches)}) and layers ({len(self.layers)}) dont match"

        for i, layer in enumerate(self.layers):
            with jax.named_scope("TransformerBlock"):
                x = layer(x, input_pos, freqs_cis, mask, caches, start, end, ragged_batch_index, ragged_block_index)
        
        with jax.named_scope("transformer_norm"):
            x = self.norm(x)
            logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, env=None) -> None:
        super().__init__()
        self.attention = Attention(config.n_head, config.n_local_heads, config.head_dim, config.dim, env, device=config.device)
        self.block_sparse_moe = MOEFeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, caches: List[Tensor], start=None, end=None, ragged_batch_index=None, ragged_block_index=None) -> Tensor:
        with jax.named_scope("Attention"):
            attn = self.attention(self.attention_norm(x), freqs_cis, mask, caches, start, end, ragged_batch_index, ragged_block_index)
        with jax.named_scope("ffn_norm"):
            h = x + attn
            ffns = self.ffn_norm(h)
        with jax.named_scope("ffn"):
            out = h + self.block_sparse_moe(ffns)
        return out


class ConditionalFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TODO(How to enable quantization?)
        self.w1 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))
        self.w2 = nn.Parameter(torch.empty(config.num_experts, config.dim, config.intermediate_size))
        self.w3 = nn.Parameter(torch.empty(config.num_experts, config.intermediate_size, config.dim))

    def forward(self, x: Tensor, expert_indices: Tensor) -> Tensor:
        with jax.named_scope("conditional_ff"):
            w1_weights = self.w1[expert_indices] # [T, A, D, D]
            w3_weights = self.w3[expert_indices] # [T, A, D, D]
            w2_weights = self.w2[expert_indices]  # [T, A, D, D]
            print("conditional ff: w1_weights.shape: %s x.shape: %s" % (w1_weights.shape, x.shape))
            x1 = F.silu(torch.einsum('ti,taoi -> tao', x, w1_weights))
            x3 = torch.einsum('ti, taoi -> tao', x, w3_weights)
            expert_outs =  torch.einsum('tao, taio -> tai', (x1 * x3), w2_weights)
        return expert_outs


class MOEFeedForward(nn.Module):
    def __init__(self, config, device='meta', env=None) -> None:
        super().__init__()
        LinearLayer = get_quantized_linear_layer(env.quant_config)
        self.gate = LinearLayer(config.dim, config.num_experts, bias=False)
        self.cond_ffn = ConditionalFeedForward(config)
        self.dim = config.dim
        self.num_activated_experts = config.num_activated_experts
    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.dim)
        # T = num_tokens, E = num_experts, D = hidden dim, A = activated experts
        # x: [T, D]
        scores = self.gate(x) # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.num_activated_experts, dim=-1) # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [T, A]
        expert_outs = self.cond_ffn(x, expert_indices)
        return torch.einsum('tai,ta -> ti', expert_outs, expert_weights)


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
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.bfloat16)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)