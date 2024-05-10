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

# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
from jax import lax
import torch_xla2

class Int8Embedding(torch.nn.Module):

  def __init__(self, num_embeddings, embedding_dims, device="cpu"):
    super().__init__()
    table = torch.ones(
        (num_embeddings, embedding_dims), device=device, dtype=torch.int8
    )
    self.register_buffer("weight", table)
    embedding_scaler = torch.ones(
        (embedding_dims,), device=device, dtype=torch.bfloat16
    )
    self.register_buffer("weight_scaler", embedding_scaler)

  def forward(self, input):
    return F.embedding(input, self.weight) * self.weight_scaler


class WeightOnlyPerChannelQuantizedLinear(torch.nn.Module):

  def __init__(self, in_features, out_features, bias=False, device=None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    weight = torch.ones(
      (out_features, in_features), dtype=torch.int8, device=device
    )
    self.register_buffer("weight", weight)

    weight_scaler = torch.ones(
        (out_features,), dtype=torch.bfloat16, device=device
    )
    self.register_buffer("weight_scaler", weight_scaler)

    assert not bias, "Quantized Linear doesn't support bias."

  def forward(self, inputs):
    return F.linear(inputs, self.weight) * self.weight_scaler


class WeightOnlyBlockwiseQuantizedLinear(torch.nn.Module):

  def __init__(self, in_features, out_features, bias, device):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    # Use dot general instead of einsum
    # Use dot general is slow now.
    self.use_dot_general = False
    # Flatten einsum operands to 3D.
    # Same perf as non flattened one.
    self.flatten = True

    block_size = 128
    n_blocks = in_features // block_size

    if self.use_dot_general:
      weight = torch.ones(
          (n_blocks, out_features, block_size), dtype=torch.int8, device=device
      )
    else:
      weight = torch.ones(
          (n_blocks, block_size, out_features), dtype=torch.int8, device=device
      )
    self.register_buffer("weight", weight)

    weight_scaler = torch.ones(
        (n_blocks, out_features), dtype=torch.bfloat16, device=device
    )
    self.register_buffer("weight_scaler", weight_scaler)

  def forward(self, inputs):

    if self.use_dot_general:
      j_inputs = inputs._elem
      j_weight = self.weight._elem
      j_weight_scaler = self.weight_scaler._elem
      j_inputs_shape = j_inputs.shape
      block_size = j_weight.shape[2]
      bs = j_inputs_shape[0]
      j_inputs_new_shape = j_inputs_shape[:-1] + (
          j_inputs_shape[-1] // block_size,
          block_size,
      )
      j_inputs = j_inputs.reshape(j_inputs_new_shape)
      j_inputs = jax.lax.collapse(j_inputs, 0, 2)
      out = jax.lax.dot_general(
          j_inputs, j_weight, dimension_numbers=([(2), (2)], [(1), (0)])
      )
      out = jax.lax.dot_general(
          out, j_weight_scaler, dimension_numbers=([(0), (0)], [(2), (1)])
      )
      out = jax.lax.transpose(out, [1, 0])
      out = out.reshape((bs, -1) + out.shape[1:])
      return torch_xla2.tensor.XLATensor2(out)
    if self.flatten:
      j_inputs = inputs._elem
      j_weight = self.weight._elem.astype(jnp.int8)
      j_weight_scaler = self.weight_scaler._elem
      block_size = j_weight.shape[1]
      j_inputs_shape = j_inputs.shape
      bs = j_inputs_shape[0]
      j_inputs_new_shape = j_inputs_shape[:-1] + (
          j_inputs_shape[-1] // block_size,
          block_size,
      )
      j_inputs = j_inputs.reshape(j_inputs_new_shape)
      j_inputs = jax.lax.collapse(j_inputs, 0, 2)
      out = jnp.einsum("scz,bsc->bsz", j_weight, j_inputs)
      out = jnp.einsum("bsz,sz->bz", out, j_weight_scaler)
      out = out.reshape((bs, -1) + out.shape[1:])
      return torch_xla2.tensor.XLATensor2(out)
    else:
      j_inputs = inputs._elem
      j_weight = self.weight._elem.astype(jnp.int8)
      j_weight_scaler = self.weight_scaler._elem
      block_size = j_weight.shape[1]
      j_inputs_shape = j_inputs.shape
      j_inputs_new_shape = j_inputs_shape[:-1] + (
          j_inputs_shape[-1] // block_size,
          block_size,
      )
      j_inputs = j_inputs.reshape(j_inputs_new_shape)
      out = jnp.einsum("scz,bdsc->bdsz", j_weight, j_inputs)
      out = jnp.einsum("bdsz,sz->bdz", out, j_weight_scaler)
      return torch_xla2.tensor.XLATensor2(out)


def get_quantized_linear_layer(config: "QuantizationConfig"):
  if not config.enable_weight_quantization:
    return nn.Linear
  if config.is_blockwise_weight:
    return WeightOnlyBlockwiseQuantizedLinear
  else:
    return WeightOnlyPerChannelQuantizedLinear


class RMSNorm(torch.nn.Module):
  """RMSNorm module."""

  def __init__(self, dim: int, eps: float = 1e-6, device="meta"):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, device=device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (
      x.shape[0],
      x.shape[-3],
      x.shape[-1],
  ), f"freqs_cis: {freqs_cis.shape }, x: {x.shape}"
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  shape[0] = x.shape[0]  # batch size
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # bs, seqlen, heads, dim
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""

  bs, n_kv_heads, slen, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:, :, None, :, :]
      .expand(bs, n_kv_heads, n_rep, slen, head_dim)
      .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
  )


class AttentionKernel:

  def __init__(self, env):
    self.env = env

  def __call__(self, xq, xk, xv, mask, cache):
    """
    Args:
      xq: torch.Tensor of (batch size, num_heads, seqlen, head_dim)
      xk: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      xv: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      mask: mask with 0 and -inf, or None
      cache: CacheManagerInterface object
    """
    bsz, num_heads, seqlen, head_dim = xq.shape
    _, num_kv_heads, _, kv_head_dim = xk.shape
    n_rep = num_heads // num_kv_heads
    if seqlen == 1:
      xq = torch.broadcast_to(xq, (xq.shape[0], xq.shape[1], 2, xq.shape[3]))

    with jax.named_scope("attn_insert_cache"):
      keys, values = cache.update(xk, xv)
      keys = repeat_kv(keys, n_rep)
      values = repeat_kv(values, n_rep)
    with jax.named_scope("attn_mat1"):
      ## Attention start
      # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
      scores = torch_xla2.extra.call_jax(
          jnp.einsum, "ikjl,ikml->ikjm", xq, keys
      ) / math.sqrt(head_dim)
      self.env.apply_sharding(scores, axis=1)
      if mask is not None:
        # if mask.shape != (1,1,16,16):
        #   breakpoint()
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    with jax.named_scope("attn_soft"):
      scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    with jax.named_scope("attn_mat2"):
      # output = torch.einsum(
      #    "ikjm,ikml->ikjl", scores, values
      # )  # (bs, n_local_heads, seqlen, head_dim)
      output = torch_xla2.extra.call_jax(
          jnp.einsum, "ikjm,ikml->ikjl", scores, values
      )
      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # For XLA matmul performance boost
      # output = torch.matmul(scores, values)
      self.env.apply_sharding(output, axis=1)
      return output


class Int8KVAttentionKernel:

  def __init__(self, env):
    self.env = env

  def __call__(self, xq, xk, xv, mask, cache):
    """
    Args:
      xq: torch.Tensor of (batch size, num_heads, seqlen, head_dim)
      xk: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      xv: torch.Tensor of (batch size, num_kv_heads, seqlen, head_dim)
      mask: mask with 0 and -inf, or None
      cache: CacheManagerInterface object
    """
    bsz, num_heads, seqlen, head_dim = xq.shape
    _, num_kv_heads, _, kv_head_dim = xk.shape
    n_rep = num_heads // num_kv_heads
    if seqlen == 1:
      xq = torch.broadcast_to(xq, (xq.shape[0], xq.shape[1], 2, xq.shape[3]))

    with jax.named_scope("attn_insert_cache"):
      keys, values, k_scaler, v_scaler = cache.update(xk, xv)
      keys = repeat_kv(keys, n_rep)
      values = repeat_kv(values, n_rep)
    with jax.named_scope("attn_mat1"):
      ## Attention start
      # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
      scores = (
          torch_xla2.extra.call_jax(jnp.einsum, "ikjl,ikml->ikjm", xq, keys)
          / math.sqrt(head_dim)
          * (k_scaler.reshape(bsz, 1, 1, keys.shape[2]))
      )
      self.env.apply_sharding(scores, axis=1)
      if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    with jax.named_scope("attn_soft"):
      scores = F.softmax(scores.float(), dim=-1).type_as(xq)
      scores = scores * v_scaler.reshape((bsz, 1, 1, keys.shape[2]))
      self.env.apply_sharding(scores, axis=1)

    with jax.named_scope("attn_mat2"):
      # output = torch.einsum(
      #    "ikjm,ikml->ikjl", scores, values
      # )  # (bs, n_local_heads, seqlen, head_dim)
      output = torch_xla2.extra.call_jax(
          jnp.einsum, "ikjm,ikml->ikjl", scores, values
      )
      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # output = torch.matmul(scores, values)
      self.env.apply_sharding(output, axis=1)
      return output


class Attention(nn.Module):
  """Attention module."""

  def __init__(self, n_heads, n_kv_heads, head_dim, hidden_size, device, env):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = head_dim
    self.n_rep = self.n_heads // self.n_kv_heads
    self.env = env
    self.hidden_size = hidden_size

    LinearLayer = get_quantized_linear_layer(env.quant_config)

    self.wo = LinearLayer(
        n_heads * self.head_dim,
        hidden_size,
        bias=False,
        device=device,
    )

    Kernel = (
        Int8KVAttentionKernel
        if env.quant_config.enable_kv_quantization
        else AttentionKernel
    )
    self.attention_kernel = Kernel(env)

    self.q_size = n_heads * self.head_dim
    self.kv_size = self.n_kv_heads * self.head_dim
    if self.env.qkv_fusion:
      self._register_load_state_dict_pre_hook(self.load_hook)
      self.wqkv = LinearLayer(
          hidden_size,
          (n_heads + 2 * self.n_kv_heads) * self.head_dim,
          bias=False,
          device=device,
      )
    else:
      self.wq = LinearLayer(
          hidden_size,
          n_heads * self.head_dim,
          bias=False,
          device=device,
      )
      self.wk = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=False,
          device=device,
      )
      self.wv = LinearLayer(
          hidden_size,
          self.n_kv_heads * self.head_dim,
          bias=False,
          device=device,
      )

  def load_hook(self, state_dict, prefix, *args):
    if prefix + "wq.weight" in state_dict:
      wq = state_dict.pop(prefix + "wq.weight")
      wk = state_dict.pop(prefix + "wk.weight")
      wv = state_dict.pop(prefix + "wv.weight")
      state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
  ):
    # print(f"check x shape: {x.shape}, dtype {x.dtype}")
    # print(f"check freqs_cis shape: {freqs_cis.shape}, dtype {freqs_cis.dtype}")
    # if mask is not None:
    #   print(f"check mask shape: {mask.shape}, dtype {mask.dtype}")
    # print(f"check x shape: {x.shape}, dtype {x.dtype}")
    # bsz, seqlen, _ = x.shape
    with jax.named_scope("attn_linear_before_cache"):
      bsz, seqlen = x.shape[0], x.shape[-2]

      # qkv fuse
      if self.env.qkv_fusion:
        xq, xk, xv = self.wqkv(x).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
      else:
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
      xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
      xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

      self.env.apply_sharding(xq, axis=2)
      self.env.apply_sharding(xk, axis=2)
      self.env.apply_sharding(xv, axis=2)

    with jax.named_scope("attn_rope"):
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    xq = xq.transpose(1, 2)

    output = self.attention_kernel(xq, xk, xv, mask, cache)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
