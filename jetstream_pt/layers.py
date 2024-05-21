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

import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import torch_xla2
from jax import lax
from jetstream_pt import torchjax
from jetstream_pt.quantize import (
    dequantize_tensor,
    load_q_weight_helper,
    quantize_tensor,
)
from torch import nn


def _calc_cosine_dist(x, y):
  x = x.flatten().to(torch.float32)
  y = y.flatten().to(torch.float32)
  return (torch.dot(x, y) / (x.norm() * y.norm())).item()


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

  def __init__(
      self,
      in_features,
      out_features,
      bias=False,
      device=None,
      is_symmetric=True,
      n_bit=8,
  ):
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

    self.is_symmetric = is_symmetric
    if not is_symmetric:
      zero_point = torch.ones(
          (out_features,), dtype=torch.bfloat16, device=device
      )
      self.register_buffer("zero_point", zero_point)
    else:
      self.register_buffer("zero_point", None)

    assert not bias, "Quantized Linear doesn't support bias."

    self.n_bit = n_bit
    # Flag to enable dequantize weight first, then do matmul. Useful for debugging.
    self.run_fake_quantize = False

  def _load_quantized_weights(self, w_q, scale, zp=None):
    """
    Load weights quantized by 'quantize_tensor'.
    """
    self.weight, self.weight_scaler, self.zero_point = load_q_weight_helper(
        w_q, scale, zp, block_size=-1
    )

  def quantize_weight_from_nn_linear(self, weight):
    assert weight.dim() == 2, "Expect 2D weight from torch.nn.Linear."
    assert weight.shape == (
        self.out_features,
        self.in_features,
    ), f"Got unexpected weight of shape {weight.shape}, expected weight shape ({self.out_features}, {self.in_features})."
    w_q, scale, zp = quantize_tensor(
        weight, (1,), self.n_bit, self.is_symmetric, block_size=-1
    )
    w_dq = dequantize_tensor(w_q, scale, zp)
    self._load_quantized_weights(w_q, scale, zp)

  def forward(self, inputs):
    if self.is_symmetric:
      return torch.mul(F.linear(inputs, self.weight), self.weight_scaler)
    else:
      out = torch.mul(F.linear(inputs, self.weight), self.weight_scaler)
      zp_out = torch.einsum("...c,z->...z", inputs, self.zero_point)
      return out - zp_out

    if self.run_fake_quantize:
      scaler = self.weight_scaler.unsqueeze(-1)
      if not self.is_symmetric:
        zero_point = self.zero_point.unsqueeze(-1) / scaler
      else:
        zero_point = None
      w_dequantized = dequantize_tensor(
          self.weight.to(torch.bfloat16), scaler, zero_point
      )
      return F.linear(inputs, w_dequantized)


class WeightOnlyBlockwiseQuantizedLinear(torch.nn.Module):

  def __init__(
      self,
      in_features,
      out_features,
      bias=False,
      device=None,
      is_symmetric=True,
      use_dot_general=False,
      block_size=128,
      n_bit=8,
  ):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    # Use dot general instead of einsum
    # Use dot general is slow now.
    self.use_dot_general = use_dot_general
    # Flatten einsum operands to 3D. XLA was slow if operands are 4D. But it's fixed now.
    # Same perf as non flattened one now.
    self.flatten = False

    self.block_size = block_size
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

    self.is_symmetric = is_symmetric
    if not self.is_symmetric:
      zero_point = torch.ones(
          (n_blocks, out_features), dtype=torch.bfloat16, device=device
      )
      self.register_buffer("zero_point", zero_point)
    else:
      self.register_buffer("zero_point", None)

    self.n_bit = n_bit
    # Flag to enable dequantize weight first, then do matmul. Useful for debugging.
    self.run_fake_quantize = False

  def _load_quantized_weights(self, w_q, scale, zp=None):
    """
    Load weights quantized by 'quantize_tensor'.'
    """
    self.weight, self.weight_scaler, self.zero_point = load_q_weight_helper(
        w_q, scale, zp, self.block_size
    )

  def quantize_weight_from_nn_linear(self, weight):
    assert weight.dim() == 2, "Expect 2D weight from torch.nn.Linear."
    assert weight.shape == (
        self.out_features,
        self.in_features,
    ), f"Unexpected weight shape ({self.out_features}, {self.in_features})."
    w_q, scale, zp = quantize_tensor(
        weight, (1,), self.n_bit, self.is_symmetric, self.block_size
    )
    w_dq = dequantize_tensor(w_q, scale, zp)
    print("check qweight cosine dist: ", _calc_cosine_dist(weight, w_dq))
    # breakpoint()
    self._load_quantized_weights(w_q, scale, zp)

  @staticmethod
  def blockwise_jax_kernel(j_inputs, j_weight, j_weight_scaler, j_zero_point):
    j_weight = j_weight.astype(jnp.int8)
    block_size = j_weight.shape[1]
    j_inputs_shape = j_inputs.shape
    j_inputs_new_shape = j_inputs_shape[:-1] + (
        j_inputs_shape[-1] // block_size,
        block_size,
    )
    j_inputs = j_inputs.reshape(j_inputs_new_shape)
    out = jnp.einsum("scz,bdsc->bdsz", j_weight, j_inputs)
    out = jnp.einsum("bdsz,sz->bdz", out, j_weight_scaler)
    if j_zero_point is not None:
      zp_out = jnp.einsum("bdsc,sz->bdz", j_inputs, j_zero_point)
      out = out - zp_out
    return out

  @staticmethod
  def blockwise_jax_kernel_dot_general(
      j_inputs, j_weight, j_weight_scaler, j_zero_point
  ):
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
    return out

  @staticmethod
  def blockwise_jax_kernel_dot_general(
      j_inputs, j_weight, j_weight_scaler, j_zero_point
  ):
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
    return out

  @staticmethod
  def blockwise_jax_kernel_einsum_flatten(
      j_inputs, j_weight, j_weight_scaler, j_zero_point
  ):
    j_weight = j_weight.astype(jnp.int8)
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
    return out

  def forward(self, inputs):
    # bsz, seqlen, _ = x.shape
    if self.use_dot_general:
      assert (
          self.zero_point is None
      ), "Blockwise quantized linear doesn't support zero_point in dot_general implementation."
      return torchjax.call_jax(
          WeightOnlyBlockwiseQuantizedLinear.blockwise_jax_kernel_dot_general,
          inputs,
          self.weight,
          self.weight_scaler,
          self.zero_point,
      )
    if self.flatten:
      assert (
          self.zero_point is None
      ), "Blockwise quantized linear doesn't support zero_point in einsum (flattened) implementation."
      return torchjax.call_jax(
          WeightOnlyBlockwiseQuantizedLinear.blockwise_jax_kernel_einsum_flatten,
          inputs,
          self.weight,
          self.weight_scaler,
          self.zero_point,
      )
    else:
      return torchjax.call_jax(
          WeightOnlyBlockwiseQuantizedLinear.blockwise_jax_kernel,
          inputs,
          self.weight,
          self.weight_scaler,
          self.zero_point,
      )

    if self.run_fake_quantize:
      weight = self.weight.permute(2, 0, 1).to(torch.bfloat16)
      scaler = self.weight_scaler.unsqueeze(-1).transpose(1, 0)
      if not self.is_symmetric:
        zero_point = self.zero_point.unsqueeze(-1).transpose(1, 0) / scaler
      else:
        zero_point = None
      w_dequantized = dequantize_tensor(
          self.weight, scaler, zero_point, block_size=128
      )
      w_dequantized = w_dequantized.reshape(w_dequantized.shape[0], -1)
      return F.linear(inputs, w_dequantized)


def get_quantized_linear_layer(config: "QuantizationConfig"):
  if not config.enable_weight_quantization:
    return nn.Linear
  if config.is_blockwise_weight:
    return WeightOnlyBlockwiseQuantizedLinear
  else:
    return WeightOnlyPerChannelQuantizedLinear


def get_quantized_enbedding_layer(config: "QuantizationConfig"):
  if not config.enable_weight_quantization:
    return nn.Embedding
  else:
    return Int8Embedding


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
      scores = torch.einsum("ikjl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)
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
      output = torch.einsum("ikjm,ikml->ikjl", scores, values)
      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # For XLA matmul performance boost
      # output = torch.matmul(scores, values)
      shard_axis = 0 if self.env.shard_on_batch else 1
      self.env.apply_sharding(output, axis=shard_axis)
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
          torch.einsum("ikjl,ikml->ikjm", xq, keys)
          / math.sqrt(head_dim)
          * (k_scaler.reshape(bsz, 1, 1, keys.shape[2]))
      )
      if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    with jax.named_scope("attn_soft"):
      scores = F.softmax(scores.float(), dim=-1).type_as(xq)
      scores = scores * v_scaler.reshape((bsz, 1, 1, keys.shape[2]))

    with jax.named_scope("attn_mat2"):
      # output = torch.einsum(
      #    "ikjm,ikml->ikjl", scores, values
      # )  # (bs, n_local_heads, seqlen, head_dim)
      output = torch.einsum("ikjm,ikml->ikjl", scores, values)
      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # output = torch.matmul(scores, values)
      shard_axis = 0 if self.env.shard_on_batch else 1
      self.env.apply_sharding(output, axis=shard_axis)
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

      shard_axis = 0 if self.env.shard_on_batch else 2
      self.env.apply_sharding(xq, axis=shard_axis)
      self.env.apply_sharding(xk, axis=shard_axis)
      self.env.apply_sharding(xv, axis=shard_axis)

    with jax.named_scope("attn_rope"):
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    xk = xk.transpose(1, 2)
    xv = xv.transpose(1, 2)
    xq = xq.transpose(1, 2)

    output = self.attention_kernel(xq, xk, xv, mask, cache)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
