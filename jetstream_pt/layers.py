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
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
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

import numpy as np

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
    if not self.run_fake_quantize:
      if self.is_symmetric:
        return torch.mul(F.linear(inputs, self.weight), self.weight_scaler)
      else:
        out = torch.mul(F.linear(inputs, self.weight), self.weight_scaler)
        zp_out = torch.einsum("...c,z->...z", inputs, self.zero_point)
        return out - zp_out
    else:
      # Fake quantization, debugging purpose.
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
  def blockwise_jax_kernel(inputs, weight, weight_scaler, zero_point):
    """Blockwise Matmul kernel impl in JAX using einsum"""
    weight = weight.astype(jnp.int8)
    block_size = weight.shape[1]
    inputs_shape = inputs.shape
    inputs_new_shape = inputs_shape[:-1] + (
        inputs_shape[-1] // block_size,
        block_size,
    )
    inputs = inputs.reshape(inputs_new_shape)
    out = jnp.einsum("scz,bdsc->bdsz", weight, inputs)
    out = jnp.einsum("bdsz,sz->bdz", out, weight_scaler)
    if zero_point is not None:
      zp_out = jnp.einsum("bdsc,sz->bdz", inputs, zero_point)
      out = out - zp_out
    return out

  @staticmethod
  def blockwise_jax_kernel_dot_general(
      inputs, weight, weight_scaler, zero_point
  ):
    """Blockwise Matmul kernel impl in JAX using dot general"""
    inputs_shape = inputs.shape
    block_size = weight.shape[2]
    bs = inputs_shape[0]
    inputs_new_shape = inputs_shape[:-1] + (
        inputs_shape[-1] // block_size,
        block_size,
    )
    inputs = inputs.reshape(inputs_new_shape)
    inputs = jax.lax.collapse(inputs, 0, 2)
    out = jax.lax.dot_general(
        inputs, weight, dimension_numbers=([(2), (2)], [(1), (0)])
    )
    out = jax.lax.dot_general(
        out, weight_scaler, dimension_numbers=([(0), (0)], [(2), (1)])
    )
    out = jax.lax.transpose(out, [1, 0])
    out = out.reshape((bs, -1) + out.shape[1:])
    return out

  @staticmethod
  def blockwise_jax_kernel_einsum_flatten(
      inputs, weight, weight_scaler, zero_point
  ):
    """Blockwise Matmul kernel impl in JAX using einsum, with operands flattened"""
    weight = weight.astype(jnp.int8)
    block_size = weight.shape[1]
    inputs_shape = inputs.shape
    bs = inputs_shape[0]
    inputs_new_shape = inputs_shape[:-1] + (
        inputs_shape[-1] // block_size,
        block_size,
    )
    inputs = inputs.reshape(inputs_new_shape)
    inputs = jax.lax.collapse(inputs, 0, 2)
    out = jnp.einsum("scz,bsc->bsz", weight, inputs)
    out = jnp.einsum("bsz,sz->bz", out, weight_scaler)
    out = out.reshape((bs, -1) + out.shape[1:])
    return out

  def forward(self, inputs):
    if not self.run_fake_quantize:
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
    else:
      # Fake quantization, debugging purpose.
      weight = self.weight.permute(2, 0, 1).to(torch.bfloat16)
      scaler = self.weight_scaler.unsqueeze(-1).transpose(1, 0)
      if not self.is_symmetric:
        zero_point = self.zero_point.unsqueeze(-1).transpose(1, 0) / scaler
      else:
        zero_point = None
      w_dequantized = dequantize_tensor(self.weight, scaler, zero_point)
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

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

def ragged_flash_attention_kernel(
    start_ref,
    end_ref,
    line_end_ref,
    pre_b_ref,
    pre_i_ref,
    q_ref,
    k_ref,
    v_ref,
    k_scaler_ref,
    v_scaler_ref,
    o_ref,
    m_ref,
    l_ref,
    bk: int,
    mask_value: float,
    normalize_var: bool,
    quantized: bool,
):
  """Pallas kernel for flash attention."""
  with jax.named_scope("attention_kernel"):
      b, i = pl.program_id(0), pl.program_id(1)

      @pl.when(i == 0)
      def init():
        with jax.named_scope("init"):
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

      length = line_end_ref[b]
      start = start_ref[b]
      end = end_ref[b]

      @pl.when(jnp.logical_and(i * bk < length, start != end))
      def run():
        with jax.named_scope("run_qk"):
            q = q_ref[...].astype(jnp.float32)
            k = k_ref[...].astype(jnp.float32)
            v = v_ref[...].astype(jnp.float32)
            m_prev, l_prev = m_ref[...], l_ref[...]

            qk = jax.lax.dot_general(
                q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
            )
            if normalize_var:
              qk = qk / jnp.sqrt(k.shape[-1])
            if quantized:
              qk = qk * k_scaler_ref[...]
        with jax.named_scope("run_mask"):
            start = start_ref[b]
            end = end_ref[b]
            iota = jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
            mask_start_lt_end = jnp.logical_and(i * bk + iota >= start, i * bk + iota < end).astype(jnp.int32)
            mask_start_gt_end = jnp.logical_or(i * bk + iota >= start, i * bk + iota < end).astype(jnp.int32)
            #mask = jax.lax.cond(start <= end, lambda: mask_start_lt_end, lambda: mask_start_gt_end)
            mask = jnp.where(start <= end, mask_start_lt_end, mask_start_gt_end)

            qk = qk + jnp.where(mask, 0.0, mask_value)

        with jax.named_scope("run_softmax"):
            m_curr = qk.max(axis=-1)

            s_curr = jnp.exp(qk - m_curr[..., None])

            l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
            if quantized:
              s_curr = s_curr * v_scaler_ref[...]
            o_curr_times_l_curr = jnp.dot(s_curr, v)
            m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
            m_next = jnp.maximum(m_prev, m_curr)
            alpha = jnp.exp(m_prev - m_next)
            beta = jnp.exp(m_curr - m_next)
            l_next = alpha * l_prev + beta * l_curr
            l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

            m_ref[...], l_ref[...] = m_next, l_next_safe
            o_ref[...] = (
                (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
            ).astype(o_ref.dtype)

@functools.partial(jax.jit, static_argnames=["bk", "mask_value", "normalize_var"])
def ragged_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    start: jax.Array,
    end: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    ragged_batch_index = None,
    ragged_block_index = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  with jax.named_scope("ragged_mqa"):
      batch_size, num_heads, head_dim = q.shape 
      seq_len = k.shape[1]  

      def kv_index_map(b, i, start_ref, end_ref, line_end_ref, ragged_batch_index_ref, ragged_block_index_ref):
        index = b * (seq_len // bk) + i
        return ragged_batch_index_ref[index], ragged_block_index_ref[index], 0

      def q_index_map(b, i, start_ref, end_ref, line_end_ref, ragged_batch_index_ref, ragged_block_index_ref):
        index = b * (seq_len // bk) + i
        return ragged_batch_index_ref[index], 0, 0

      def scaler_index_map(b, i, *_):
        return b, 0, i

      line_end = jnp.where(start < end, end, seq_len - 1)


      if k_scaler is not None:
          out, m, l = pl.pallas_call(
              functools.partial(
                  ragged_flash_attention_kernel,
                  bk=bk,
                  mask_value=mask_value,
                  normalize_var=normalize_var,
                  quantized=False,
              ),
              grid_spec=pltpu.PrefetchScalarGridSpec(
                  num_scalar_prefetch=5,
                  in_specs=[
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                      pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                      pl.BlockSpec(scaler_index_map, (None, 1, bk)),
                      pl.BlockSpec(scaler_index_map, (None, 1, bk)),
                  ],
                  out_specs=[
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                      pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                  ],
                  grid=(batch_size, seq_len // bk),
              ),
              compiler_params=dict(dimension_semantics=("parallel", "arbitrary")),
              out_shape=[
                  q,
                  jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
                  jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
              ],
          )(start, end, line_end, ragged_batch_index, ragged_block_index, q, k, v, k_scaler, v_scaler)
      else:
        out, m, l = pl.pallas_call(
          functools.partial(
              ragged_flash_attention_kernel,
              bk=bk,
              mask_value=mask_value,
              normalize_var=normalize_var,
              quantized=True,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=5,
              in_specs=[
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
                pl.BlockSpec(kv_index_map, (None, bk, head_dim)),
              ],
              out_specs=[
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
                pl.BlockSpec(q_index_map, (None, num_heads, head_dim)),
              ],
              grid=(batch_size, seq_len // bk),
          ),
          compiler_params=dict(dimension_semantics=("parallel", "arbitrary")),
          out_shape=[
              q,
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
              jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          ],
        )(start, end, line_end, ragged_batch_index, ragged_block_index, q, k, v)
  return out, (m[..., 0], l[..., 0])


@functools.partial(jax.jit, static_argnames=['bk', 'mask_value', 'normalize_var', 'shard_axis'])
def ragged_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    start: jax.Array,
    end: jax.Array,
    ragged_batch_index: jax.Array,
    ragged_block_index: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    bk: int = 512,
    mask_value : float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
    shard_axis: int = 1
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi head attention.
  Args:
    q: A [batch_size, compute_dim, num_heads, head_dim] jax.Array.
    k: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    v: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    start: A i32[batch_size] jax.Array
    end: A i32[batch_size] jax.Array
    bk: An integer that is the sequence block size.
    logit_cap: An optional float that caps logits via tanh. By default there is
      no logit capping.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    out_dtype: An optional dtype for the output. If not provided, the output
      dtype will be q's dtype.
  Returns:
    The output of attention([batch_size, num_heads, compute_dim, head_dim]),
    along with the max logit ([batch_size, num_heads, compute_dim, 1]) and
    softmax denominator ([batch_size, num_heads, compute_dim, 1]).
  """
  mask_value = DEFAULT_MASK_VALUE
  seqlen = q.shape[-2]
  if k_scaler is None:
    replicated_in_axes = 4
    replicated_inputs = (ragged_batch_index, ragged_block_index)
  else:
    replicated_in_axes = 6
    replicated_inputs = (jnp.squeeze(k_scaler, -1), jnp.squeeze(v_scaler, -1), ragged_batch_index, ragged_block_index)

  with jax.named_scope("ragged_mha_vmap"):
    out, (m, l) = jax.vmap(
      functools.partial(
          ragged_mqa,
          bk=bk,
          mask_value=mask_value,
          normalize_var=normalize_var,
          #out_dtype=out_dtype,
      ),
      in_axes=(shard_axis, shard_axis, shard_axis, *([None]*replicated_in_axes)),
      out_axes=shard_axis,
    )(q, k, v, start, end, *replicated_inputs)
  return out, (m, l)


def dense_attention(xq, keys, values, k_scaler=None, v_scaler=None, mask=None):
  bsz, _, _, head_dim = xq.shape
  with jax.named_scope("attn_mat1"):
      ## Attention start
      # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
      scores = torch.einsum("ikjl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)
      if k_scaler is not None:
        scores = scores * (k_scaler.reshape(bsz, 1, 1, keys.shape[2]))
      if mask is not None:
        # if mask.shape != (1,1,16,16):
        #   breakpoint()
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
  with jax.named_scope("attn_soft"):
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    if v_scaler is not None:
      scores = scores * v_scaler.reshape((bsz, 1, 1, keys.shape[2]))

  with jax.named_scope("attn_mat2"):
    # output = torch.einsum(
    #    "ikjm,ikml->ikjl", scores, values
    # )  # (bs, n_local_heads, seqlen, head_dim)
    output = torch.einsum("ikjm,ikml->ikjl", scores, values)
  return output

class AttentionKernel:

  def __init__(self, env):
    self.env = env
    self.shard_axis = 0 if self.env.shard_on_batch else 1
    qkv_pspec = self.env.partition_by_axis(self.shard_axis) # Number of heads
    others_pspec = self.env.partition_by_axis()
    self.binded_ragged_mha = functools.partial(ragged_mha, bk=self.env.block_size, shard_axis=self.shard_axis)
    self.binded_ragged_mha = shard_map(ragged_mha, env.mesh, in_specs=(*([qkv_pspec] * 3), *([others_pspec] * 4)), out_specs=(qkv_pspec, (others_pspec, others_pspec)), check_rep=False)
    self.binded_ragged_mha = jax.jit(self.binded_ragged_mha)

  def __call__(self, xq, xk, xv, mask, cache, start, end, ragged_batch_index, ragged_block_index):
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
  
    with jax.named_scope("attn_qkv"):
      if self.env.ragged_mha and seqlen == 1:
        output, _ = torch_xla2.interop.call_jax(self.binded_ragged_mha, xq, keys, values, start, end, ragged_batch_index, ragged_block_index)
      else:
        output = dense_attention(xq, keys, values, None, None, mask)

      if seqlen == 1:
        output = output[:, :, 0:1, :]
      # For XLA matmul performance boost
      # output = torch.matmul(scores, values)
      self.env.apply_sharding(output, axis=self.shard_axis)
      return output


class Int8KVAttentionKernel:

  def __init__(self, env):
    self.env = env
    self.shard_axis = 0 if self.env.shard_on_batch else 1
    qkv_pspec = self.env.partition_by_axis(self.shard_axis) # Number of heads
    others_pspec = self.env.partition_by_axis()
    self.binded_ragged_mha_quantized = functools.partial(ragged_mha, bk=self.env.block_size, shard_axis=self.shard_axis)
    self.binded_ragged_mha_quantized = shard_map(self.binded_ragged_mha_quantized, env.mesh, in_specs=(*([qkv_pspec] * 3), *([others_pspec]*6)), out_specs=(qkv_pspec, (others_pspec, others_pspec)), check_rep=False)
    self.binded_ragged_mha_quantized = jax.jit(self.binded_ragged_mha_quantized)

  def __call__(self, xq, xk, xv, mask, cache, start, end, ragged_batch_index, ragged_block_index):
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

    with jax.named_scope("attn_qkv"):
      if self.env.ragged_mha and seqlen == 1:
        output, _ = torch_xla2.interop.call_jax(self.binded_ragged_mha_quantized, xq, keys, values, start, end, ragged_batch_index, ragged_block_index, k_scaler, v_scaler)
      else:
        output= dense_attention(xq, keys, values, k_scaler, v_scaler, mask)

      if seqlen == 1:
        output = output[:, :, 0:1, :]

      self.env.apply_sharding(output, axis=self.shard_axis)
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
      start,
      end,
      ragged_batch_index,
      ragged_block_index,
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

    output = self.attention_kernel(xq, xk, xv, mask, cache, start, end, ragged_batch_index, ragged_block_index).type_as(xq)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)
