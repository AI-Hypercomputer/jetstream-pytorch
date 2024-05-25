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

from typing import Tuple, Union

import torch
import jax.numpy as jnp

EPS = 1e-5


def quantize_tensor(
    w: torch.Tensor,
    reduce_axis: Union[Tuple[int], int],
    n_bit: int = 8,
    symmetric: bool = True,
    block_size: int = -1,
):
  """
  Quantize weight tensor w along 'reduce_axis'.

  Args:
    w: weight tensor to be quantized.
    reduce_axis: axises along which to quantize.
    n_bit: Quantize to n_bit bits. (Use int8 container for n_bits < 8).
    symmetric: Whether quantization is symmetric.
    block_size: Blocksize for blockwise quantization. -1 for per-channel quant.

  Return:
    w_q: Quantized weight in int8 container
    scale: scalar for quantized tensor
    zero_point: zero_point for quantized tensor, None if symmetric quantization
  """

  assert 0 < n_bit <= 8, "Quantization bits must be between [1, 8]."
  if isinstance(reduce_axis, int):
    reduce_axis = (reduce_axis,)

  if block_size > 0:
    axis = reduce_axis[0]
    w_shape = w.shape
    assert w_shape[axis] % block_size == 0
    w = w.reshape(w_shape[:axis] + (-1, block_size) + w_shape[axis + 1 :])
    reduce_axis = axis + 1

  max_int = 2 ** (n_bit - 1) - 1
  min_int = -(2 ** (n_bit - 1))
  if not symmetric:
    max_val = w.amax(dim=reduce_axis, keepdim=True)
    min_val = w.amin(dim=reduce_axis, keepdim=True)
    scales = (max_val - min_val).clamp(min=EPS) / float(max_int - min_int)
    zero_point = min_int - min_val / scales
  else:
    max_val = w.abs().amax(dim=reduce_axis, keepdim=True)
    max_val = max_val.clamp(min=EPS)
    scales = max_val / max_int
    zero_point = 0

  w = torch.clamp(
      torch.round(w * (1.0 / scales) + zero_point), min_int, max_int
  ).to(torch.int8)

  return w, scales, zero_point if not symmetric else None


def dequantize_tensor(w, scale, zero_point=None):
  """Dequantize tensor quantized by quantize_tensor."""
  if zero_point is not None:
    return (w - zero_point) * scale

  return w * scale


def load_q_weight_helper(w_q, scale, zp=None, block_size=-1):
  """Helper function to update the shape of quantized weight to match
  what quantized linear layer expects."""
  if block_size < 0:
    w_q = w_q.to(torch.int8)
    if zp is not None:
      zp = (zp * scale).squeeze(-1).to(torch.bfloat16)
    scale = scale.squeeze(-1).to(torch.bfloat16)
  else:
    w_q = w_q.permute(1, 2, 0).to(torch.int8)
    if zp is not None:
      zp = (zp * scale).transpose(1, 0).squeeze(-1).to(torch.bfloat16)
    scale = scale.transpose(1, 0).squeeze(-1).to(torch.bfloat16)
  return w_q, scale, zp


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
