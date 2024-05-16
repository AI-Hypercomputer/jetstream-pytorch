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
    assert len(reduce_axis) == 1, "blockwise quant only works along 1 dim."
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
    zero_point = min_int - torch.round(min_val / scales)
    # zero_point = (min_int - torch.round(min_val / scales)).clamp_(
    #     min_int, max_int
    # )
  else:
    max_val = w.abs().amax(dim=reduce_axis, keepdim=True)
    max_val = max_val.clamp(min=EPS)
    scales = max_val / max_int
    zero_point = 0

  w = torch.clamp(
      torch.round(w * (1.0 / scales)) + zero_point, min_int, max_int
  ).to(torch.int8)

  if symmetric:
    zero_point = None
  return w, scales, zero_point


def dequantize_tensor(w, scale, zero_point=None, block_size=-1):
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
