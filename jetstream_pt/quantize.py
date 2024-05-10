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

import torch

EPS = torch.finfo(torch.float32).eps

def quantize_torch_int8(val, reduce_axis):
  """quantize torch int8"""
  # val is (batch, heads, seqlen, dim)
  scale = torch.amax(val.abs(), axis=reduce_axis, keepdim=True)
  scale = scale / 127
  scale = torch.where(scale == 0, torch.ones_like(scale), scale)
  return (val / scale).to(torch.int8), scale


def quantize_torch(val, reduce_axis, n_bits=8):
  # val is (batch, heads, seqlen, dim)
  scale = torch.amax(val.abs(), axis=reduce_axis, keepdim=True)
  mag = 2**(n_bits - 2) - 1
  scale = scale / mag
  scale = torch.where(scale == 0, torch.ones_like(scale), scale)
  eps = torch.zeros_like(scale).fill_(EPS)
  scale = torch.max(scale, eps)
  return (val / scale).to(torch.int8), scale

def quantize_blockwise(val, decomp_axis=0, n_bits=8, block_size=128):
  # val is (in_features, out_features)
  assert len(val.shape) == 2
  val = val.to(torch.float32)
  val = val.reshape(-1, block_size, val.shape[-1])
  quant_val, scale = quantize_torch(val, (1,), n_bits)
  return quant_val, scale


def dequantize_torch_int8(val, scale):
  """dequantize torch int8"""
  return val * scale


from dataclasses import dataclass
from typing import Optional
import torch.ao.quantization.fx._decomposed
@dataclass
class TensorQConfig:
  dtype: torch.dtype = torch.int8
  axis: int = -1
  quant_min: int = -128
  quant_max: int = 127
  symmetric_quant: bool = True
  is_blockwise: bool = False
  blocks_size: int = 128

def _find_per_channel_min_max(x: torch.Tensor, axis: int):
  x_dim = x.size()
  new_axis_list = [i for i in range(len(x_dim))]
  new_axis_list[axis] = 0
  new_axis_list[0] = axis
  y = x.permute(new_axis_list)
  y = torch.flatten(y, start_dim=1)
  return torch.aminmax(y, dim=1)

def _find_qparams(x: torch.Tensor, qconfig : TensorQConfig):
  # Only support per-channel symmetric quant to int8 now
  axis = qconfig.axis
  dtype = qconfig.dtype
  symmetric_quant = qconfig.symmetric_quant
  quant_min = qconfig.quant_min
  quant_max = qconfig.quant_max
  assert axis >= 0 and axis < len(x.shape)
  assert dtype == torch.int8
  min_val, max_val = _find_per_channel_min_max(x, axis)
  min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
  max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
  scale = torch.ones(min_val_neg.size(), dtype=torch.float32)
  if symmetric_quant:
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scale = max_val_pos / (float(quant_max - quant_min) / 2)
    eps = torch.zeros_like(scale).fill_(EPS)
    scale = torch.max(scale, eps)
    return scale, None
  else:
    assert symmetric_quant

def _quantize_to_dtype(x: torch.Tensor, qconfig: TensorQConfig,
                       scale: torch.Tensor,
                       zero_point: Optional[torch.Tensor] = None):
  if zero_point is None:
      zero_point = torch.zeros_like(scale)
  return torch.ops.quantized_decomposed.quantize_per_channel(
      x, scale, zero_point, qconfig.axis, qconfig.quant_min,
      qconfig.quant_max, qconfig.dtype
  )

def quantize_tensor(x: torch.Tensor, qconfig : TensorQConfig):
  if not qconfig.is_blockwise:
    scale, zp = _find_qparams(x, qconfig)
    x_int = _quantize_to_dtype(x, qconfig, scale, zp)
    return x_int, scale, zp
  else:
    assert len(x.shape) == 2
    x = x.reshape(-1, qconfig.blocks_size, x.shape[-1])
    qconfig.axis = 1
    scale, zp = _find_qparams(x, qconfig)
    x_int = _quantize_to_dtype(x, qconfig, scale, zp)
    return x_int, scale, zp
