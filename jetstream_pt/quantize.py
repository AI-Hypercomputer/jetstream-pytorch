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


def quantize_torch_int8(val, reduce_axis):
  """quantize torch int8"""
  # val is (batch, heads, seqlen, dim)
  scale = torch.amax(val.abs(), axis=reduce_axis, keepdim=True)
  scale = scale / 127
  scale = torch.where(scale == 0, torch.ones_like(scale), scale)
  return (val / scale).to(torch.int8), scale

def quantize_torch_int4(val, reduce_axis):
    # val is (batch, heads, seqlen, dim)
    scale = torch.amax(val.abs(), axis=reduce_axis, keepdim=True)
    scale = scale / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (val / scale).to(torch.int8), scale


def dequantize_torch_int8(val, scale):
  """dequantize torch int8"""
  return val * scale
