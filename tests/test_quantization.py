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

import jax
import unittest
from jetstream_pt import cache_manager
from jetstream_pt import quantize
import torch
import torch_xla2
import jax.numpy as jnp


class QuantizationTest(unittest.TestCase):

  def _xla_tensor(self, shape):
    res = torch.randn(shape, dtype=torch.bfloat16)
    return torch_xla2.tensor.move_to_device(res)

  def test_kv_cache(self):
    cache_shape = (3, 2, 100, 2)  # bs, num heads, seqlen, dim
    with jax.default_device(jax.devices("cpu")[0]):
      cache = cache_manager.Int8KVCacheGenerate.empty(cache_shape, None, False)
      # seqlen is 1
      k = self._xla_tensor((3, 2, 1, 2))
      v = self._xla_tensor((3, 2, 1, 2))

      cache.input_pos = [57]
      new_k, new_v, scaler_k, scaler_v = cache.update(k, v)
      new_k = new_k * scaler_k
      new_v = new_v * scaler_v

      self.assertTrue(
          jnp.allclose(k._elem, new_k._elem[:, :, 57:58, :], atol=0.1)
      )
      self.assertTrue(
          jnp.allclose(v._elem, new_v._elem[:, :, 57:58, :], atol=0.1)
      )


if __name__ == "__main__":
  unittest.main()
