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

import unittest
import jax
import jax.numpy as jnp
from tests import helpers
import torch
import torch_xla2

from jetstream_pt import cache_manager, layers, quantize
from jetstream_pt.torchjax import from_jax, to_jax


class QuantizationTest(unittest.TestCase):
  """test kv cache quantization"""

  def _xla_tensor(self, shape):
    res = torch.randn(shape, dtype=torch.bfloat16)
    return torch_xla2.default_env().to_xla(res)

  def test_kv_cache(self):
    """test kv cache quantization"""
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

  def test_kv_kernel(self):
    """test kv cache quantization"""
    cache_shape = (3, 2, 100, 2)  # bs, num heads, seqlen, dim
    with jax.default_device(jax.devices("cpu")[0]):
      env, _ = helpers.make_env_tiny(False)
      key = jax.random.PRNGKey(123)
      key2 = jax.random.PRNGKey(456)
      cache_k_jax = jax.random.normal(key, cache_shape)
      cache_v_jax = jax.random.normal(key2, cache_shape)

      cache_k, cache_v = from_jax((cache_k_jax, cache_v_jax))

      cache = cache_manager.KVCacheGenerate(cache_k, cache_v, [0], None)

      # 1 is seqlen
      xq = jax.random.normal(key, (3, 2, 1, 2))
      xk = jax.random.normal(key, (3, 2, 1, 2))
      xv = jax.random.normal(key, (3, 2, 1, 2))

      xq, xk, xv = from_jax((xq, xk, xv))

      attention_float = layers.AttentionKernel(env)
      float_res = attention_float(xq, xk, xv, None, cache)

      # ==

      cache_k, cache_v = from_jax((cache_k_jax, cache_v_jax))
      cache_k_int, cache_k_scaler = quantize.quantize_torch_int8(
          cache_k, (1, 3)
      )
      cache_v_int, cache_v_scaler = quantize.quantize_torch_int8(
          cache_v, (1, 3)
      )
      cache_int = cache_manager.Int8KVCacheGenerate(
          cache_k_int, cache_v_int, cache_k_scaler, cache_v_scaler, [0], None
      )
      attention_quant = layers.Int8KVAttentionKernel(env)
      int_res = attention_quant(xq, xk, xv, None, cache_int)

      self.assertTrue(jnp.allclose(float_res.jax(), int_res.jax(), atol=0.01))


if __name__ == "__main__":
  unittest.main()
