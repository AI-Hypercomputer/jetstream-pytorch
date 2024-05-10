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
from jetstream_pt.layers import WeightOnlyBlockwiseQuantizedLinear
from jetstream_pt.quantize import quantize_torch, quantize_blockwise
from torch_xla2 import tensor
from torch.utils import _pytree as pytree

import jax.sharding as jsharding
from jax.experimental import mesh_utils

# key = jax.random.PRNGKey(12345)


class QuantizationTest(unittest.TestCase):
  """test kv cache quantization"""

  def _xla_tensor(self, shape):
    res = torch.randn(shape, dtype=torch.bfloat16)
    return torch_xla2.tensor.move_to_device(res)

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

      cache_k, cache_v = torch_xla2.tensor.wrap((cache_k_jax, cache_v_jax))

      cache = cache_manager.KVCacheGenerate(cache_k, cache_v, [0], None)

      # 1 is seqlen
      xq = jax.random.normal(key, (3, 2, 1, 2))
      xk = jax.random.normal(key, (3, 2, 1, 2))
      xv = jax.random.normal(key, (3, 2, 1, 2))

      xq, xk, xv = torch_xla2.tensor.wrap((xq, xk, xv))

      attention_float = layers.AttentionKernel(env)
      float_res = attention_float(xq, xk, xv, None, cache)

      # ==

      cache_k, cache_v = torch_xla2.tensor.wrap((cache_k_jax, cache_v_jax))
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


  def test_quantize_tensor(self):
    cache_shape = (3, 2, 100, 2)  # bs, num heads, seqlen, dim
    with jax.default_device(jax.devices("cpu")[0]):
      cache = cache_manager.Int8KVCacheGenerate.empty(cache_shape, None)
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

  def test_quantize_tensor(self):
    with jax.default_device(jax.devices("cpu")[0]):
      w = self._xla_tensor((512, 256))
      x = self._xla_tensor((2, 256))
      # import pdb;pdb.set_trace()
      w_q, s = quantize_blockwise(w.transpose(0,1))
      print(w_q.shape)
      print(s.shape)
      w_dq = (w_q * s).reshape(-1, w.shape[0]).transpose(0,1)
      # w_dq = torch_xla2.tensor.j2t(w_dq._elem)
      # import pdb;pdb.set_trace()
      # print(torch.max((w_dq - w).abs()))
      print((w_dq - w).norm())
    w = torch.rand(512, 256)
    x = torch.rand(2, 256)
    w_q, s = quantize_blockwise(w.transpose(0,1))
    w_dq = (w_q * s).reshape(-1, w.shape[0]).transpose(0,1)
    print(torch.max((w_dq - w).abs()))
    print((w_dq - w).norm())
  
  def test_per_channel(self):
    with jax.default_device(jax.devices("cpu")[0]):
      w = torch.rand((512, 256))
      x = torch.rand((2, 256))
      w_q, s = quantize_torch(w, (1,))
      w_dq = w_q * s
      print((w_dq - w).norm())
      
      w = torch_xla2.tensor.move_to_device(w)
      x = torch_xla2.tensor.move_to_device(x)
      w_q, s = quantize_torch(w, (1,))
      w_dq = (w_q * s)
      print((w_dq - w).norm())
    
  def test_hex_llm_quant(self):
    from jetstream_pt.quantize import quantize_tensor, TensorQConfig
    print("test per-channel")
    qconfig = TensorQConfig(is_blockwise=False, axis=1)
    w = torch.rand((512, 256))
    w_q, s, _ = quantize_tensor(w, qconfig)
    w_dq = w_q * s
    print("my quant diff: ", w_dq - w)
    print("my quant norm: ", (w_dq - w).norm())
    with jax.default_device(jax.devices("cpu")[0]):
      w_q, s = quantize_torch(w, (1,))
      w_dq = (w_q * s)
      print("my quant diff: ", w_dq - w)
      print("my quant norm: ", (w_dq - w).norm())

      print("test blockwise")
      qconfig = TensorQConfig(is_blockwise=True)
    

  def test_int4_quantized_layer(self):

    def get_sharding_spec():
      num_of_partitions = jax.device_count()
      mesh = jsharding.Mesh(
          mesh_utils.create_device_mesh((num_of_partitions, 1)),
          axis_names=("x", "y"),
      )
      sharding = ["x", None]
      sharding_spec = jsharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*sharding)
      )
      return sharding_spec

    def _move_weight_to_jax(state_dict):
      def make_array(t):
        res = jax.random.normal(
            jax.random.key(0), shape=t.shape, dtype=jnp.bfloat16
        )
        res = res.astype(torch_xla2.tensor.t2j_dtype(t.dtype))
        return res

      return pytree.tree_map_only(torch.Tensor, make_array, state_dict)

    layer = WeightOnlyBlockwiseQuantizedLinear(1024, 2048, False, "meta")

    @jax.jit
    def f(weights, args):
      paramst, argst = torch_xla2.tensor.wrap((weights, args))
      with torch_xla2.tensor.XLADispatchMode():
        res = torch.func.functional_call(layer, paramst, argst)[0]
      return torch_xla2.tensor.unwrap(res)

    state_dict_jax = _move_weight_to_jax(layer.state_dict())
    for k, v in state_dict_jax.items():
      if k == "weight":
        print(f"cast weight {k} to int4.")
        state_dict_jax[k] = v.astype(jnp.int4)
        # state_dict_jax[k] = jax.device_put(v, get_sharding_spec())
    input = jax.random.normal(key, shape=(2, 32, 1024), dtype=jnp.bfloat16)
    # input = jax.device_put(input, get_sharding_spec())
    # out = f(state_dict_jax, input)
    print(f.lower(state_dict_jax, input).as_text("hlo"))
    print(f.lower(state_dict_jax, input).compile().as_text())

  def test_sharding_spec(self):

    num_of_partitions = jax.device_count()
    mesh = jsharding.Mesh(
        mesh_utils.create_device_mesh((num_of_partitions, 1)),
        axis_names=("x", "y"),
    )

    axis = 1
    sharding = [None] * (axis + 1)
    sharding[axis] = "x"
    sharding = [None, "x", None]
    sharding_spec = jsharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*sharding)
    )
    print(sharding_spec)


if __name__ == "__main__":
  unittest.main()
