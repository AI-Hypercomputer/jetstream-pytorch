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

import functools
import unittest

import jax
import jax.numpy as jnp
import jax.sharding as jsharding
import torch
import torch_xla2
from jax.experimental import mesh_utils
from jetstream_pt import cache_manager, layers, quantize, torchjax
from jetstream_pt.environment import QuantizationConfig
from jetstream_pt.layers import (
    WeightOnlyBlockwiseQuantizedLinear,
    WeightOnlyPerChannelQuantizedLinear,
)
from jetstream_pt.quantize import dequantize_tensor, quantize_tensor
from tests import helpers
from torch.utils import _pytree as pytree
from torch_xla2 import tensor

torch.manual_seed(12345)


class QuantizationTest(unittest.TestCase):
  """test kv cache quantization"""

  def _xla_tensor(self, shape):
    res = torch.randn(shape, dtype=torch.bfloat16)
    return torch_xla2.default_env().to_xla(res)

  def _calc_cosine_dist(self, x, y):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)
    return (torch.dot(x, y) / (x.norm() * y.norm())).item()

  def _nn_linear_run_and_compare(
        self,
        nn_linear,
        qlinear_layer,
        arg,
    ):
      torch_result = nn_linear(arg)
      qlinear_layer.quantize_weight_from_nn_linear(nn_linear.weight)
      result = helpers.call_xla_model(
          qlinear_layer, qlinear_layer.state_dict(), arg
      )
      diff = result - torch_result
      return result, torch_result, diff

  def _print_diff(self, w, w_dq):
    print("Print diff:")
    print("  diff: ", w - w_dq)
    print("  max diff: ", torch.max(w - w_dq))
    print("  norm: ", (w - w_dq).norm())
    print("  cosine dist: ", self._calc_cosine_dist(w, w_dq))

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

      cache_k, cache_v = torchjax.to_torch((cache_k_jax, cache_v_jax))

      cache = cache_manager.KVCacheGenerate(cache_k, cache_v, [0], None)

      # 1 is seqlen
      xq = jax.random.normal(key, (3, 2, 1, 2))
      xk = jax.random.normal(key, (3, 2, 1, 2))
      xv = jax.random.normal(key, (3, 2, 1, 2))

      xq, xk, xv = torchjax.to_torch((xq, xk, xv))

      attention_float = layers.AttentionKernel(env)
      float_res = attention_float(xq, xk, xv, None, cache)

      # ==

      cache_k, cache_v = torchjax.to_torch((cache_k_jax, cache_v_jax))
      cache_k_int, cache_k_scaler, _ = quantize_tensor(cache_k, (1, 3))
      cache_v_int, cache_v_scaler, _ = quantize_tensor(cache_v, (1, 3))
      cache_int = cache_manager.Int8KVCacheGenerate(
          cache_k_int, cache_v_int, cache_k_scaler, cache_v_scaler, [0], None
      )
      attention_quant = layers.Int8KVAttentionKernel(env)
      int_res = attention_quant(xq, xk, xv, None, cache_int)

      self.assertTrue(jnp.allclose(float_res.jax(), int_res.jax(), atol=0.01))

  def test_quantize_dequantize_tensor(self):

    def quantize_dequantize_weight(w, n_bit):
      # print(f"original w {w}")
      # Per-channel qdq.
      w_q, s, _ = quantize_tensor(w, (1,), n_bit=n_bit, symmetric=True)
      # print(f"w_q {w_q}, s {s}")
      w_dq = dequantize_tensor(w_q, s)
      # print(f"w_dq {w_dq}")
      if n_bit == 8:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.1))
      elif n_bit == 4:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.4))
      # Per-channel asymmetric qdq.
      w_q_asym, s_asym, zp_asym = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False
      )
      # print(f"w_q_asym {w_q_asym}, s_asym {s_asym}, zp_asym {zp_asym}")
      w_dq_asym = dequantize_tensor(w_q_asym, s_asym, zp_asym)
      # print(f"w_dq_asym {w_dq_asym}")
      # self._print_diff(w, w_dq)
      # self._print_diff(w, w_dq_asym)
      # Asymmetric is more accurate than symmetric.
      self.assertLess((w - w_dq_asym).norm(), (w - w_dq).norm())
      # Blockwise quant.
      w_block_q, s_block, _ = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=True, block_size=2
      )
      w_block_dq = dequantize_tensor(w_block_q, s_block)
      w_block_dq = w_block_dq.view(w_block_dq.shape[0], -1)
      # self._print_diff(w, w_block_dq)
      # Blockwise quant is more accurate than per-channel.
      self.assertLess((w - w_block_dq).norm(), (w - w_dq).norm())
      # Blockwise asymmetric
      w_block_q, s_block, zp = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False, block_size=2
      )
      w_block_asym_dq = dequantize_tensor(w_block_q, s_block, zero_point=zp)
      w_block_asym_dq = w_block_asym_dq.view(w_block_asym_dq.shape[0], -1)
      # self._print_diff(w, w_block_asym_dq)
      # Blockwise asymmetric is more accurate than blockwise symmetric.
      self.assertLess((w - w_block_asym_dq).norm(), (w - w_block_dq).norm())

    w = torch.randn(2, 8)
    for bit in [4, 8]:
      with self.subTest(bit=bit):
        quantize_dequantize_weight(w, bit)

  def test_weight_only_quant(self):

    out_features = 2048
    in_features = 2048
    block_size = 128

    arg = torch.randn(2, 16, in_features).to(torch.bfloat16)
    nn_linear = torch.nn.Linear(
        in_features, out_features, bias=False, dtype=torch.bfloat16
    )
    # Test symmetric quant
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features
    )
    res, torch_res, per_channel_diff = self._nn_linear_run_and_compare(
        nn_linear, per_channel_q_linear, arg
    )
    self.assertTrue(torch.allclose(res, torch_res, atol=2))
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features
    )
    res, torch_res, block_diff = self._nn_linear_run_and_compare(nn_linear, block_q_linear, arg)
    # self.assertTrue(torch.allclose(res, torch_res, atol=1.5))
    # Block quant is more accurate than per_channel quant.
    self.assertLess(block_diff.norm(), per_channel_diff.norm())

    # Test asymmetric quant
    quant_config = QuantizationConfig(is_symmetric_weight=False)
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, quant_config=quant_config
    )
    res, torch_res, per_channel_diff2 = self._nn_linear_run_and_compare(
        nn_linear, per_channel_q_linear, arg
    )
    # self._print_diff(res, torch_res)
    self.assertTrue(torch.allclose(res, torch_res, atol=2))
    quant_config = QuantizationConfig(is_symmetric_weight=False, is_blockwise_weight=True)
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features, quant_config=quant_config
    )
    # block_q_linear.run_fake_quantize = True
    res, torch_res, block_diff2 = self._nn_linear_run_and_compare(
        nn_linear, block_q_linear, arg
    )
    # self._print_diff(res, torch_res)
    self.assertLess(per_channel_diff2.norm(), per_channel_diff.norm())
    # FIXME: Now asymmetric blockwise quant has higher error than asymmetric per-channel.
    # self.assertLess(block_diff2.norm(), per_channel_diff2.norm())

  def test_int4_weight_loading(self):
    layer = WeightOnlyBlockwiseQuantizedLinear(1024, 2048)
    state_dict_jax = torchjax.from_torch(
        helpers.to_xla_tensor(layer.state_dict())
    )
    state_dict_jax["weight"] = state_dict_jax["weight"].astype(jnp.int4)
    state_dict_torch = torchjax.to_torch(state_dict_jax)
    self.assertTrue(state_dict_torch["weight"]._elem.dtype == jnp.int4)

  def test_blockwise_quantized_linear_sharding(self):

    @functools.partial(
        jax.jit,
        static_argnums=(0,),
    )
    def f(layer, weights, args):
      paramst, argst = torchjax.to_torch((weights, args))
      with torch_xla2.default_env():
        res = torch.func.functional_call(layer, paramst, argst)
      return torchjax.from_torch(res)

    layer = WeightOnlyBlockwiseQuantizedLinear(1024, 2048)
    state_dict_jax = torchjax.from_torch(
        helpers.to_xla_tensor(layer.state_dict())
    )
    input = jax.random.normal(
        jax.random.key(0), shape=(2, 32, 1024), dtype=jnp.bfloat16
    )

    def shard_and_lower(f, layer, state_dict_jax, input, shardings):
      for k, v in state_dict_jax.items():
        if k == "weight":
          state_dict_jax[k] = v.astype(jnp.int4)
          state_dict_jax[k] = jax.device_put(v, sharding[0])
        if k == "weight_scaler":
          state_dict_jax[k] = jax.device_put(v, sharding[1])
      pre_opt = f.lower(layer, state_dict_jax, input).as_text("hlo")
      post_opt = f.lower(layer, state_dict_jax, input).compile().as_text()
      return post_opt

    env, _ = helpers.make_env_tiny()
    shardings = [
        (env.sharding_by_axis(0), env.sharding_by_axis(0)),  # wq/wk/wv sharding
        (env.sharding_by_axis(2), env.sharding_by_axis(1)),  # wo sharding
        #  (sharding_by_axis(1), sharding_by_axis(0)), # bad sharding
    ]
    for sharding in shardings:
      opt_hlo = shard_and_lower(f, layer, state_dict_jax, input, sharding)
      self.assertFalse("all-to-all" in opt_hlo)
      self.assertFalse("all-reduce-scatter" in opt_hlo)
  
  def test_activation_quant_per_channel(self):

    out_features = 8
    in_features = 4
    block_size = 128
    
    arg = torch.randn(2, 1, in_features).to(torch.bfloat16)
    nn_linear = torch.nn.Linear(
        in_features, out_features, bias=False, dtype=torch.bfloat16
    )
    quant_config = QuantizationConfig(
      enable_weight_quantization=True,
      enable_activation_quantization=True,
    )
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, quant_config=quant_config
    )
    res, torch_res, _ = self._nn_linear_run_and_compare(
        nn_linear, per_channel_q_linear, arg
    )
    self.assertGreater(self._calc_cosine_dist(res, torch_res), 0.9999)    


if __name__ == "__main__":
  unittest.main()
