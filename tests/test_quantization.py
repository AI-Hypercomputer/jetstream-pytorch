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

import copy
import functools
import unittest

import jax
import jax.numpy as jnp
import jax.sharding as jsharding
import torch
import torch_xla2
from jax.experimental import mesh_utils
from jetstream_pt import cache_manager, layers, quantize, torchjax, environment
from jetstream_pt.environment import QuantizationConfig
from jetstream_pt.layers import (
    WeightOnlyBlockwiseQuantizedLinear,
    WeightOnlyPerChannelQuantizedLinear,
)
from jetstream_pt.quantize_model import quantize_model
from jetstream_pt.quantize import dequantize_tensor, quantize_tensor
from tests import helpers
from torch.utils import _pytree as pytree
from torch_xla2 import tensor
import copy

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
    def update_env_data(env_data):
      env_data.ring_buffer=False
      env_data.ragged_mha=False
      env_data.flash_attention=True
      env_data.generate_cache_stacked=True
      env_data.new_cache_stacked=True
      env_data.lazy_cache_update=True
      env_data.quant_config.enable_kv_quantization = True
      env_data.batch_size = 4
    env, _ = helpers.make_env_tiny(True, update_env_data)
    
    batch = env.batch_size
    if env.generate_cache_stacked:
      cache_shape = (env.num_layers, batch, 2, 100, 2)  # layer, bs, num heads, seqlen, dim
    else:
      cache_shape = (batch, 2, 100, 2)  # bs, num heads, seqlen, dim
    with jax.default_device(jax.devices("cpu")[0]):
      
      cache = cache_manager.KVCacheGenerate.empty(
          cache_shape, None, env
      )
      # seqlen is 1
      k = self._xla_tensor((batch, 2, 1, 2))
      v = self._xla_tensor((batch, 2, 1, 2))

      def update_finalize_compare(in_k, in_v, in_layer, in_pos):
        cache.input_pos = [in_pos] if env.ring_buffer else jnp.array([in_pos] * batch)

        # layer id may or may not take effect, depends on the env config.
        cache.update(in_k, in_v, layer_id=in_layer)
        cache.finalize()
        if env.quant_config.enable_kv_quantization:
          new_k = cache.cache_k * cache.k_scaler
          new_v = cache.cache_v * cache.v_scaler
        else:
          new_k = cache.cache_k
          new_v = cache.cache_v

        if env.generate_cache_stacked:
          self.assertTrue(
              jnp.allclose(k._elem, new_k._elem[in_layer, :, :, in_pos:(in_pos + 1), :], atol=0.1)
          )
          self.assertTrue(
              jnp.allclose(v._elem, new_v._elem[in_layer, :, :, in_pos:(in_pos + 1), :], atol=0.1)
          )
        else:
          self.assertTrue(
              jnp.allclose(k._elem, new_k._elem[:, :, in_pos:(in_pos + 1), :], atol=0.1)
          )
          self.assertTrue(
              jnp.allclose(v._elem, new_v._elem[:, :, in_pos:(in_pos + 1), :], atol=0.1)
          )
      update_finalize_compare(k, v, in_layer=1, in_pos=57)
      update_finalize_compare(k, v, in_layer=1, in_pos=58)
      update_finalize_compare(k, v, in_layer=2, in_pos=3)

  def test_kv_kernel(self):
    """test kv cache quantization"""
    def update_env_data(env_data):
      env_data.ring_buffer=False
      env_data.ragged_mha=False
      env_data.flash_attention=True
      env_data.generate_cache_stacked=True
      env_data.new_cache_stacked=True
      env_data.lazy_cache_update=True
      env_data.quant_config.enable_kv_quantization=False
      env_data.batch_size = 4
    env, _ = helpers.make_env_tiny(False, update_env_data)

    batch = env.batch_size
    if env.generate_cache_stacked:
      cache_shape = (env.num_layers, batch, 2, 100, 2)  # bs, num heads, seqlen, dim
    else:
      cache_shape = (batch, 2, 100, 2)  # layers, bs, num heads, seqlen, dim

    with jax.default_device(jax.devices("cpu")[0]):
      
      key = jax.random.PRNGKey(123)
      key2 = jax.random.PRNGKey(456)
      cache_k_jax = jax.random.normal(key, cache_shape, dtype=env.default_dtype)
      cache_v_jax = jax.random.normal(key2, cache_shape, dtype=env.default_dtype)
      # cache_k_jax = jnp.zeros(cache_shape, dtype=env.default_dtype)
      # cache_v_jax = jnp.zeros(cache_shape, dtype=env.default_dtype)

      start = jnp.zeros((batch,), dtype=jnp.int32)

      cache_k, cache_v, start = torchjax.to_torch((cache_k_jax, cache_v_jax, start))
      
      # Prepare quantized cache before written in
      cache_k_int, cache_k_scaler, _ = quantize_tensor(cache_k, (-3, -1))
      cache_v_int, cache_v_scaler, _ = quantize_tensor(cache_v, (-3, -1))

      # 1 is seqlen
      xq = jax.random.normal(key, (batch, 2, 1, 2), dtype=env.default_dtype)
      xk = jax.random.normal(key, (batch, 2, 1, 2), dtype=env.default_dtype)
      xv = jax.random.normal(key, (batch, 2, 1, 2), dtype=env.default_dtype)

      xq, xk, xv = torchjax.to_torch((xq, xk, xv))

      def get_var(position: int):
        pos = [position] if env.ring_buffer else jnp.array([position] * batch, dtype=jnp.int64)
        mask = jax.lax.broadcast_in_dim(jnp.array([0] * position + [float("-inf")] * (100 - position)), (env.batch_size, 100), (1,))
        mask = torchjax.to_torch((mask))
        return pos, mask


      cache = cache_manager.KVCacheGenerate(cache_k, cache_v, None, None, env)
      # layer_id doesn't matter, will assign later
      attention_float = layers.AttentionKernel(env, layer_id=0)

      float_res = []
      def update_finalize_record(in_attention, in_cache, in_q, in_k, in_v, in_layer, in_pos):
        pos, mask = get_var(in_pos)
        in_attention.layer_id=in_layer
        in_cache.input_pos = pos
        ret = in_attention(in_q, in_k, in_v, mask, in_cache, start=start, end=pos)
        in_cache.finalize()
        return ret

      float_res.append(update_finalize_record(attention_float, cache, xq, xk, xv, 1, 57))
      float_res.append(update_finalize_record(attention_float, cache, xq, xk, xv, 1, 58))
      float_res.append(update_finalize_record(attention_float, cache, xq, xk, xv, 2, 3))

      # Running into the issue of multiple env object always share the same quant_config. 
      # Record the results and compare as a workaround.
      env._data.quant_config.enable_kv_quantization = True
      env = environment.JetEngineEnvironment(env._data)

      cache_int = cache_manager.KVCacheGenerate(
          cache_k_int,
          cache_v_int,
          None,
          None,
          env,
          cache_k_scaler=cache_k_scaler,
          cache_v_scaler=cache_v_scaler
      )
      # layer_id doesn't matter, will assign later
      attention_quant = layers.Int8KVAttentionKernel(env, layer_id=0)

      int_res = []
      int_res.append(update_finalize_record(attention_quant, cache_int, xq, xk, xv, 1, 57))
      int_res.append(update_finalize_record(attention_quant, cache_int, xq, xk, xv, 1, 58))
      int_res.append(update_finalize_record(attention_quant, cache_int, xq, xk, xv, 2, 3))

      for f, i in zip(float_res, int_res):
        self.assertTrue(jnp.allclose(f.jax(), i.jax(), atol=0.01))
  
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
      w_dq_asym = dequantize_tensor(w_q_asym, s_asym, zp_asym)
      # Asymmetric is more accurate than symmetric.
      self.assertLess(
          (w - w_dq_asym).norm(),
          (w - w_dq).norm(),
      )
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

    w = (
        torch.randn(2, 8) + 2
    )  # Add a bias to normal dist to test asymmetric quant.
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
    res, torch_res, block_diff = self._nn_linear_run_and_compare(
        nn_linear, block_q_linear, arg
    )
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
    quant_config = QuantizationConfig(
        is_symmetric_weight=False, is_blockwise_weight=True
    )
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

  def test_quant_creator(self):

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
    quantized = layers.create_quantized_from_nn_linear(nn_linear, quant_config)
    res, torch_res, _ = self._nn_linear_run_and_compare(
        nn_linear, quantized, arg
    )
    self.assertGreater(self._calc_cosine_dist(res, torch_res), 0.9999)

  def test_3_layers(self):

    class Model(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024, 2048, bias=False)
        self.linear2 = torch.nn.Linear(2048, 2048, bias=False)
        self.linear3 = torch.nn.Linear(2048, 1024, bias=False)

      def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

    m = Model()
    m.to(torch.bfloat16)
    arg = torch.randn(2, 16, 1024).to(torch.bfloat16)
    torch_res = m(arg)

    quant_config = QuantizationConfig(
        enable_weight_quantization=True,
        enable_activation_quantization=False,
    )
    qm = quantize_model(m, quant_config)
    res = helpers.call_xla_model(qm, qm.state_dict(), arg)
    self.assertGreater(self._calc_cosine_dist(res, torch_res), 0.9999)


if __name__ == "__main__":
  unittest.main()
