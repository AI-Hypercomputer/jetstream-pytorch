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
from jetstream_pt import cache_manager, layers, quantize
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
    return torch_xla2.tensor.move_to_device(res)

  def _to_xla_tensor(self, tree):
    return pytree.tree_map_only(
        torch.Tensor, torch_xla2.tensor.move_to_device, tree
    )

  def _call_xla_model(self, model, weights, args):
    with jax.default_device(jax.devices("cpu")[0]):
      xla_weights, xla_inputs = self._to_xla_tensor((weights, args))
      result = torch.func.functional_call(model, xla_weights, xla_inputs)
      result_torch = torch_xla2.tensor.j2t(result._elem)
      return result_torch

  def _calc_cosine_dist(self, x, y):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)
    return (torch.dot(x, y) / (x.norm() * y.norm())).item()

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
      cache_k_int, cache_k_scaler, _ = quantize.quantize_tensor(cache_k, (1, 3))
      cache_v_int, cache_v_scaler, _ = quantize.quantize_tensor(cache_v, (1, 3))
      cache_int = cache_manager.Int8KVCacheGenerate(
          cache_k_int, cache_v_int, cache_k_scaler, cache_v_scaler, [0], None
      )
      attention_quant = layers.Int8KVAttentionKernel(env)
      int_res = attention_quant(xq, xk, xv, None, cache_int)

      self.assertTrue(jnp.allclose(float_res.jax(), int_res.jax(), atol=0.01))

  def test_quantize_dequantize_tensor(self):

    def quantize_dequantize_weight(w, n_bit):
      w_q, s, _ = quantize_tensor(w, (1,), n_bit=n_bit, symmetric=True)
      w_dq = dequantize_tensor(w_q, s)
      if n_bit == 8:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.1))
      elif n_bit == 4:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.4))

      w_q_asym, s_asym, zp_asym = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False
      )
      w_dq_asym = dequantize_tensor(w_q_asym, s_asym, zp_asym)
      self._print_diff(w, w_dq_asym)

      self.assertLess((w - w_dq_asym).norm(), (w - w_dq).norm())

      w_block_q, s_block, _ = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=True, block_size=128
      )
      w_block_dq = dequantize_tensor(w_block_q, s_block, block_size=128)
      w_block_dq = w_block_dq.view(w_block_dq.shape[0], -1)
      self._print_diff(w, w_block_dq)
      self.assertLess((w - w_block_dq).norm(), (w - w_dq).norm())

      w_block_q, s_block, zp = quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False, block_size=128
      )
      w_block_asym_dq = dequantize_tensor(
          w_block_q, s_block, zero_point=zp, block_size=128
      )
      w_block_asym_dq = w_block_asym_dq.view(w_block_asym_dq.shape[0], -1)
      self._print_diff(w, w_block_asym_dq)

      self.assertLess((w - w_block_asym_dq).norm(), (w - w_block_dq).norm())

    w = torch.randn(128, 2048)
    for bit in [4, 8]:
      with self.subTest(bit=bit):
        quantize_dequantize_weight(w, bit)

  def test_quant_linear(self):

    out_features = 2048
    in_features = 2048
    block_size = 128

    @torch.no_grad()
    def run_and_compare(
        nn_linear,
        qlinear_layer,
        arg,
    ):
      torch_result = nn_linear(arg)
      qlinear_layer.quantize_weight_from_nn_linear(nn_linear.weight)
      with jax.default_device(jax.devices("cpu")[0]):
        result = self._call_xla_model(
            qlinear_layer, qlinear_layer.state_dict(), arg
        )
      diff = result - torch_result
      return result, torch_result, diff

    arg = torch.randn(2, 16, in_features).to(torch.bfloat16)
    nn_linear = torch.nn.Linear(
        in_features, out_features, bias=False, dtype=torch.bfloat16
    )
    # Test symmetric quant
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features
    )
    res, torch_res, per_channel_diff = run_and_compare(
        nn_linear, per_channel_q_linear, arg
    )
    self.assertTrue(torch.allclose(res, torch_res, atol=2))
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features
    )
    res, torch_res, block_diff = run_and_compare(nn_linear, block_q_linear, arg)
    # self.assertTrue(torch.allclose(res, torch_res, atol=1.5))
    # Block quant is more accurate than per_channel quant.
    self.assertLess(block_diff.norm(), per_channel_diff.norm())

    # Test asymmetric quant
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    res, torch_res, per_channel_diff2 = run_and_compare(
        nn_linear, per_channel_q_linear, arg
    )
    self._print_diff(res, torch_res)
    self.assertTrue(torch.allclose(res, torch_res, atol=2))
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    block_q_linear.run_fake_quantize = True
    res, torch_res, block_diff2 = run_and_compare(
        nn_linear, block_q_linear, arg
    )
    self._print_diff(res, torch_res)
    self.assertLess(per_channel_diff2.norm(), per_channel_diff.norm())
    # FIXME: Now asymmetric blockwise quant has higher error than asymmetric per-channel.
    # self.assertLess(block_diff2.norm(), per_channel_diff2.norm())

  def test_asymmetric_quant(self):

    out_features = 2048
    in_features = 2048
    block_size = 128

    n_bit = 8
    w = torch.randn((out_features, in_features))  # [out_channel, in_channel]
    arg = torch.randn(2, 16, in_features).to(torch.bfloat16)
    torch_result = torch.matmul(arg, w.t().to(torch.bfloat16))

    # Per-channel asymmetric quant.
    w_q, s, zp = quantize_tensor(
        w,
        (1,),
        n_bit=n_bit,
        symmetric=False,
        block_size=-1,
    )
    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    per_channel_q_linear._load_quantized_weights(w_q, s, zp)
    res_per_channel = self._call_xla_model(
        per_channel_q_linear, per_channel_q_linear.state_dict(), arg
    )
    self.assertTrue(
        torch.allclose(res_per_channel, torch_result, rtol=0.05, atol=3)
    )
    # Blockwise asymmetric quant.
    w_q_block, s_block, zp_block = quantize_tensor(
        w,
        (1,),
        n_bit=n_bit,
        symmetric=False,
        block_size=block_size,
    )
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    block_q_linear._load_quantized_weights(w_q_block, s_block, zp_block)
    block_q_linear.quantize_weight_from_nn_linear(w)
    res_blockwise = self._call_xla_model(
        block_q_linear, block_q_linear.state_dict(), arg
    )
    self.assertTrue(
        torch.allclose(res_blockwise, torch_result, rtol=0.05, atol=3)
    )
    # block_q_linear.run_fake_quantize = True
    # res_blockwise_fake_quant = self._call_xla_model(
    #     block_q_linear, block_q_linear.state_dict(), arg
    # )
    self.assertLess(
        (res_blockwise - torch_result).norm(),
        (res_per_channel - torch_result).norm(),
    )

  def test_blockwise_quantized_linear_sharding(self):

    # TODO(lsy323): Move sharding_by_axis outside of JetEngineEnvironment to a sharding util file
    # and reuse the util functions.
    def sharding_by_axis(axis):
      """return sharding partition spc by axis, options are x, y, -1 or Noe"""
      num_of_partitions = jax.device_count()
      mesh = jsharding.Mesh(
          mesh_utils.create_device_mesh((num_of_partitions, 1)),
          axis_names=("x", "y"),
      )
      if axis == -1 or axis is None:
        return jsharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      sharding = [None] * (axis + 1)
      sharding[axis] = "x"
      sharding_spec = jsharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*sharding)
      )
      return sharding_spec

    def _move_weight_to_jax(state_dict):
      def make_array(t):
        with jax.default_device(jax.devices("cpu")[0]):
          res = jax.random.normal(
              jax.random.key(0), shape=t.shape, dtype=jnp.bfloat16
          )
          res = res.astype(torch_xla2.tensor.t2j_dtype(t.dtype))
          return res

      return pytree.tree_map_only(torch.Tensor, make_array, state_dict)

    @functools.partial(
        jax.jit,
        static_argnums=(0,),
    )
    def f(layer, weights, args):
      paramst, argst = torch_xla2.tensor.wrap((weights, args))
      with torch_xla2.tensor.XLADispatchMode():
        res = torch.func.functional_call(layer, paramst, argst)
      return torch_xla2.tensor.unwrap(res)

    layer = WeightOnlyBlockwiseQuantizedLinear(1024, 2048, False, "meta")
    state_dict_jax = _move_weight_to_jax(layer.state_dict())
    input = jax.random.normal(
        jax.random.key(0), shape=(2, 32, 1024), dtype=jnp.bfloat16
    )

    def lower_f(f, layer, state_dict_jax, input, shardings):
      for k, v in state_dict_jax.items():
        if k == "weight":
          state_dict_jax[k] = v.astype(jnp.int4)
          state_dict_jax[k] = jax.device_put(v, sharding[0])
        if k == "weight_scaler":
          state_dict_jax[k] = jax.device_put(v, sharding[1])
      pre_opt = f.lower(layer, state_dict_jax, input).as_text("hlo")
      post_opt = f.lower(layer, state_dict_jax, input).compile().as_text()
      return post_opt

    shardings = [
        (sharding_by_axis(0), sharding_by_axis(0)),  # wq/wk/wv sharding
        (sharding_by_axis(2), sharding_by_axis(1)),  # wo sharding
        #  (sharding_by_axis(1), sharding_by_axis(0)), # bad sharding
    ]
    for sharding in shardings:
      opt_hlo = lower_f(f, layer, state_dict_jax, input, sharding)
      self.assertFalse("all-to-all" in opt_hlo)
      self.assertFalse("all-reduce-scatter" in opt_hlo)


if __name__ == "__main__":
  unittest.main()
