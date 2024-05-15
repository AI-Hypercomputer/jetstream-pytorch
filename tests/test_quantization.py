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
from jetstream_pt.layers import WeightOnlyPerChannelQuantizedLinear, WeightOnlyBlockwiseQuantizedLinear
from jetstream_pt.quantize import quantize_torch, quantize_blockwise
from jetstream_pt.quantize import pseudo_quantize_tensor, awq_dequantize, quantize_torch_int8, dequantize_torch_int8, quantize_torch
from jetstream_pt.quantize import quantize_tensor, TensorQConfig
from torch_xla2 import tensor
from torch.utils import _pytree as pytree

import jax.sharding as jsharding
from jax.experimental import mesh_utils

key = jax.random.PRNGKey(12345)
torch.manual_seed(123)


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

  def test_qdq(self):
    def print_diff(w, w_dq):
      print("diff: ", w - w_dq)
      print("max diff: ", torch.max(w - w_dq))
      print("norm: ", (w - w_dq).norm())

    def qdq_w(w, n_bit):
      w_q, s = pseudo_quantize_tensor(w, (1,), n_bit=n_bit, symmetric=True)
      w_dq = awq_dequantize(w_q, s)
      if n_bit == 8:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.1))
      elif n_bit == 4:
        self.assertTrue(torch.allclose(w, w_dq, atol=0.4))

      w_q_asym, s_asym, zp_asym = pseudo_quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False
      )
      w_dq_asym = awq_dequantize(w_q_asym, s_asym, zp_asym)
      print_diff(w, w_dq_asym)

      self.assertLess((w - w_dq_asym).norm(), (w - w_dq).norm())

      # w_q_orig, s_orig = quantize_torch(w, 1, n_bits=8)
      # w_dq_orig = dequantize_torch_int8(w_q_orig, s_orig)

      # qconfig = TensorQConfig(axis = 0)
      # w_q_hex, s_hex, _ = quantize_tensor(w, qconfig)
      # s_hex = s_hex.unsqueeze(1)
      # w_dq_hex = dequantize_torch_int8(w_q_hex, s_hex)

      w_block_q, s_block = pseudo_quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=True, block_size=128
      )
      w_block_dq = awq_dequantize(w_block_q, s_block, block_size=128)
      w_block_dq = w_block_dq.view(w_block_dq.shape[0], -1)
      print_diff(w, w_block_dq)
      self.assertLess((w - w_block_dq).norm(), (w - w_dq).norm())

      w_block_q, s_block, zp = pseudo_quantize_tensor(
          w, (1,), n_bit=n_bit, symmetric=False, block_size=128
      )
      w_block_asym_dq = awq_dequantize(
          w_block_q, s_block, zero_point=zp, block_size=128
      )
      w_block_asym_dq = w_block_asym_dq.view(w_block_asym_dq.shape[0], -1)
      print_diff(w, w_block_asym_dq)

      self.assertLess((w - w_block_asym_dq).norm(), (w - w_block_dq).norm())
      # print("s:", s)
      # print("s_orig:", s_orig)
      # print("s_hex:", s_hex)
      # print("diff: ", w - w_dq)
      # print("new norm: ", (w - w_dq).norm())
      # print("diff: ", w - w_dq_orig)
      # print("orig norm: ", (w - w_dq_orig).norm())
      # print("diff: ", w - w_dq_hex)
      # print("hex norm: ", (w - w_dq_hex).norm())

    w = torch.randn(128, 2048)
    for bit in [4, 8]:
      with self.subTest(bit=bit):
        qdq_w(w, bit)

  def test_awq_quant(self):

    def print_diff(w, w_dq):
      print("diff: ", w - w_dq)
      print("max diff: ", torch.max(w - w_dq))
      print("norm: ", (w - w_dq).norm())

    out_features = 2048
    in_features = 2048
    block_size = 128

    w = torch.randn((out_features, in_features))
    w_q, s = pseudo_quantize_tensor(w, (1,))
    w_dq = awq_dequantize(w_q, s)
    self.assertTrue(torch.allclose(w, w_dq, atol=0.1))

    w_q_block, s_block = pseudo_quantize_tensor(
        w,
        (1,),
        n_bit=8,
        symmetric=True,
        block_size=block_size,
    )
    w_dq_block = awq_dequantize(w_q_block, s_block, block_size=block_size)
    w_dq_block = w_dq_block.reshape(w_dq_block.shape[0], -1)
    self.assertTrue(torch.allclose(w, w_dq_block, atol=0.1))
    self.assertLess((w_dq_block - w).norm(), (w_dq - w).norm())

    arg = torch.rand(2, 16, in_features).to(torch.bfloat16)
    torch_result = torch.matmul(arg, w.t().to(torch.bfloat16))

    def run_and_compare(
        linear_layer, arg, torch_result, symmetric=True, block_size=-1
    ):
      if symmetric:
        w_q, s = pseudo_quantize_tensor(
            w, (1,), symmetric=symmetric, block_size=block_size
        )
        linear_layer.load_quantized_weights(w_q, s)
      else:
        w_q, s, zp = pseudo_quantize_tensor(
            w, (1,), symmetric=symmetric, block_size=block_size
        )
        linear_layer.load_quantized_weights(w_q, s, zp)

      result = self._call_xla_model(
          linear_layer, linear_layer.state_dict(), arg
      )
      diff = result - torch_result
      print("diff: ", diff)
      print("diff norm: ", diff.norm())
      print("max diff: ", diff.abs().max())
      print("cos dist: ", self._calc_cosine_dist(result, torch_result))
      return result, diff

    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features
    )
    # per_channel_q_linear.weight = w_q.to(torch.int8)
    # per_channel_q_linear.weight_scaler = s.to(torch.bfloat16).squeeze(-1)
    # res_per_channel = self._call_xla_model(
    #     per_channel_q_linear, per_channel_q_linear.state_dict(), arg
    # )
    # per_channel_diff = res_per_channel - torch_result
    # print("per-channel diff: ", per_channel_diff)
    # print("per-channel norm: ", per_channel_diff.norm())
    # print("per-channel norm: ", per_channel_diff.abs().max())
    # print("cos dist: ", self._calc_cosine_dist(res_per_channel, torch_result))
    res, per_channel_diff = run_and_compare(
        per_channel_q_linear, arg, torch_result
    )
    self.assertTrue(torch.allclose(res, torch_result, atol=1))

    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features
    )
    # block_q_linear.weight = w_q_block.permute(1, 2, 0).to(torch.int8)
    # block_q_linear.weight_scaler = s_block.transpose(1, 0).squeeze(-1).to(torch.bfloat16)
    # res_blockwise = self._call_xla_model(
    #     block_q_linear, block_q_linear.state_dict(), arg
    # )

    # print("torch res: ", torch_result)
    # print("res_per_channel: ", res_per_channel)
    # print("res_blockwise res: ", res_blockwise)

    # block_diff = res_blockwise - torch_result
    # print("block diff: ", block_diff)
    # print("block norm: ", block_diff.norm())
    # print("block max error: ", block_diff.abs().max())
    # print("cos dist: ", self._calc_cosine_dist(res_blockwise, torch_result))
    res, block_diff = run_and_compare(
        block_q_linear, arg, torch_result, block_size=128
    )
    self.assertTrue(torch.allclose(res, torch_result, atol=1))
    self.assertLess(block_diff.norm(), per_channel_diff.norm())

    # w_q, s, zp = pseudo_quantize_tensor(
    #     w,
    #     (1,),
    #     symmetric=False,
    # )
    per_channel_q_zp_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    # per_channel_q_zp_linear.weight = w_q.to(torch.int8)
    # per_channel_q_zp_linear.weight_scaler = s.squeeze(-1).to(torch.bfloat16)
    # per_channel_q_zp_linear.zero_point = (zp * s).squeeze(-1).to(torch.bfloat16)
    # res_per_channel = self._call_xla_model(
    #     per_channel_q_zp_linear, per_channel_q_zp_linear.state_dict(), arg
    # )
    # per_channel_diff = res_per_channel - torch_result
    # print("my quant norm: ", per_channel_diff.norm())
    # print("max diff: ", per_channel_diff.abs().max())
    # print("cos dist: ", self._calc_cosine_dist(res_per_channel, torch_result))
    res, per_channel_zp_diff = run_and_compare(
        per_channel_q_zp_linear, arg, torch_result, symmetric=False
    )

    # self.assertTrue(
    #     torch.allclose(res_per_channel, torch_result, atol=1)
    # )

    # w_block_q, s_block, zp_block = pseudo_quantize_tensor(
    #       w, (1,), symmetric=False, block_size=128
    #   )
    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features,
        out_features,
        is_symmetric=False,
    )
    res, block_zp_diff = run_and_compare(
        block_q_linear, arg, torch_result, symmetric=False, block_size=128
    )
    # breakpoint()
    # block_q_linear.weight = w_block_q.permute(1, 2, 0).to(torch.int8)
    # block_q_linear.weight_scaler = s_block.transpose(1, 0).squeeze(-1).to(torch.bfloat16)
    # block_q_linear.zero_point = (
    #     (zp_block * s_block).transpose(1, 0).squeeze(-1).to(torch.bfloat16)
    # )
    # res_blockwise = self._call_xla_model(
    #     block_q_linear, block_q_linear.state_dict(), arg
    # )
    # block_diff = res_blockwise - torch_result
    # print("per-channel diff: ", per_channel_diff)
    # print("my quant norm: ", block_diff.norm())
    # print("max diff: ", block_diff.abs().max())
    # print("cos dist: ", self._calc_cosine_dist(res_blockwise, torch_result))
    # self.assertTrue(
    #     torch.allclose(res_blockwise, torch_result, atol=1)
    # )

  def test_asymmetric_quant(self):
    def print_diff(w, w_dq):
      print("diff: ", w - w_dq)
      print("max diff: ", torch.max(w - w_dq))
      print("norm: ", (w - w_dq).norm())

    out_features = 2048
    in_features = 2048
    block_size = 128

    w = torch.rand((out_features, in_features))  # [out_channel, in_channel]
    print("test per-channel")
    w_q, s, zp = pseudo_quantize_tensor(
        w,
        (1,),
        n_bit=8,
        symmetric=False,
        block_size=-1,
    )
    w_dq = awq_dequantize(w_q, s, zp)
    print("my quant diff: ", w_dq - w)
    print("my quant norm: ", (w_dq - w).norm())
    w_q_block, s_block, zp_block = pseudo_quantize_tensor(
        w,
        (1,),
        n_bit=4,
        symmetric=False,
        block_size=block_size,
    )
    w_dq_block = awq_dequantize(
        w_q_block, s_block, zp_block, block_size=block_size
    )
    w_dq_block = w_dq_block.reshape(w_dq_block.shape[0], -1)
    print("my quant diff: ", w_dq_block - w)
    print("my quant norm: ", (w_dq_block - w).norm())

    arg = torch.rand(2, 16, in_features).to(torch.bfloat16)
    torch_result = torch.matmul(arg, w.t().to(torch.bfloat16))

    per_channel_q_linear = WeightOnlyPerChannelQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    per_channel_q_linear.weight = w_q.to(torch.int8)
    per_channel_q_linear.weight_scaler = s.to(torch.bfloat16).squeeze(-1)
    per_channel_q_linear.zero_point = (zp * s).to(torch.bfloat16).squeeze(-1)
    res_per_channel = self._call_xla_model(
        per_channel_q_linear, per_channel_q_linear.state_dict(), arg
    )

    block_q_linear = WeightOnlyBlockwiseQuantizedLinear(
        in_features, out_features, is_symmetric=False
    )
    block_q_linear.weight = w_q_block.permute(1, 2, 0).to(torch.int8)
    block_q_linear.weight_scaler = (
        s_block.transpose(1, 0).to(torch.bfloat16).squeeze(-1)
    )
    block_q_linear.zero_point = (
        (zp_block * s_block).transpose(1, 0).to(torch.bfloat16).squeeze(-1)
    )
    res_blockwise = self._call_xla_model(
        block_q_linear, block_q_linear.state_dict(), arg
    )

    print("torch res: ", torch_result)
    print("res_per_channel: ", res_per_channel)
    print("res_blockwise res: ", res_blockwise)
    per_channel_diff = res_per_channel - torch_result
    print("per-channel diff: ", per_channel_diff)
    print("my quant norm: ", per_channel_diff.norm())
    print("my quant norm: ", per_channel_diff.abs().max())

    block_q_linear.run_fake_quantize = True
    res_blockwise_fake_quant = self._call_xla_model(
        block_q_linear, block_q_linear.state_dict(), arg
    )

    print("fake quant")
    print_diff(res_blockwise_fake_quant, torch_result)
    # block_fq_diff = res_blockwise_fake_quant - torch_result
    # print("per-channel diff: ", block_fq_diff)

    self.assertTrue(
        torch.allclose(res_per_channel, torch_result, rtol=0.05, atol=3)
    )

    block_diff = res_blockwise - torch_result
    print("my quant diff: ", block_diff)
    print("my quant norm: ", block_diff.norm())
    print("my quant norm: ", block_diff.abs().max())
    self.assertTrue(
        torch.allclose(res_blockwise, torch_result, rtol=0.05, atol=3)
    )
    self.assertLess(block_diff.norm(), per_channel_diff.norm())

  def test_int4_quantized_layer(self):

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
        res = torch.func.functional_call(layer, paramst, argst)
      return torch_xla2.tensor.unwrap(res)

    state_dict_jax = _move_weight_to_jax(layer.state_dict())
    input = jax.random.normal(key, shape=(2, 32, 1024), dtype=jnp.bfloat16)

    def lower_f(f, state_dict_jax, input, shardings):
      for k, v in state_dict_jax.items():
        if k == "weight":
          state_dict_jax[k] = v.astype(jnp.int4)
          state_dict_jax[k] = jax.device_put(v, sharding[0])
        if k == "weight_scaler":
          state_dict_jax[k] = jax.device_put(v, sharding[1])
      pre_opt = f.lower(state_dict_jax, input).as_text("hlo")
      post_opt = f.lower(state_dict_jax, input).compile().as_text()
      return post_opt

    shardings = [
        (sharding_by_axis(0), sharding_by_axis(0)),  # wq/wk/wv sharding
        (sharding_by_axis(2), sharding_by_axis(1)),  # wo sharding
        #  (sharding_by_axis(1), sharding_by_axis(0)), # bad sharding
    ]
    for sharding in shardings:
      opt_hlo = lower_f(f, state_dict_jax, input, sharding)
      self.assertFalse("all-to-all" in opt_hlo)
      self.assertFalse("all-reduce-scatter" in opt_hlo)

  # def test_quantize_tensor(self):
  #   w = torch.rand(512, 256)
  #   x = torch.rand(2, 256)
  #   w_q, s = quantize_blockwise(w.transpose(0, 1))
  #   w_dq = (w_q * s).reshape(-1, w.shape[0]).transpose(0, 1)
  #   print(torch.max((w_dq - w).abs()))
  #   print((w_dq - w).norm())
  #   with jax.default_device(jax.devices("cpu")[0]):
  #     w = torch_xla2.tensor.move_to_device(w)
  #     x = torch_xla2.tensor.move_to_device(x)
  #     # import pdb;pdb.set_trace()
  #     w_q, s = quantize_blockwise(w.transpose(0, 1))
  #     w_dq = (w_q * s).reshape(-1, w.shape[0]).transpose(0, 1)
  #     # w_dq = torch_xla2.tensor.j2t(w_dq._elem)
  #     # import pdb;pdb.set_trace()
  #     # print(torch.max((w_dq - w).abs()))
  #     # import pdb; pdb.set_trace()
  #     print((w_dq - w).norm())

  # def test_per_channel(self):
  #   with jax.default_device(jax.devices("cpu")[0]):
  #     n_bits = 6
  #     w = torch.rand((512, 256))
  #     x = torch.rand((2, 256))
  #     w_q, s = quantize_torch(w, (1,), n_bits=n_bits)
  #     w_dq = w_q * s
  #     print(torch.max((w_dq - w).abs()))
  #     print((w_dq - w).norm())

  #     w = torch_xla2.tensor.move_to_device(w)
  #     x = torch_xla2.tensor.move_to_device(x)
  #     w_q, s = quantize_torch(w, (1,), n_bits=n_bits)
  #     w_dq = w_q * s
  #     print((w_dq - w).norm())

  # def test_hex_llm_quant(self):
  #   from jetstream_pt.quantize import quantize_tensor, TensorQConfig

  #   print("test per-channel")
  #   qconfig = TensorQConfig(is_blockwise=False, axis=1)
  #   w = torch.rand((512, 256))
  #   w_q, s, _ = quantize_tensor(w, qconfig)
  #   w_dq = w_q * s
  #   print("my quant diff: ", w_dq - w)
  #   print("my quant norm: ", (w_dq - w).norm())
  #   with jax.default_device(jax.devices("cpu")[0]):
  #     w_q, s = quantize_torch(w, (1,))
  #     w_dq = w_q * s
  #     print("my quant diff: ", w_dq - w)
  #     print("my quant norm: ", (w_dq - w).norm())

  # def test_hex_llm_quant_blockwise(self):
  #   from jetstream_pt.quantize import quantize_tensor, TensorQConfig
  #   print("test per-channel")
  #   w = torch.rand((512, 256))
  #   qconfig = TensorQConfig(is_blockwise=False, axis=1)
  #   w_q, s, _ = quantize_tensor(w, qconfig)
  #   print(w_q.shape)
  #   print(s.shape)
  #   w_dq = w_q * s
  #   print("my quant diff: ", w_dq - w)
  #   print("my quant norm: ", (w_dq - w).norm())
  #   with jax.default_device(jax.devices("cpu")[0]):
  #     w_q, s = quantize_torch(w, (1,))
  #     w_dq = w_q * s
  #     print("my quant diff: ", w_dq - w)
  #     print("my quant norm: ", (w_dq - w).norm())

  #     print("test blockwise")
  #     qconfig = TensorQConfig(is_blockwise=True)


if __name__ == "__main__":
  unittest.main()
