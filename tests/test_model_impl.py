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

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest
import jax
import jax.numpy as jnp
import torch
import torch_xla2
from . import helpers

from jetstream_pt.third_party.llama import model_exportable
from jetstream_pt.third_party.llama import model_original
from jetstream_pt.third_party.gemma import model_original as gemma_orig
from jetstream_pt.third_party.gemma import model as gemma
from jetstream_pt import torchjax
from jetstream_pt import layers
from jetstream_pt import cache_manager


class ModelComponentTest(unittest.TestCase):
  """Test diff between original model and xla model for transformer,
  transformer block, attention and other component in model"""

  def setUp(self):
    """setup torch env"""
    jax.config.update("jax_enable_x64", False)
    torch.set_default_dtype(torch.float32)

  def _prefill_mask(self, seqlen, start_pos):
    mask = torch.full((seqlen, seqlen), float("-inf"))

    mask = torch.triu(mask, diagonal=1)

    # When performing key-value caching, we compute the attention scores
    # only for the new sequence. Thus, the matrix of scores is of size
    # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
    # j > cache_len + i, since row i corresponds to token cache_len + i.
    mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask])
    return mask

  def _make_freqs_cis(self, model_arg, seqlen, start_pos):
    freqs_cis = model_original.precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2
        # because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly
        # allows for dynamism of token lengths while training or fine-tuning.
        model_arg.dim // model_arg.n_heads,
        model_arg.max_seq_len * 2,
    )
    freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
    return freqs_cis

  def _to_xla_tensor(self, tree):
    return torch_xla2.default_env().to_xla(tree)

  def _call_xla_model(self, model, weights, args):
    with jax.default_device(jax.devices("cpu")[0]):
      xla_weights, xla_inputs = self._to_xla_tensor((weights, args))
      result = torch.func.functional_call(model, xla_weights, xla_inputs)
      result_torch = torch_xla2.tensor.j2t(result._elem)
      return result_torch

  def _generate_mask(self, cache_length, pos, seqlen):
    x = jnp.arange(0, cache_length)
    cond = jnp.logical_and(x <= pos, x >= pos - seqlen)
    res = jnp.where(cond, 0, float("-inf"))
    return torchjax.to_torch(res)

  def _compare_cache(self, cache_torch, cache_jax):
    _, seq, _, _ = cache_torch.shape
    cache_j = torch_xla2.tensor.j2t(cache_jax._elem)
    for s in range(seq):
      print("diff ", (cache_torch[0, s] - cache_j[0, :, s]).norm())

  def _make_one_cache_for_generate(self, env, pos):
    cache_array_k = jnp.zeros(env.cache_shape)

    cache_array_v = jnp.zeros(env.cache_shape)
    cache_array_k, cache_array_v = torchjax.to_torch(
        (cache_array_k, cache_array_v)
    )
    cache_decode = cache_manager.KVCacheGenerate(
        cache_array_k, cache_array_v, pos, None
    )
    return cache_decode

  # pylint: disable-next=all
  def test_attention(self):
    env, model_arg = helpers.make_env_tiny(False)

    attention_orig = model_original.Attention(model_arg)
    attention_ours = layers.Attention(
        n_heads=model_arg.n_heads,
        n_kv_heads=model_arg.n_kv_heads,
        head_dim=model_arg.dim // model_arg.n_heads,
        hidden_size=model_arg.dim,
        device="cpu",
        env=env,
    )

    seqlen = 32
    batch = 1
    x = torch.randn(
        (batch, seqlen, model_arg.dim)
    )  # (batch, seqlen, embedding dim)
    start_pos = 0
    freqs_cis = self._make_freqs_cis(model_arg, seqlen, start_pos)
    mask = self._prefill_mask(seqlen, start_pos)
    inputs_orig = (x, start_pos, freqs_cis, mask)

    expected_out = attention_orig(*inputs_orig)

    cache = cache_manager.KVCachePrefill()
    freqs_cis = freqs_cis.reshape(batch, seqlen, -1)
    input_ours = (
        x,
        freqs_cis,
        mask,
        cache,
    )

    result_torch = self._call_xla_model(
        attention_ours, attention_orig.state_dict(), input_ours
    )

    print("Single Attention: Diff norm", (result_torch - expected_out).norm())
    self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-4))

    pos = 32  #
    cache_decode = self._make_one_cache_for_generate(env, pos)

    # insert prefilled cache entry
    cache_decode.cache_k._elem = cache_decode.cache_k._elem.at[
        :, :, :pos, :
    ].set(cache.cache_k._elem)

    cache_decode.cache_v._elem = cache_decode.cache_v._elem.at[
        :, :, :pos, :
    ].set(cache.cache_v._elem)

    # self._compare_cache(attention_orig.cache_k, cache_decode.cache_k)
    # Now do one with decode
    x2 = torch.randn((batch, 1, model_arg.dim))
    freqs_cis = self._make_freqs_cis(model_arg, 1, 32)
    inputs_orig2 = (
        x2,
        pos,
        freqs_cis,
        None,  # mask is none for decode
    )
    expected_out = attention_orig(*inputs_orig2)
    cache_decode.pos = [pos]  # next position to update
    mask = self._generate_mask(env.cache_sequence_length, pos, seqlen)
    mask = mask.reshape(1, 1, 1, -1)  # seq dim is the last one
    freqs_cis = freqs_cis.reshape(batch, 1, -1)
    input_ours2 = (x2, freqs_cis, mask, cache_decode)
    result_torch = self._call_xla_model(
        attention_ours, attention_orig.state_dict(), input_ours2
    )

    print(
        "Single Attention: decode diff norm",
        (result_torch - expected_out).norm(),
    )
    self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-4))

  def test_gemma_attention(self):
    with jax.default_matmul_precision("float32"):
      env, model_arg = helpers.make_env_tiny(False)

      hidden_size = model_arg.dim
      num_heads = model_arg.n_heads
      num_kv_heads = model_arg.n_kv_heads
      head_dim = model_arg.dim // model_arg.n_heads
      # env._data.qkv_fusion = True

      def init_weights(model):
        state_dict = model.state_dict()
        res = {}
        for k, v in state_dict.items():
          # x = random.randint(1, 10)
          res[k] = torch.randn(v.shape)  # * x
        model.load_state_dict(res, assign=True)

      attention_orig = gemma_orig.GemmaAttention(
          hidden_size=hidden_size,
          num_heads=num_heads,
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
          quant=False,
      )
      init_weights(attention_orig)

      attention_ours = gemma.GemmaAttention(
          hidden_size=hidden_size,
          num_heads=num_heads,
          num_kv_heads=num_kv_heads,
          head_dim=head_dim,
          device="meta",
          env=env,
      )

      def load_hook(state_dict, prefix, *args):
        qkv = state_dict.pop(prefix + "qkv_proj.weight")
        q, k, v = qkv.split(
            [
                attention_orig.q_size,
                attention_orig.kv_size,
                attention_orig.kv_size,
            ],
            dim=0,
        )
        state_dict[prefix + "wq.weight"] = q
        state_dict[prefix + "wk.weight"] = k
        state_dict[prefix + "wv.weight"] = v

      seqlen = 32
      batch = 1
      x = torch.randn(
          (batch, seqlen, hidden_size)
      )  # (batch, seqlen, embedding dim)
      start_pos = 0
      freqs_cis = self._make_freqs_cis(model_arg, seqlen, start_pos)
      mask = self._prefill_mask(seqlen, start_pos)
      kv_write_indexes = torch.arange(0, seqlen)
      cache_k = torch.zeros((batch, seqlen, num_heads, head_dim))
      cache_v = torch.zeros((batch, seqlen, num_heads, head_dim))
      inputs_orig = (x, freqs_cis, kv_write_indexes, (cache_k, cache_v), mask)

      expected_out = attention_orig(*inputs_orig)

      cache = cache_manager.KVCachePrefill()
      freqs_cis = freqs_cis.reshape(batch, seqlen, -1)
      input_ours = (
          x,
          freqs_cis,
          mask,
          cache,
      )

      state_dict = dict(attention_orig.state_dict())
      load_hook(state_dict, "")
      result_torch = self._call_xla_model(
          attention_ours, state_dict, input_ours
      )

      print(
          "Single Gemma Attention: Diff norm",
          (result_torch - expected_out).norm(),
      )
      print(
          "Single Gemma Attention: Diff max",
          torch.max((result_torch - expected_out).abs()),
      )

      self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-3))

  # pylint: disable-next=all
  def test_transformer_block(self):
    env, model_arg = helpers.make_env_tiny(False)

    block_orig = model_original.TransformerBlock(0, model_arg)
    block_ours = model_exportable.TransformerBlock(0, model_arg, env)

    batch = 1
    seqlen = 32
    x = torch.randn(
        (batch, seqlen, model_arg.dim)
    )  # (batch, seqlen, embedding dim)
    start_pos = 0
    freqs_cis = self._make_freqs_cis(model_arg, seqlen, start_pos)
    mask = self._prefill_mask(seqlen, start_pos)
    inputs_orig = (x, start_pos, freqs_cis, mask)

    expected_out = block_orig(*inputs_orig)

    cache = cache_manager.KVCachePrefill()
    freqs_cis = freqs_cis.reshape(batch, seqlen, -1)
    input_ours = (
        x,
        freqs_cis,
        mask,
        cache,
    )

    result_torch = self._call_xla_model(
        block_ours, block_orig.state_dict(), input_ours
    )

    print("Single TransBlock: Diff norm", (result_torch - expected_out).norm())
    self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-4))

    pos = 32  #
    cache_decode = self._make_one_cache_for_generate(env, pos)

    # insert prefilled cache entry
    cache_decode.cache_k._elem = cache_decode.cache_k._elem.at[
        :, :, :pos, :
    ].set(cache.cache_k._elem)
    cache_decode.cache_v._elem = cache_decode.cache_v._elem.at[
        :, :, :pos, :
    ].set(cache.cache_v._elem)

    # Now do one with decode
    x2 = torch.randn((1, 1, model_arg.dim))
    freqs_cis = self._make_freqs_cis(model_arg, 1, 32)
    inputs_orig2 = (
        x2,
        pos,
        freqs_cis,
        None,  # mask is none for decode
    )
    expected_out = block_orig(*inputs_orig2)
    cache_decode.pos = [pos]  # next position to update
    mask = self._generate_mask(env.cache_sequence_length, pos, seqlen)
    mask = mask.reshape(1, 1, 1, -1)  # seq dim is the last one
    freqs_cis = freqs_cis.reshape(batch, 1, -1)
    input_ours2 = (x2, freqs_cis, mask, cache_decode)
    result_torch = self._call_xla_model(
        block_ours, block_orig.state_dict(), input_ours2
    )

    print(
        "Single Attention: decode diff norm",
        (result_torch - expected_out).norm(),
    )
    self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-4))

  # pylint: disable-next=all
  def test_transformer(self):
    """test transformer diff between original model vs xla_model"""
    env, model_arg = helpers.make_env_tiny(False)

    model_orig = model_original.Transformer(model_arg)
    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis
    model_ours = model_exportable.Transformer(model_arg, env)

    seqlen = 32
    x = torch.randint(0, 32000, (1, seqlen))  # (batch, seqlen, embedding dim)
    start_pos = 0
    mask = self._prefill_mask(seqlen, start_pos)
    inputs_orig = (x, start_pos)

    expected_out = model_orig(*inputs_orig)

    caches = env.make_caches_prefill()
    input_pos = torch.arange(0, seqlen)
    input_ours = (
        x,
        input_pos,
        caches,
        mask,
    )

    result_torch = self._call_xla_model(model_ours, state_dict, input_ours)

    print("Transformer: Diff norm", (result_torch - expected_out).norm())
    self.assertTrue(torch.allclose(result_torch, expected_out, atol=1e-4))


if __name__ == "__main__":
  unittest.main()
