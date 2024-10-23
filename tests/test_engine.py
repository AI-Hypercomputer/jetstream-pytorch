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

# pylint: disable=all

import unittest
import jax
import jax.numpy as jnp

from jetstream_pt.third_party.llama import model_exportable
from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.engine import DecodeState
from jetstream_pt.engine import Prefix
from tests import helpers
from jetstream_pt import cache_manager


class MockEngine(PyTorchEngine):

  def _call_model_prefill(self, weights, tokens, input_indexes):
    caches = [
        cache_manager.KVCachePrefill(
            self.env.quant_config.enable_kv_quantization
        )
        for _ in self.pt_model.layers
    ]
    # logits = jnp.ones((self.env.batch_size, 1), jnp.float32)
    assert (
        self.env.batch_size == 1
    ), f"The batch size {self.env.batch_size} != 1"
    logits = jnp.array([[0.5, 0.6, 0.7, 0.8]])
    return logits, caches

  def _call_model_generate(
      self,
      weights,
      tokens,
      input_indexes,
      caches,
      cache_scales,
      mask,
      start,
      input_pos,
      ragged_batch_index,
      ragged_block_index,
      page_token_indices,
  ):
    logits = jnp.array(
        [
            [[0.5, 0.6, 0.7, 0.8]],
            # [[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]],
            [[0.4, 0.3, 0.2, 0.1]],
        ]
    )
    return logits, caches, cache_scales


class EngineTest(unittest.TestCase):

  def setup(self, batch_size=1):
    env, model_arg = helpers.make_env_tiny(
        bf16_enable=True, batch_size=batch_size
    )
    model_ours = model_exportable.Transformer(model_arg, env)
    engine = MockEngine(pt_model=model_ours, env=env)
    engine.rng = jax.random.key(0)
    return engine

  def test_sampling_2D(self):
    # test greedy
    engine = self.setup()
    self.assertEqual(engine.env.sampling_algorithm, "greedy")
    logits = jnp.array([[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]])
    token = engine._sampling(
        logits, "greedy", engine.rng, temperature=1.0, topk=1, nucleus_topp=0.0
    )
    self.assertEqual(token, jnp.array([[0]]))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test weighted
    engine.env.sampling_algorithm = "weighted"
    engine.env.temperature = 5.0
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        temperature=5.0,
        topk=1,
        nucleus_topp=0.0,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test topk
    engine.env.sampling_algorithm = "topk"
    engine.env.temperature = 5.0
    engine.env.topk = 4
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        temperature=5.0,
        topk=4,
        nucleus_topp=0.0,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test nucleus
    engine.env.sampling_algorithm = "nucleus"
    engine.env.temperature = 1.0
    engine.env.nucleus_topp = 0.8
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        temperature=0.0,
        topk=1,
        nucleus_topp=0.8,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

  def test_sampling_3D(self):
    # test greedy
    engine = self.setup(batch_size=2)

    self.assertEqual(engine.env.sampling_algorithm, "greedy")
    logits = jnp.array(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8]],
            [[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]],
        ]
    )
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        engine.env.temperature,
        engine.env.topk,
        engine.env.nucleus_topp,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test weighted
    engine.env.sampling_algorithm = "weighted"
    engine.env.temperature = 10.0
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        engine.env.temperature,
        engine.env.topk,
        engine.env.nucleus_topp,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [1]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test topk
    engine.env.sampling_algorithm = "topk"
    engine.env.temperature = 1.0
    engine.env.topk = 3
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        engine.env.temperature,
        engine.env.topk,
        engine.env.nucleus_topp,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[1], [0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test nucleus
    engine.env.sampling_algorithm = "nucleus"
    engine.env.temperature = 1.0
    engine.env.nucleus_topp = 0.8
    token = engine._sampling(
        logits,
        engine.env.sampling_algorithm,
        engine.rng,
        engine.env.temperature,
        engine.env.topk,
        engine.env.nucleus_topp,
    )
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [1]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

  def test_custom_sampling_3D(self):
    engine = self.setup(batch_size=2)
    engine.rng = jax.random.key(3)
    engine.splited_rngs = jax.random.split(
        engine.rng, num=engine.env.batch_size
    )
    engine.env.sampling_algorithm = ""

    # Need a different engine of batch size of 1 to reshape the output
    engine_b1 = self.setup()
    engine_b1.rng = jax.random.key(3)
    logits = jnp.array(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8]],
            [[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]],
        ]
    )

    # test greedy
    token = engine._custom_sampling(
        logits,
        jnp.array([0, 0]),
        engine.splited_rngs,
        temperature=jnp.array([[0.0], [0.0]]),
        topk=jnp.array([[0], [0]]),
        nucleus_topp=jnp.array([[0.0], [0.0]]),
    )
    original_tokens = []
    for i in range(2):
      original_token = engine_b1._sampling(
          logits[i],
          "greedy",
          engine.splited_rngs[i],
          temperature=1.0,
          topk=0,
          nucleus_topp=0.0,
      )
      original_tokens.append(original_token)
    original_tokens = jnp.concatenate(original_tokens)

    print(f"custom sampling token {token} vs original tokens {original_tokens}")
    self.assertTrue(jnp.array_equal(token, original_tokens))
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test weighted
    engine.env.sampling_algorithm = "weighted"
    token = engine._custom_sampling(
        logits,
        jnp.array([1, 1]),
        engine.splited_rngs,
        temperature=jnp.array([1, 1]),
        topk=jnp.array([0, 0]),
        nucleus_topp=jnp.array([0.0, 0.0]),
    )
    original_tokens = []
    for i in range(2):
      original_token = engine_b1._sampling(
          logits[i],
          "weighted",
          engine.splited_rngs[i],
          temperature=1,
          topk=0,
          nucleus_topp=0.0,
      )
      original_tokens.append(original_token)
    original_tokens = jnp.concatenate(original_tokens)

    print(f"custom sampling token {token} vs original tokens {original_tokens}")
    self.assertTrue(jnp.array_equal(token, original_tokens))
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [2]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test topk
    engine.env.sampling_algorithm = "topk"
    token = engine._custom_sampling(
        logits,
        jnp.array([3, 3]),
        engine.splited_rngs,
        temperature=jnp.array([[1.0], [1.0]]),
        topk=jnp.array([[3], [3]]),
        nucleus_topp=jnp.array([[0.0], [0.0]]),
    )
    original_tokens = []
    for i in range(2):
      original_token = engine_b1._sampling(
          logits[i],
          "topk",
          engine.splited_rngs[i],
          temperature=1.0,
          topk=3,
          nucleus_topp=0.0,
      )
      original_tokens.append(original_token)
    original_tokens = jnp.concatenate(original_tokens)

    print(f"custom sampling token {token} vs original tokens {original_tokens}")
    self.assertTrue(jnp.array_equal(token, original_tokens))
    self.assertTrue(jnp.array_equal(token, jnp.array([[1], [2]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test nucleus
    engine.env.sampling_algorithm = "nucleus"
    token = engine._custom_sampling(
        logits,
        jnp.array([2, 2]),
        engine.splited_rngs,
        temperature=jnp.array([[1.0], [1.0]]),
        topk=jnp.array([[0], [0]]),
        nucleus_topp=jnp.array([[0.8], [0.8]]),
    )

    original_tokens = []
    for i in range(2):
      original_token = engine_b1._sampling(
          logits[i],
          "nucleus",
          engine.splited_rngs[i],
          temperature=1.0,
          topk=0,
          nucleus_topp=0.8,
      )
      original_tokens.append(original_token)
    original_tokens = jnp.concatenate(original_tokens)
    print(f"custom sampling token {token} vs original tokens {original_tokens}")
    self.assertTrue(jnp.array_equal(token, original_tokens))
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [2]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test greedy + topk
    token = engine._custom_sampling(
        logits,
        jnp.array([0, 3]),
        engine.splited_rngs,
        temperature=jnp.array([[0.0], [1.0]]),
        topk=jnp.array([[0], [3]]),
        nucleus_topp=jnp.array([[0.0], [0.0]]),
    )
    original_tokens = []

    i = 0
    original_token = engine_b1._sampling(
        logits[i],
        "greedy",
        engine.splited_rngs[i],
        temperature=0.0,
        topk=0,
        nucleus_topp=0.8,
    )
    original_tokens.append(original_token)

    i = 1
    original_token = engine_b1._sampling(
        logits[i],
        "topk",
        engine.splited_rngs[i],
        temperature=1.0,
        topk=3,
        nucleus_topp=0.0,
    )
    original_tokens.append(original_token)

    original_tokens = jnp.concatenate(original_tokens)

    print(f"custom sampling token {token} vs original tokens {original_tokens}")
    self.assertTrue(jnp.array_equal(token, original_tokens))
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [2]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

  def test_prefill_with_custom_sampling(self):
    engine = self.setup()
    engine.env.sampling_algorithm = ""

    # Inputs doesn't matter
    params = jnp.zeros((1,), jnp.float32)
    padded_tokens = jnp.zeros((1,), jnp.float32)
    true_length = 1

    # Greedy
    # algorithm, temperature, topk, nucleus_topp
    sampler = [0, 1.0, 3, 0.8]
    prefix, _ = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
    )
    token = prefix.token
    print(f"Greedy output: {token}")
    self.assertTrue(jnp.array_equal(token, jnp.array([[3]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    print(
        f"prefix sampler config {prefix.sampler_config} vs sampler {jnp.array(sampler)}"
    )
    self.assertAlmostEqual(
        prefix.sampler_config.all(), jnp.array(sampler).all()
    )

    # Weighted
    sampler = [1, 10.0, 3, 0.8]
    prefix, _ = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
    )
    token = prefix.token
    print(f"Weighted output: {token}")
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))
    self.assertAlmostEqual(
        prefix.sampler_config.all(), jnp.array(sampler).all()
    )

    # Nucleus
    sampler = [2, 1.0, 3, 0.0]
    prefix, _ = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
    )
    token = prefix.token
    print(f"Topk output: {token}")
    self.assertTrue(jnp.array_equal(token, jnp.array([[3]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))
    self.assertAlmostEqual(
        prefix.sampler_config.all(), jnp.array(sampler).all()
    )

    # Topk
    sampler = [3, 1.0, 3, 0.8]
    prefix, _ = engine.prefill(
        params=params,
        padded_tokens=padded_tokens,
        true_length=true_length,
        sampler=sampler,
    )
    token = prefix.token
    print(f"Nucleus output: {token}")
    self.assertTrue(jnp.array_equal(token, jnp.array([[3]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))
    self.assertAlmostEqual(
        prefix.sampler_config.all(), jnp.array(sampler).all()
    )

  def test_insert_no_wrap_with_custom_sampling(self):
    engine = self.setup()
    engine.env.sampling_algorithm = ""
    engine.env.batch_size = 2
    cache_shape = engine.env.cache_shape

    sampler_config_raw = [0, 1.0, 3, 0.8]
    sampler_config = jnp.array(sampler_config_raw)

    prefill_cache_shape = (1, cache_shape[1], 16, cache_shape[3])
    prefill_cache = []
    for _ in range(engine.env.num_layers):
      prefill_cache.append(
          (
              jnp.ones(prefill_cache_shape, dtype=jnp.bfloat16),
              jnp.ones(prefill_cache_shape, dtype=jnp.bfloat16),
          )
      )

    prefix = Prefix(
        token=jnp.ones((1)),
        caches=prefill_cache,
        seq_len=16,
        sampler_config=sampler_config,
    )

    doesnt_matter = jnp.array([0])
    kv_cache = engine.env.make_caches_generate()
    kv_cache = [c.state() for c in kv_cache]

    decode_state = DecodeState(
        tokens=jnp.zeros((engine.env.batch_size, 1)),
        caches=kv_cache,
        cache_scales=[doesnt_matter],
        current_position=16,
        lens=jnp.zeros((engine.env.batch_size, 1)),
        start=jnp.zeros((engine.env.batch_size, 1)),
        input_pos=jnp.zeros((engine.env.batch_size,)),
        mask=jnp.zeros((engine.env.batch_size, 128)),
        sampler_config=jnp.zeros((engine.env.batch_size, 4)),
    )

    # Insert to slot 1
    result_decode_state = engine._insert_no_wrap(prefix, decode_state, slot=1)

    self.assertAlmostEqual(
        result_decode_state.tokens.all(), decode_state.tokens.all()
    )
    self.assertAlmostEqual(
        result_decode_state.sampler_config.all(),
        jnp.array([[0, 0, 0, 0], sampler_config_raw]).all(),
    )

  def test_decode_with_custom_sampling(self):
    engine = self.setup(batch_size=2)
    engine.rng = jax.random.key(3)
    engine.splited_rngs = jax.random.split(
        engine.rng, num=engine.env.batch_size
    )
    engine.env.sampling_algorithm = ""

    # Inputs doesn't matter
    doesnt_matter = jnp.array([0])
    params = doesnt_matter

    decode_state = DecodeState(
        tokens=jnp.zeros((engine.env.batch_size, 1)),
        caches=[doesnt_matter],
        cache_scales=[doesnt_matter],
        current_position=0,
        lens=jnp.zeros((engine.env.batch_size, 1)),
        start=doesnt_matter,
        input_pos=jnp.zeros((engine.env.batch_size,)),
        mask=jnp.zeros((engine.env.batch_size, 1)),
        sampler_config=jnp.array([[0, 0.0, 0, 0.0], [3, 1.0, 3, 0.0]]),
    )

    # Topk + Weighted
    # algorithm, temperature, topk, nucleus_topp
    decode_state, _ = engine.generate_impl(
        params=params, decode_state=decode_state
    )
    token = decode_state.tokens
    print(f"Greedy output: {token}")
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [2]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))


#     def test_insert(self):
#         seqlen = 32
#         engine = self._make_small_engine()
#         env = engine.env

#         heads, head_dim = env.num_heads, env.head_dim

#         # Caches contains a sequence
#         caches_to_write = torch_xla2.tensor.wrap(
#             jnp.broadcast_to(
#                 jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
#                 (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
#         )

#         caches = [(caches_to_write, caches_to_write)
#                   for _ in range(env.num_layers)]
#         prefill_result = Prefix(
#             jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
#             caches,
#             seqlen
#         )
#         decode_state = engine.init_decode_state()
#         updated = engine.insert(
#             prefill_result, decode_state, slot=jnp.int32(1)
#         )

#         _, hl, sl, dl = updated.caches[0][0].shape
#         for k, v in updated.caches:
#             for h in range(hl):
#                 for s in range(-seqlen, 0):
#                     for d in range(dl):
#                         self.assertEqual(k[1, h, s, d], seqlen + jnp.bfloat16(s))

#     def test_insert2(self):
#         seqlen = 32
#         engine = self._make_small_engine()
#         env = engine.env

#         heads, head_dim = env.num_heads, env.head_dim

#         # Caches contains a sequence
#         caches_to_write = torch_xla2.tensor.wrap(
#             jnp.broadcast_to(
#                 jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
#                 (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
#         )

#         caches = [(caches_to_write, caches_to_write)
#                   for _ in range(env.num_layers)]
#         prefill_result = Prefix(
#             jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
#             caches,
#             seqlen
#         )
#         decode_state = engine.init_decode_state()
#         decode_state = DecodeState(
#             decode_state.tokens,
#             decode_state.caches,
#             decode_state.cache_scales,
#             10, # current position
#             decode_state.lens,
#             decode_state.validity,
#             decode_state.true_length,
#             decode_state.mask,
#         )
#         updated = engine.insert(
#             prefill_result, decode_state, slot=jnp.int32(1)
#         )

#         _, hl, sl, dl = updated.caches[0][0].shape
#         for k, v in updated.caches:
#             for h in range(hl):
#                 for s in range(-seqlen, 0):
#                     exp_value = seqlen + jnp.bfloat16(s)
#                     index = s + 10
#                     for d in range(dl):
#                         self.assertEqual(
#                             k[1, h, index, d], seqlen + jnp.bfloat16(s))

#     def test_insert3(self):
#         seqlen = 32
#         engine = self._make_small_engine(quantize=True)
#         env = engine.env

#         heads, head_dim = env.num_heads, env.head_dim

#         # Caches contains a sequence
#         caches_to_write = torch_xla2.tensor.wrap(
#             jnp.broadcast_to(
#                 jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
#                 (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
#         )

#         caches = [(caches_to_write, caches_to_write)
#                   for _ in range(env.num_layers)]
#         prefill_result = Prefix(
#             jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
#             caches,
#             seqlen
#         )
#         decode_state = engine.init_decode_state()
#         decode_state = DecodeState(
#             decode_state.tokens,
#             decode_state.caches,
#             decode_state.cache_scales,
#             10, # current position
#             decode_state.lens,
#         )
#         updated = engine.insert(
#             prefill_result, decode_state, slot=jnp.int32(1)
#         )

#         _, hl, sl, dl = updated.caches[0][0].shape
#         for (k, v), (k_scale, v_scale) in zip(updated.caches, updated.cache_scales):
#             for h in range(hl):
#                 for s in range(-seqlen, 0):
#                     for d in range(dl):
#                         inflated = k * k_scale
#                         index = s + 10
#                         self.assertTrue(
#                             jnp.allclose(inflated[1, h, index, d],
#                             seqlen + jnp.bfloat16(s), atol=0.1, rtol=0.01),
#                             f"{inflated[1, h, index, d]} vs. {seqlen + jnp.bfloat16(s)}")

#     def test_insert4(self):
#         seqlen = 32
#         engine = self._make_small_engine(quantize=True)
#         env = engine.env

#         heads, head_dim = env.num_heads, env.head_dim

#         # Caches contains a sequence
#         caches_to_write = torch_xla2.tensor.wrap(
#             jnp.broadcast_to(
#                 jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
#                 (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
#         )

#         caches = [(caches_to_write, caches_to_write)
#                   for _ in range(env.num_layers)]
#         prefill_result = Prefix(
#             jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
#             caches,
#             seqlen
#         )
#         decode_state = engine.init_decode_state()
#         updated = engine.insert(
#             prefill_result, decode_state, slot=jnp.int32(1)
#         )

#         _, hl, sl, dl = updated.caches[0][0].shape
#         for (k, v), (k_scale, v_scale) in zip(updated.caches, updated.cache_scales):
#             for h in range(hl):
#                 for s in range(-seqlen, 0):
#                     for d in range(dl):
#                         inflated = k * k_scale
#                         self.assertTrue(
#                             jnp.allclose(inflated[1, h, s, d],
#                             seqlen + jnp.bfloat16(s), atol=0.1, rtol=0.01),
#                             f"{inflated[1, h, s, d]} vs. {seqlen + jnp.bfloat16(s)}")

#     def test_tiny(self):
#         engine = self._make_small_engine()
#         model_arg = engine.env._model_arg
#         model_orig = model_original.Transformer(model_arg)
#         state_dict = dict(model_orig.state_dict())
#         state_dict['freqs_cis'] = model_orig.freqs_cis
#         model_ours = model_exportable.Transformer(model_arg, env)
#         engine.pt_model = model_ours

#         # prefill


if __name__ == "__main__":
  unittest.main()
