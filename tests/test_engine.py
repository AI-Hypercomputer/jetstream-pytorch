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
from tests import helpers


class EngineTest(unittest.TestCase):

  def setup(self):
    env, model_arg = helpers.make_env_tiny(bf16_enable=True)
    model_ours = model_exportable.Transformer(model_arg, env)
    engine = PyTorchEngine(pt_model=model_ours, env=env)
    engine.rng = jax.random.PRNGKey(0)
    return engine

  def test_sampling_2D(self):
    # test greedy
    engine = self.setup()
    self.assertEqual(engine.env.sampling_algorithm, "greedy")
    logits = jnp.array([[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]])
    token = engine._sampling(logits, batch_size=1)
    self.assertEqual(token, jnp.array([[0]]))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test weighted
    engine.env.sampling_algorithm = "weighted"
    engine.env.temperature = 5.0
    token = engine._sampling(logits, batch_size=1)
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test topk
    engine.env.sampling_algorithm = "topk"
    engine.env.temperature = 5.0
    engine.env.topk = 4
    token = engine._sampling(logits, batch_size=1)
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test nucleus
    engine.env.sampling_algorithm = "nucleus"
    engine.env.temperature = 0.0
    engine.env.nucleus_topp = 0.8
    token = engine._sampling(logits, batch_size=1)
    self.assertTrue(jnp.array_equal(token, jnp.array([[0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

  def test_sampling_3D(self):
    # test greedy
    engine = self.setup()
    self.assertEqual(engine.env.sampling_algorithm, "greedy")
    logits = jnp.array(
        [
            [[0.4, 0.3, 0.2, 0.1], [0.5, 0.6, 0.7, 0.8]],
            [[0.5, 0.6, 0.7, 0.8], [0.4, 0.3, 0.2, 0.1]],
        ]
    )
    token = engine._sampling(logits, batch_size=2)
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test weighted
    engine.env.sampling_algorithm = "weighted"
    engine.env.temperature = 10.0
    token = engine._sampling(logits, batch_size=2)
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [1]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test topk
    engine.env.sampling_algorithm = "topk"
    engine.env.temperature = 1.0
    engine.env.topk = 3
    token = engine._sampling(logits, batch_size=2)
    self.assertTrue(jnp.array_equal(token, jnp.array([[1], [0]])))
    self.assertTrue(jnp.isdtype(token, jnp.int32))

    # test nucleus
    engine.env.sampling_algorithm = "nucleus"
    engine.env.temperature = 1.0
    engine.env.nucleus_topp = 0.8
    token = engine._sampling(logits, batch_size=2)
    self.assertTrue(jnp.array_equal(token, jnp.array([[3], [1]])))
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
