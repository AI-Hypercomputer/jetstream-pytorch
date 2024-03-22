# Copyright 2024 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Any
import unittest
import torch
import torch_xla2
import jax.numpy as jnp

from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData
from jetstream_pt.engine import PyTorchEngine, Prefix, DecodeState

# This model will output tokens with value of 2
# and will update caches with value of 1.0
class Dummy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.params = None

    def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
    ):
        batch_size, seqlen = tokens.shape
        for cache in caches:
            cache.update(torch.ones((batch_size, seqlen)))
        return torch.ones((batch_size, seqlen), dtype=torch.int32) * 2
    



class EngineTest(unittest.TestCase):

    def _make_small_engine(self, quantize=False):
        env_data = JetEngineEnvironmentData()
        env_data.max_input_sequence_length = 128
        env_data.max_input_sequence_length = 128
        env_data.cache_sequence_length = 128
        env_data.model_type = 'llama-2-tiny'
        if quantize:
            env_data.enable_kv_quantization = True
            env_data.enable_weight_quantization = True

        env = JetEngineEnvironment(env_data)
        model = Dummy()
        model.params = env._model_arg  # llama's model arg

        engine = PyTorchEngine(model, env)
        return engine


    def test_insert(self):
        seqlen = 32
        engine = self._make_small_engine()
        env = engine.env

        heads, head_dim = env.num_heads, env.head_dim

        # Caches contains a sequence
        caches_to_write = torch_xla2.tensor.wrap(
            jnp.broadcast_to(
                jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
                (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
        )

        caches = [(caches_to_write, caches_to_write)
                  for _ in range(env.num_layers)]
        prefill_result = Prefix(
            jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
            caches,
            seqlen
        )
        decode_state = engine.init_decode_state()
        self.assertEqual(decode_state.current_position, 0)
        updated = engine.insert(
            prefill_result, decode_state, slot=jnp.int32(1)
        )
        self.assertEqual(decode_state.current_position, 0)

        _, hl, sl, dl = updated.caches[0][0].shape
        for k, v in updated.caches:
            for h in range(hl):
                for s in range(-seqlen, 0):
                    for d in range(dl):
                        self.assertEqual(k[1, h, s, d], seqlen + jnp.bfloat16(s))

    def test_insert2(self):
        seqlen = 32
        engine = self._make_small_engine()
        env = engine.env

        heads, head_dim = env.num_heads, env.head_dim

        # Caches contains a sequence
        caches_to_write = torch_xla2.tensor.wrap(
            jnp.broadcast_to(
                jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
                (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
        )

        caches = [(caches_to_write, caches_to_write)
                  for _ in range(env.num_layers)]
        prefill_result = Prefix(
            jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
            caches,
            seqlen
        )
        decode_state = engine.init_decode_state()
        decode_state = DecodeState(
            decode_state.tokens,
            decode_state.caches,
            decode_state.cache_scales,
            10, # current position 
            decode_state.lens,
        )
        updated = engine.insert(
            prefill_result, decode_state, slot=jnp.int32(1)
        )

        _, hl, sl, dl = updated.caches[0][0].shape
        for k, v in updated.caches:
            for h in range(hl):
                for s in range(-seqlen, 0):
                    exp_value = seqlen + jnp.bfloat16(s)
                    index = s + 10
                    for d in range(dl):
                        self.assertEqual(
                            k[1, h, index, d], seqlen + jnp.bfloat16(s))

    def test_insert3(self):
        seqlen = 32
        engine = self._make_small_engine(quantize=True)
        env = engine.env

        heads, head_dim = env.num_heads, env.head_dim

        # Caches contains a sequence
        caches_to_write = torch_xla2.tensor.wrap(
            jnp.broadcast_to(
                jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
                (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
        )

        caches = [(caches_to_write, caches_to_write)
                  for _ in range(env.num_layers)]
        prefill_result = Prefix(
            jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
            caches,
            seqlen
        )
        decode_state = engine.init_decode_state()
        decode_state = DecodeState(
            decode_state.tokens,
            decode_state.caches,
            decode_state.cache_scales,
            10, # current position 
            decode_state.lens,
        )
        updated = engine.insert(
            prefill_result, decode_state, slot=jnp.int32(1)
        )

        _, hl, sl, dl = updated.caches[0][0].shape
        for (k, v), (k_scale, v_scale) in zip(updated.caches, updated.cache_scales):
            for h in range(hl):
                for s in range(-seqlen, 0):
                    for d in range(dl):
                        inflated = k * k_scale
                        index = s + 10
                        self.assertTrue(
                            jnp.allclose(inflated[1, h, index, d], 
                            seqlen + jnp.bfloat16(s), atol=0.1, rtol=0.01), 
                            f"{inflated[1, h, index, d]} vs. {seqlen + jnp.bfloat16(s)}")

    def test_insert4(self):
        seqlen = 32
        engine = self._make_small_engine(quantize=True)
        env = engine.env

        heads, head_dim = env.num_heads, env.head_dim

        # Caches contains a sequence
        caches_to_write = torch_xla2.tensor.wrap(
            jnp.broadcast_to(
                jnp.arange(0, seqlen).reshape((1, 1, seqlen, 1)),
                (1, heads, seqlen, head_dim)).astype(jnp.bfloat16)
        )

        caches = [(caches_to_write, caches_to_write)
                  for _ in range(env.num_layers)]
        prefill_result = Prefix(
            jnp.zeros((1, ), dtype=jnp.int32), # doesn't matter
            caches,
            seqlen
        )
        decode_state = engine.init_decode_state()
        updated = engine.insert(
            prefill_result, decode_state, slot=jnp.int32(1)
        )

        _, hl, sl, dl = updated.caches[0][0].shape
        for (k, v), (k_scale, v_scale) in zip(updated.caches, updated.cache_scales):
            for h in range(hl):
                for s in range(-seqlen, 0):
                    for d in range(dl):
                        inflated = k * k_scale
                        self.assertTrue(
                            jnp.allclose(inflated[1, h, s, d], 
                            seqlen + jnp.bfloat16(s), atol=0.1, rtol=0.01), 
                            f"{inflated[1, h, s, d]} vs. {seqlen + jnp.bfloat16(s)}")


if __name__ == '__main__':
    unittest.main()