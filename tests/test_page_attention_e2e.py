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


from typing import List, Tuple
import unittest
import os

import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch_xla2
from torch.utils import _pytree as pytree
from absl.testing import parameterized

from jetstream_pt.engine import PyTorchEngine, DecodeState
from jetstream_pt.third_party.llama import model_exportable, model_args
from jetstream_pt.third_party.llama.generation_original import LlamaOriginal
from jetstream_pt import environment
from tests import helpers


class LlamaE2ETest(parameterized.TestCase):
  """This test class includes all E2E test for llama2"""

  def _from_torch(self, tree):
    return pytree.tree_map_only(torch.Tensor, torch_xla2.tensor.t2j, tree)

  def _get_params(self, env, model_arg):
    seed = 1
    torch.manual_seed(1)

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)

    model_orig = llama_original.model

    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis

    model_ours = model_exportable.Transformer(model_arg, env)

    for k, v in model_ours.state_dict().items():
      if "scale" in k:
        state_dict[k] = helpers.to_xla_tensor(v)

    params = self._from_torch(state_dict)
    return params    

    # pylint: disable-next=all
  def _insert(self, engine, params):
    # pylint: disable-next=all
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))

    decode_state = engine.init_decode_state()
    decode_state = DecodeState(
      decode_state.tokens,
      decode_state.caches,
      decode_state.cache_scales,
      10,
      decode_state.lens,
      decode_state.start,
      decode_state.input_pos, 
      decode_state.mask,
    )
  
    slot = 0
    # pylint: disable-next=all
    prefill_result, _ = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )

    # pylint: disable-next=all
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    return decode_state
  
  # pylint: disable-next=all
  def _decode(self, engine, params):
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))

    decode_state = engine.init_decode_state()
    decode_state = DecodeState(
      decode_state.tokens,
      decode_state.caches,
      decode_state.cache_scales,
      10,
      decode_state.lens,
      decode_state.start,
      decode_state.input_pos, 
      decode_state.mask,
    )
  
    slot = 0
    # pylint: disable-next=all
    prefill_result, _ = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )

    # pylint: disable-next=all
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)
    decode_state = engine.generate(params, decode_state)
    return decode_state

    # out_tokens = []
    # while True:
    #   # pylint: disable-next=all
    #   decode_state, result_tokens = engine.generate(params, decode_state)
    #   slot_data = result_tokens.get_result_at_slot(slot)
    #   slot_tokens = slot_data.tokens
    #   slot_lengths = slot_data.lengths

    #   token_id = slot_tokens[slot, 0].item()
    #   out_tokens.append(token_id)
    #   if slot_lengths > max_output_length:
    #     break
    # return out_tokens, expected_output_tokens[0]

  
  def get_compress_kv_cache(
      self,
      decode,
      slot,
  ) -> List[Tuple[jax.Array, jax.Array]]:
    decode_caches = decode.caches
    len = decode.input_pos[slot]
    return [(k[slot, :, 0:len, :], v[slot, :, 0:len, :]) for k, v in decode_caches]

  # def test_prefill(self):
  #   "end to end jetstream llama test with bfloat16"
  #   jax.config.update("jax_platform_name", "cpu")
  #   jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)
  #   print(f"---------> {jax.devices()}")

  #   env, model_arg = helpers.make_env_tiny(bf16_enable=True)
  #   model = model_exportable.Transformer(model_arg, env)
  #   engine = PyTorchEngine(pt_model=model, env=env)
  #   params = self._get_params(env, model_arg)
    
  #   page_env, page_model_arg = helpers.make_page_attention_env_tiny(bf16_enable=True)
  #   page_model = model_exportable.Transformer(page_model_arg, page_env)
  #   page_engine = PyTorchEngine(pt_model=page_model, env=page_env)
  #   page_params = self._get_params(page_env, page_model_arg)
    
  #   decode = self._insert(engine, params)
  #   page_decode = self._insert(page_engine, page_params)
    
  #   cache = self.get_compress_kv_cache(decode, 0)
  #   page_caches = page_engine.page_attention_manager.get_compress_kv_cache(page_decode.caches, 0)
    
  #   self.assertEqual(decode.tokens, page_decode.tokens)
  #   for (k, v), (pk, pv) in zip(cache, page_caches):
  #     self.assertTrue(jnp.array_equal(k, pk))
  #     self.assertTrue(jnp.array_equal(v, pv))

  def test_decode(self):
    "end to end jetstream llama test with bfloat16"
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)
    print(f"---------> {jax.devices()}")

    env, model_arg = helpers.make_env_tiny(bf16_enable=True)
    model = model_exportable.Transformer(model_arg, env)
    engine = PyTorchEngine(pt_model=model, env=env)
    params = self._get_params(env, model_arg)
    # params = engine.load_params()
    
    page_env, page_model_arg = helpers.make_page_attention_env_tiny(bf16_enable=True)
    page_model = model_exportable.Transformer(page_model_arg, page_env)
    page_engine = PyTorchEngine(pt_model=page_model, env=page_env)
    page_params = self._get_params(page_env, page_model_arg)
    # page_params = page_engine.load_params()
    
    #decode = self._decode(engine, params)
    page_decode = self._decode(page_engine, page_params)
    
    # cache = self.get_compress_kv_cache(decode, 0)
    # page_caches = page_engine.page_attention_manager.get_compress_kv_cache(page_decode.caches, 0)
    
    # self.assertEqual(decode.tokens, page_decode.tokens)
    # for (k, v), (pk, pv) in zip(cache, page_caches):
    #   self.assertTrue(jnp.array_equal(k, pk))
    #   self.assertTrue(jnp.array_equal(v, pv))      



  

if __name__ == "__main__":
  unittest.main()
