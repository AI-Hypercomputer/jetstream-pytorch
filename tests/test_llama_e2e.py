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

import torch
import os
from torch.utils import _pytree as pytree
import torch_xla2
import jax
import jax.numpy as jnp
import numpy as np
from jetstream_pt.engine import PyTorchEngine
from jetstream_pt.third_party.llama2 import model_exportable
from jetstream_pt.third_party.llama2.generation_original import LlamaOriginal
from jetstream_pt import environment


import unittest


class LlamaE2ETest(unittest.TestCase):

  def setup(self):
    torch.set_default_dtype(torch.bfloat16)

  def _to_jax(self, tree):
    return pytree.tree_map_only(torch.Tensor, torch_xla2.tensor.t2j, tree)

  def _make_env(self, bf16_enable=True):
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    jax.config.update("jax_dynamic_shapes", False)
    jax.config.update("jax_traceback_filtering", "off")
    env_data = environment.JetEngineEnvironmentData()
    env_data.max_input_sequence_length = 128
    env_data.max_input_sequence_length = 128
    env_data.cache_sequence_length = 128
    env_data.model_type = "llama-2-tiny"
    env_data.batch_size = 1
    env_data.bf16_enable = bf16_enable
    env = environment.JetEngineEnvironment(env_data)
    env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
    return env

  def test_original_llama2_seed(self):
    jax.config.update("jax_platform_name", "cpu")
    x = jnp.square(2)
    print(f"---------> {jax.devices()}")
    torch.set_default_dtype(torch.bfloat16)
    env = self._make_env()
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )
    output_tokens_multiple = []
    for i in [1, 999, 99999]:
      llama_original = LlamaOriginal.build(tokenizer_path, model_arg, i)
      prompt_tokens = [tokens]
      output_tokens = llama_original.generate(prompt_tokens, 10)
      output_tokens_multiple.append(output_tokens)

    for index, output_tokens in enumerate(output_tokens_multiple):
      print(f"------------------- index: {index}, tokens:{output_tokens}")
      if index > 0:
        self.assertNotEqual(
            output_tokens_multiple[index], output_tokens_multiple[index - 1]
        )

  def test_jetstream_llama2_seed(self):
    jax.config.update("jax_platform_name", "cpu")
    x = jnp.square(2)
    print(f"---------> {jax.devices()}")

    torch.set_default_dtype(torch.bfloat16)
    env = self._make_env()
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    max_output_length = 10

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    seed = 1
    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    model_orig = llama_original.model

    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis
    params = self._to_jax(state_dict)

    output_tokens_multiple = []
    for i in [1, 2, 3]:
      torch.manual_seed(1)
      model_ours = model_exportable.Transformer(model_arg, env)
      engine = PyTorchEngine(pt_model=model_ours, env=env)

      decode_state = engine.init_decode_state()
      slot = 0
      prefill_result = engine.prefill(
          params=params, padded_tokens=padded_tokens, true_length=true_length
      )

      decode_state = engine.insert(prefill_result, decode_state, slot=slot)

      out_tokens = []
      while True:
        decode_state, result_tokens = engine.generate(params, decode_state)
        slot_data = result_tokens.get_result_at_slot(slot)
        slot_tokens = slot_data.tokens
        slot_lengths = slot_data.lengths

        token_id = slot_tokens[slot, 0].item()
        out_tokens.append(token_id)
        if slot_lengths > max_output_length:
          break

      output_tokens_multiple.append(out_tokens)

    for index, output_tokens in enumerate(output_tokens_multiple):
      print(f"------------------- index: {index}, tokens:{output_tokens}")
      if index > 0:
        self.assertEqual(
            output_tokens_multiple[index], output_tokens_multiple[index - 1]
        )

  def _llama_e2e(self, env):
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    torch.manual_seed(1)
    max_output_length = 32

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    prompt_tokens = [tokens]
    expected_output_tokens = llama_original.generate(
        prompt_tokens, max_output_length
    )

    model_orig = llama_original.model

    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis

    model_ours = model_exportable.Transformer(model_arg, env)

    engine = PyTorchEngine(pt_model=model_ours, env=env)

    params = self._to_jax(state_dict)
    decode_state = engine.init_decode_state()
    slot = 0

    prefill_result = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )

    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    out_tokens = []
    while True:
      decode_state, result_tokens = engine.generate(params, decode_state)
      slot_data = result_tokens.get_result_at_slot(slot)
      slot_tokens = slot_data.tokens
      slot_lengths = slot_data.lengths

      token_id = slot_tokens[slot, 0].item()
      out_tokens.append(token_id)
      if slot_lengths > max_output_length:
        break
    return out_tokens, expected_output_tokens[0]

  def test_llama_e2e_float32(self):
    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")

    env = self._make_env(bf16_enable=False)
    out_tokens, expected_output_tokens = self._llama_e2e(env)
    self.assertEqual(out_tokens, expected_output_tokens)

  def test_llama_e2e_bfloat16(self):
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)
    print(f"---------> {jax.devices()}")

    env = self._make_env(bf16_enable=True)
    out_tokens, expected_output_tokens = self._llama_e2e(env)
    self.assertNotEqual(out_tokens, expected_output_tokens)

  def test_llama_e2e_two_addtional_tokens(self):
    jax.config.update("jax_platform_name", "cpu")
    x = jnp.square(2)
    print(f"---------> {jax.devices()}")

    torch.set_default_dtype(torch.bfloat16)
    env = self._make_env()
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    tokens = np.append(tokens, [15050, 3503], axis=-1)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    torch.manual_seed(1)
    max_output_length = 10

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    prompt_tokens = [tokens]
    expected_output_tokens = llama_original.generate(
        prompt_tokens, max_output_length
    )

    model_orig = llama_original.model

    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis

    model_ours = model_exportable.Transformer(model_arg, env)

    engine = PyTorchEngine(pt_model=model_ours, env=env)

    params = self._to_jax(state_dict)
    decode_state = engine.init_decode_state()
    slot = 0

    prefill_result = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )

    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    out_tokens = []
    while True:
      decode_state, result_tokens = engine.generate(params, decode_state)
      slot_data = result_tokens.get_result_at_slot(slot)
      slot_tokens = slot_data.tokens
      slot_lengths = slot_data.lengths

      token_id = slot_tokens[slot, 0].item()
      out_tokens.append(token_id)
      if slot_lengths > max_output_length:
        break
    print(f"-------------------->out_tokens:{out_tokens}")
    print(
        f"-------------------->expected_output_tokens:{expected_output_tokens}"
    )
    # self.assertEqual(out_tokens ,expected_output_tokens)

  def test_llama_e2e_four_addtional_tokens(self):
    jax.config.update("jax_platform_name", "cpu")
    x = jnp.square(2)
    print(f"---------> {jax.devices()}")

    torch.set_default_dtype(torch.bfloat16)
    env = self._make_env()
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    tokens = np.append(tokens, [15050, 3503, 11833, 28551], axis=-1)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    torch.manual_seed(1)
    max_output_length = 10

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    prompt_tokens = [tokens]
    expected_output_tokens = llama_original.generate(
        prompt_tokens, max_output_length
    )

    model_orig = llama_original.model

    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis

    model_ours = model_exportable.Transformer(model_arg, env)

    engine = PyTorchEngine(pt_model=model_ours, env=env)

    params = self._to_jax(state_dict)
    decode_state = engine.init_decode_state()
    slot = 0

    prefill_result = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )

    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    out_tokens = []
    while True:
      decode_state, result_tokens = engine.generate(params, decode_state)
      slot_data = result_tokens.get_result_at_slot(slot)
      slot_tokens = slot_data.tokens
      slot_lengths = slot_data.lengths

      token_id = slot_tokens[slot, 0].item()
      out_tokens.append(token_id)
      if slot_lengths > max_output_length:
        break
    print(f"-------------------->out_tokens:{out_tokens}")
    print(
        f"-------------------->expected_output_tokens:{expected_output_tokens}"
    )
    # self.assertEqual(out_tokens ,expected_output_tokens)

  def test_llama_with_original_prefill_decode_32(self):
    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")

    torch.set_default_dtype(torch.float32)
    env = self._make_env(bf16_enable=False)
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    torch.manual_seed(1)
    max_output_length = 5

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    model_orig = llama_original.model
    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis
    model_ours = model_exportable.Transformer(model_arg, env)
    engine = PyTorchEngine(pt_model=model_ours, env=env)
    params = self._to_jax(state_dict)
    slot = 0
    expected_output_tokens = []
    out_tokens = []

    prompt_tokens = [tokens]
    decode_state_original = llama_original.prefill(
        prompt_tokens, max_output_length
    )

    decode_state = engine.init_decode_state()
    prefill_result = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )
    out_tokens = prefill_result.token
    self.assertEquals(out_tokens, decode_state_original.out_tokens[0][0])
    print(f"-------------------->: prefill step: --->")
    print(
        f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}"
    )
    print(f"-------------------->out_tokens: {out_tokens}")

    # self._diff_value(prefill_result.logits[0:10, :], torch.squeeze(decode_state_original.logits), "prefill logits")
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    for i in range(0, max_output_length - 1):
      decode_state_original = llama_original.decode(decode_state_original)
      decode_state, result_tokens = engine.generate(params, decode_state)

      # self._diff_value(decode_state.logits, decode_state_original.logits, "prefill logits")

      self.assertEquals(
          np.asanyarray(decode_state.tokens),
          np.array(decode_state_original.out_tokens),
      )
      print(f"-------------------->: decode step {i}: --->")
      print(
          f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}"
      )
      print(f"-------------------->out_tokens: {decode_state.tokens}")

  def test_llama_with_original_prefill_decode(self):
    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")

    torch.set_default_dtype(torch.float32)
    env = self._make_env()
    model_arg = env._model_arg
    tokens = np.arange(10, dtype=np.int32)
    true_length = tokens.shape[-1]
    padded_tokens = np.pad(tokens, (0, 6))
    padded_tokens = jnp.array(padded_tokens)

    seed = 1
    torch.manual_seed(1)
    max_output_length = 5

    file_dir = os.path.dirname(__file__)
    tokenizer_path = os.path.join(
        file_dir, "../jetstream_pt/third_party/llama2/tokenizer.model"
    )

    # orginal
    llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
    model_orig = llama_original.model
    state_dict = dict(model_orig.state_dict())
    state_dict["freqs_cis"] = model_orig.freqs_cis
    model_ours = model_exportable.Transformer(model_arg, env)
    engine = PyTorchEngine(pt_model=model_ours, env=env)
    params = self._to_jax(state_dict)
    slot = 0
    expected_output_tokens = []
    out_tokens = []

    prompt_tokens = [tokens]
    decode_state_original = llama_original.prefill(
        prompt_tokens, max_output_length
    )

    decode_state = engine.init_decode_state()
    prefill_result = engine.prefill(
        params=params, padded_tokens=padded_tokens, true_length=true_length
    )
    out_tokens = prefill_result.token
    print(f"-------------------->: prefill step: --->")
    print(
        f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}"
    )
    print(f"-------------------->out_tokens: {out_tokens}")

    # self._diff_value(prefill_result.logits[0:10, :], torch.squeeze(decode_state_original.logits), "prefill logits")
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)

    for i in range(0, max_output_length - 1):
      decode_state_original = llama_original.decode(decode_state_original)
      decode_state, result_tokens = engine.generate(params, decode_state)

      # self._diff_value(decode_state.logits, decode_state_original.logits, "prefill logits")
      print(f"-------------------->: decode step {i}: --->")
      print(
          f"-------------------->orginal out_tokens: {decode_state_original.out_tokens}"
      )
      print(f"-------------------->out_tokens: {decode_state.tokens}")


if __name__ == "__main__":
  unittest.main()
