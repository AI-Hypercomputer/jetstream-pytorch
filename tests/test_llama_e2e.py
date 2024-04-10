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
        return pytree.tree_map_only(
            torch.Tensor,
            torch_xla2.tensor.t2j, tree)    

    def _make_env(self):
        jax.config.update('jax_dynamic_shapes', False)
        jax.config.update('jax_traceback_filtering', 'off')
        env_data = environment.JetEngineEnvironmentData()
        env_data.max_input_sequence_length = 128
        env_data.max_input_sequence_length = 128
        env_data.cache_sequence_length = 128
        env_data.model_type = 'llama-2-tiny'
        env_data.batch_size = 1
        env = environment.JetEngineEnvironment(env_data)
        env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
        return env


    def test_original_llama2_seed(self):
        jax.config.update('jax_platform_name', 'cpu')
        x = jnp.square(2)
        print(f"---------> {jax.devices()}")
        torch.set_default_dtype(torch.bfloat16)
        env = self._make_env()
        model_arg = env._model_arg 
        tokens = np.arange(10, dtype=np.int32)
        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')
        output_tokens_multiple = []
        for i in [1, 999, 99999]:
            llama_original = LlamaOriginal.build(tokenizer_path, model_arg, i)
            prompt_tokens = [tokens]
            output_tokens = llama_original.generate(prompt_tokens, 10)
            output_tokens_multiple.append(output_tokens)

        for index, output_tokens in enumerate(output_tokens_multiple): 
            print(f"------------------- index: {index}, tokens:{output_tokens}")
            if index > 0:
                self.assertNotEqual(output_tokens_multiple[index], output_tokens_multiple[index - 1])

    def test_jetstream_llama2_seed(self):
        jax.config.update('jax_platform_name', 'cpu')
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
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')


        seed = 1
        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis
        params = self._to_jax(state_dict)

        output_tokens_multiple = []
        for i in [1, 2, 3]:
            torch.manual_seed(1)
            model_ours = model_exportable.Transformer(model_arg, env)
            engine = PyTorchEngine(
                pt_model=model_ours,
                env=env
            )
            
            decode_state = engine.init_decode_state()
            slot = 0 
            prefill_result = engine.prefill(
                params=params, padded_tokens=padded_tokens, true_length=true_length
            )

            decode_state = engine.insert(
                prefill_result, decode_state, slot=slot
            )

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
                self.assertEqual(output_tokens_multiple[index], output_tokens_multiple[index - 1])

    def test_llama_e2e(self):
        jax.config.update('jax_platform_name', 'cpu')
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
        torch.manual_seed(1)
        max_output_length = 10

        file_dir = os.path.dirname(__file__)
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )

        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )

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
        print(f"-------------------->expected_output_tokens:{expected_output_tokens}")
        # self.assertEqual(out_tokens ,expected_output_tokens)




    def test_llama_e2e_two_addtional_tokens(self):
        jax.config.update('jax_platform_name', 'cpu')
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
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )

        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )

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
        print(f"-------------------->expected_output_tokens:{expected_output_tokens}")
        # self.assertEqual(out_tokens ,expected_output_tokens)


    def test_llama_e2e_four_addtional_tokens(self):
        jax.config.update('jax_platform_name', 'cpu')
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
        tokenizer_path = os.path.join(file_dir, '../jetstream_pt/third_party/llama2/tokenizer.model')

        # orginal
        llama_original = LlamaOriginal.build(tokenizer_path, model_arg, seed)
        prompt_tokens = [tokens]
        expected_output_tokens = llama_original.generate(prompt_tokens, max_output_length)


        model_orig = llama_original.model

        state_dict = dict(model_orig.state_dict())
        state_dict['freqs_cis'] = model_orig.freqs_cis


        model_ours = model_exportable.Transformer(model_arg, env)
        
        engine = PyTorchEngine(
            pt_model=model_ours,
            env=env
        )

        params = self._to_jax(state_dict)
        decode_state = engine.init_decode_state()
        slot = 0 

        prefill_result = engine.prefill(
            params=params, padded_tokens=padded_tokens, true_length=true_length
        )

        decode_state = engine.insert(
            prefill_result, decode_state, slot=slot
        )

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
        print(f"-------------------->expected_output_tokens:{expected_output_tokens}")
        # self.assertEqual(out_tokens ,expected_output_tokens)        

if __name__ == '__main__':
    unittest.main()


