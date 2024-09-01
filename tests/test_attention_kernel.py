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
import jax.sharding as jsharding
import torch
import torch_xla2
from torch.utils import _pytree as pytree
from absl.testing import parameterized

from jetstream_pt.engine import PyTorchEngine, DecodeState
import jetstream_pt.attention_kernel as ak
from jetstream_pt.third_party.llama import model_exportable, model_args
from jetstream_pt.third_party.llama.generation_original import LlamaOriginal
from jetstream_pt import environment
from tests import helpers
from jetstream_pt import torchjax


P = jax.sharding.PartitionSpec

class AttentionKernelTest(parameterized.TestCase):
  """This test class includes all E2E test for llama2"""

  # pylint: disable-next=all
  def _make_page_attention_env(self, bf16_enable=True, env_data_update_fn=lambda _: None):
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    jax.config.update("jax_dynamic_shapes", False)
    jax.config.update("jax_traceback_filtering", "off")
    config = model_args.get_model_args("llama-2-tiny", 1024, 1, True)
    config.n_kv_heads = 8
    config.n_heads = 8
    environment_data = environment.JetEngineEnvironmentData()
    environment_data.page_size = 64
    environment_data.total_num_pages = 256
    environment_data.block_size = 512
    environment_data.max_input_sequence_length = 1024
    environment_data.max_input_sequence_length = 1024
    environment_data.cache_sequence_length = 2048
    environment_data.bf16_enable = bf16_enable
    environment_data.model_type = "llama-2-tiny"
    environment_data.batch_size = 1
    environment_data.num_layers = config.n_layers
    environment_data.cache_shape = (
        config.n_kv_heads,
        environment_data.total_num_pages,
        environment_data.page_size,
        config.dim // config.n_heads,    
    )
    environment_data.testing = True
    env_data_update_fn(environment_data)
    env = environment.JetEngineEnvironment(environment_data)
    env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
    return env, config



  def _from_torch(self, tree):
    return pytree.tree_map_only(torch.Tensor, torch_xla2.tensor.t2j, tree)


  def test_prefill_step0(self):
    "end to end jetstream llama test with bfloat16"
    #jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGHEST)
    print(f"---------> {jax.devices()}")
    
    env, config = self._make_page_attention_env()
    q_sharding = jsharding.NamedSharding(env.mesh, P(None, "x", None))
    kv_sharding = jsharding.NamedSharding(env.mesh, P("x", None, None, None))
    
    
 
    n_heads = 8
    total_num_pages = env._data.total_num_pages
    page_size = env._data.total_num_pages
    
    xq = self.get_xq(1, n_heads, 0)
    xq = jax.device_put(xq, q_sharding)
    keys = self.get_keys(n_heads, total_num_pages, page_size, 0)
    keys = jax.device_put(keys, kv_sharding)
    values = self.get_values(n_heads, total_num_pages, page_size, 0)
    values = jax.device_put(values, kv_sharding)
    seq_lens = self.get_seq_lens(0)
    page_indices = self.get_page_indices()
    
    xq, keys, values, seq_lens, page_indices = torchjax.to_torch((xq, keys, values, seq_lens, page_indices))
    
    out = ak.call_paged_attention(env, xq, keys, values, seq_lens, page_indices)
    print(out)
    
    # page_env, page_model_arg = self._make_page_attention_env(bf16_enable=True)
    
    # decode = self._insert(engine, params)
    # page_decode = self._insert(page_engine, page_params)
    
    # cache = self.get_compress_kv_cache(decode, 0)
    # page_caches = page_engine.page_attention_manager.get_compress_kv_cache(page_decode.caches, 0)
    
    # self.assertEqual(decode.tokens, page_decode.tokens)
    # for (k, v), (pk, pv) in zip(cache, page_caches):
    #   self.assertTrue(jnp.array_equal(k, pk))
    #   self.assertTrue(jnp.array_equal(v, pv))

  def get_xq(self, batch, head, step):
    # xq[0:1,0:1,0:2]
    xq1= [[[0.976562, 0.335938]]]
    xq2 = [[[0.494141, 1.66406]]]
    xq = xq1 if step == 0 else xq2
    xq = jnp.asarray(xq, dtype=jnp.float32)
    xq = jnp.tile(xq, (batch, head, 1))
    return xq
    
  def get_keys(self, head, pages, page_size, step):
    # keys[0, 0, 0:9, 0:2]
    key1 = [[[[-0.419922, -0.194336],
          [-0.0270996, -0.211914],
          [-0.234375, -0.178711],
          [0.279297, 0.699219],
          [0.0114746, -0.138672],
          [-0.886719, -0.296875],
          [0.106934, -0.269531],
          [0.133789, -0.363281],
          [0.953125, 0.165039],]]]  
          
    # keys[0, 0, 0:10, 0:2]
    key2 = [[[-0.419922, -0.194336],
        [-0.0270996, -0.211914],
        [-0.234375, -0.178711],
        [0.279297, 0.699219],
        [0.0114746, -0.138672],
        [-0.886719, -0.296875],
        [0.106934, -0.269531],
        [0.133789, -0.363281],
        [0.953125, 0.165039],
        [-0.0247803, 0.197266]]]
    
    key = key1 if step == 0 else key2
    key = jnp.asarray(key)  
    
    r = jnp.zeros((head, pages, page_size, 2),  dtype=jnp.float32)
    r = r.at[:, 0:1, 0:key.shape[2], :].set(key)
    return r
    
  def get_values(self, head, pages, page_size, step):
    # values[0:1, 0:1, :9, 0:2]
    v1 = [[[[-0.000770569, -0.019043],
          [-0.00793457, -0.00564575],
          [0.00234985, -0.0113525],
          [0.00311279, -0.00210571],
          [0.012085, 0.00242615],
          [-0.00665283, -0.00382996],
          [-0.000762939, -0.00738525],
          [-0.00811768, 0.00646973],
          [-0.00352478, -0.00128174]]]]   
    
    # values[0:1, 0:1, :10, 0:2]
    v2 = [[-0.000770569, -0.019043],
        [-0.00793457, -0.00564575],
        [0.00234985, -0.0113525],
        [0.00311279, -0.00210571],
        [0.012085, 0.00242615],
        [-0.00665283, -0.00382996],
        [-0.000762939, -0.00738525],
        [-0.00811768, 0.00646973],
        [-0.00352478, -0.00128174],
        [0.0014801, -0.00915527]] 
    v = v1 if step == 0 else v2
    v = jnp.asarray(v)  
    
    r = jnp.zeros((head, pages, page_size, 2), dtype=jnp.float32)
    r = r.at[:, 0:1, 0:v.shape[2], :].set(v)
    return r     
    # in_specs = (
    #     P(None, kv_head_mesh_axis_name, None),  # q
    #     P(kv_head_mesh_axis_name, None, None, None),  # k
    #     P(kv_head_mesh_axis_name, None, None, None),  # v
    #     P(),  # lengths
    #     P(),  # page_indices
    # )
  
  
  def get_seq_lens(self, step):
    lens = [9] if step == 0 else [10]
    return jnp.asarray(lens, dtype=jnp.int32)

  def get_page_indices(self):
    #(1, 32)
    indices = [ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]  
    return jnp.asarray(indices, dtype=jnp.int32).reshape(-1, 32)

  def get_output(self):
    # (1, 1, 2)
    output = [[[-0.00352478, -0.001297]]]
    return jnp.asarray(output)
  
if __name__ == "__main__":
  unittest.main()
