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

import jax
import jax.sharding as jsharding
from jax.experimental import mesh_utils

import dataclasses
from typing import Tuple, Dict

from jetstream_pt.third_party.llama2 import model_args
from jetstream_pt import cache_manager
import torch_xla2



@dataclasses.dataclass
class JetEngineEnvironmentData:
    checkpoint_path: str = '' # if empty string then use model's state_dict()
    checkpoint_format: str = 'safetensors' # torch, safetensors
    
    tokenizer_path: str = ''

    max_input_sequence_length: int = 1024
    max_decode_length: int = 1024
    batch_size: int = 32 # batch size is generate step batch size
    cache_sequence_length: int = 2048  # size of the cache.

    enable_weight_quantization: bool = False
    enable_kv_quantization: bool = False

    model_type: str = 'llama-2-13b'  # this implies the model config 

    # Names of the axis of the tensors for QKV in Attention.
    # This is also the dimensions of KV cache
    attention_kv_axis_names: Tuple[str, ...] = ('batch', 'num_attn_heads', 'sequence_length', 'head_dim')

    # This is the axis to shard among the number of available devices
    # This string must be one of the values of attention_kv_axis_names above
    kv_cache_shard_axis: str = 'num_attn_heads'

    # Override sharding axis of a weight by name
    experimental_sharding_axis_override: Dict[str, int] = dataclasses.field(default_factory=dict)

    # QKV fusion has negative performance on TPU, slicing takes longer
    qkv_fusion: bool = False

    # If Ture, use bfloat16 as dtype. If False, use float32 as dtype 
    bf16_enable: bool = True

class JetEngineEnvironment:

    def __init__(self, data: JetEngineEnvironmentData):
        self._data = data
        # Get 13b
        self._model_arg = model_args.get_model_args(
            data.model_type.replace('llama-2-', ''),
            context_length=data.max_input_sequence_length,
            batch_size=data.batch_size,
            vocab_size=32000,  # ?
            bf16_enable=data.bf16_enable,
            )

        self.batch_size = self._data.batch_size
        self.seq_len = self._data.max_input_sequence_length
        self.num_layers = self._model_arg.n_layers
        self.num_kv_heads = self._model_arg.n_kv_heads
        self.num_heads = self._model_arg.n_heads
        self.head_dim = self._model_arg.dim // self._model_arg.n_heads
        self.cache_sequence_length = self._data.cache_sequence_length
        self.bf16_enable = self._data.bf16_enable

        Mesh = jax.sharding.Mesh
        P = jax.sharding.PartitionSpec

        num_of_partitions = jax.device_count()  # TODO
        # make mesh etc.
        self._mesh = jsharding.Mesh(
            mesh_utils.create_device_mesh((num_of_partitions, 1)),
            axis_names=("x", "y"),
        )

        self.y_sharding = jsharding.NamedSharding(self._mesh, P(None, "x"))
        self.x_sharding = jsharding.NamedSharding(self._mesh, P("x"))
        self.replicated = jsharding.NamedSharding(self._mesh, P())

        cache_sharding = ("x" if axis == self._data.kv_cache_shard_axis else None
                          for axis in self._data.attention_kv_axis_names)
        self.cache_sharding = jsharding.NamedSharding(self._mesh, P(*cache_sharding))

    def __getattr__(self, name):
        return getattr(self._data, name)

    @property
    def tokenizer_path(self):
        return self._data.tokenizer_path

    # This is used by model to add activation sharding.
    def apply_sharding(self, tensor, *, axis: int | None):
        if not isinstance(tensor, torch_xla2.tensor.XLATensor2):
            return
        sharding_spec = self.sharding_by_axis(axis)
        tensor._elem = jax.lax.with_sharding_constraint(tensor._elem, sharding_spec)

    def sharding_by_axis(self, axis):
        if axis == -1 or axis is None:
            return jsharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec())
        sharding = [None] * (axis + 1)
        sharding[axis] = "x"
        sharding_spec = jsharding.NamedSharding(self._mesh, jax.sharding.PartitionSpec(*sharding))
        return sharding_spec
        
    def make_caches_prefill(self):
        caches = []
        for _ in range(self.num_layers):
            caches.append(cache_manager.KVCachePrefill())
        return caches

    def make_caches_generate(self):
        caches = []
        shape = (self.batch_size, self.num_kv_heads, self._data.cache_sequence_length, self.head_dim)
        for _ in range(self.num_layers):
            if self.enable_kv_quantization:
                caches.append(cache_manager.Int8KVCacheGenerate.empty(shape, self.cache_sharding, self.bf16_enable))
            else:
                caches.append(cache_manager.KVCacheGenerate.empty(shape, self.cache_sharding, self.bf16_enable))
        return caches




