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

import dataclasses
from typing import Tuple

import jax
import jax.sharding as jsharding
from jax.experimental import mesh_utils
import torch_xla2
import yaml


from jetstream_pt import cache_manager


@dataclasses.dataclass
# pylint: disable-next=all
class QuantizationConfig:
  enable_weight_quantization: bool = False
  num_bits_weight: int = 8
  is_blockwise_weight: bool = False
  block_size_weight: int = 128
  is_symmetric_weight: bool = True

  enable_activation_quantization: bool = False
  enable_kv_quantization: bool = False


@dataclasses.dataclass
# pylint: disable-next=all
class JetEngineEnvironmentData:
  checkpoint_path: str = ""  # if empty string then use model's state_dict()
  checkpoint_format: str = "safetensors"  # torch, safetensors

  tokenizer_path: str = ""

  max_input_sequence_length: int = 1024
  max_decode_length: int = 1024
  batch_size: int = 32  # batch size is generate step batch size
  cache_sequence_length: int = 2048  # size of the cache.

  quant_config: QuantizationConfig = QuantizationConfig()

  model_type: str = "llama-2-13b"  # this implies the model config

  # Names of the axis of the tensors for QKV in Attention.
  # This is also the dimensions of KV cache
  attention_kv_axis_names: Tuple[str, ...] = (
      "batch",
      "num_attn_heads",
      "sequence_length",
      "head_dim",
  )

  # Shape of cache len(cache_shape) == len(attention_kv_axis_names)
  cache_shape: Tuple[int, ...] = ()

  num_layers: int = 0

  # This is the axis to shard among the number of available devices
  # This string must be one of the values of attention_kv_axis_names above
  kv_cache_shard_axis: str = "num_attn_heads"

  # QKV fusion has negative performance on TPU, slicing takes longer
  qkv_fusion: bool = False

  # If Ture, use bfloat16 as dtype. If False, use float32 as dtype
  bf16_enable: bool = True

  sharding_config_path: str = ""

  # Whether to shard on batch dimension. i.e. data parallel.
  shard_on_batch: bool = False

  # Whether to enable ragged multi head attention.
  ragged_mha: bool = False

  # The block size for the ragged attention.
  block_size: int = 512

  # Starting position
  starting_position: int = 512

  # Ring buffer
  ring_buffer: bool = False

  flash_attention: bool = True

  generate_cache_stacked: bool = True
  # Variables used in token sampling
  # sampling algorithm to use ("greedy", "weighted", "neucleus", "topk")
  sampling_algorithm: str = "greedy"

  # size of top k used when sampling next token
  topk: int = 0

  # restricting to p probability mass before sampling
  nucleus_topp: float = 0.0

  # temperature parameter for scaling probability
  temperature: float = 1.0


# pylint: disable-next=all
class JetEngineEnvironment:

  def __init__(self, data: JetEngineEnvironmentData):
    self._data = data

    self.batch_size = self._data.batch_size
    self.seq_len = self._data.max_input_sequence_length
    self.cache_len = self._data.cache_sequence_length
    self.ragged_mha = self._data.ragged_mha
    self.block_size = self._data.block_size
    self.starting_position = self._data.starting_position
    self.flash_attention = self._data.flash_attention
    self.generate_cache_stacked = self._data.generate_cache_stacked
    self.num_layers = self._data.num_layers
    self.ring_buffer = self._data.ring_buffer

    if self.generate_cache_stacked:
      self.cache_shape = (self.num_layers, *self._data.cache_shape)
    else:
      self.cache_shape = self._data.cache_shape

    P = jax.sharding.PartitionSpec

    num_of_partitions = jax.device_count()
    # make mesh etc.
    self.mesh = jsharding.Mesh(
        mesh_utils.create_device_mesh((num_of_partitions, 1)),
        axis_names=("x", "y"),
    )

    self.y_sharding = jsharding.NamedSharding(self.mesh, P(None, "x"))
    self.x_sharding = jsharding.NamedSharding(self.mesh, P("x"))
    self.replicated = jsharding.NamedSharding(self.mesh, P())

    if self.generate_cache_stacked:
      
      self.attention_kv_axis_names = (
      "layer",
      "batch",
      "num_attn_heads",
      "sequence_length",
      "head_dim",
    )
    if data.shard_on_batch:
      self.kv_cache_shard_axis = "batch"
    else:
      self.kv_cache_shard_axis = "num_attn_heads"

    cache_sharding_axis = self.attention_kv_axis_names.index(
        self.kv_cache_shard_axis
    )

    if self.cache_shape[cache_sharding_axis] == 1:
      # cannot shard on an axis that is 1
      # default to last
      cache_sharding_axis = len(self.cache_shape) - 1

    self.cache_sharding = self.sharding_by_axis(cache_sharding_axis)
    self._load_sharding_config()

  def _load_sharding_config(self):
    """Load sharding config"""
    if self._data.sharding_config_path:
      with open(self._data.sharding_config_path, encoding="utf-8") as f:
        self._sharding_config = yaml.safe_load(f)
    else:
      self._sharding_config = {}

  def __getattr__(self, name):
    return getattr(self._data, name)

  # This is used by model to add activation sharding.
  def apply_sharding(self, tensor, *, axis: int | None):
    """Apply sharding for tensor"""
    if not isinstance(tensor, torch_xla2.tensor.XLATensor2):
      return
    sharding_spec = self.sharding_by_axis(axis)
    # pylint: disable-next=all
    tensor._elem = jax.lax.with_sharding_constraint(tensor._elem, sharding_spec)

  def partition_by_axis(self, axis=None):
    """return sharding partition spc by axis, options are x, y, -1 or Noe"""
    if axis == -1 or axis is None:
      return jax.sharding.PartitionSpec()
    sharding = [None] * (axis + 1)
    sharding[axis] = "x"
    sharding_spec = jax.sharding.PartitionSpec(*sharding)
    return sharding_spec

  def sharding_by_axis(self, axis):
    """return sharding partition spc by axis, options are x, y, -1 or Noe"""
    return jsharding.NamedSharding(self.mesh, self.partition_by_axis(axis))

  def make_caches_prefill(self):
    """Create kv caches for inference prefill"""
    caches = []
    for _ in range(self.num_layers):
      caches.append(cache_manager.KVCachePrefill())
    return caches

  def make_caches_generate(self):
    """Create kv caches for inference generation"""
    caches = []

    layered_cache_count = 1 if self.generate_cache_stacked else self.num_layers

    for _ in range(layered_cache_count):
      if self._data.quant_config.enable_kv_quantization:
        caches.append(
            cache_manager.Int8KVCacheGenerate.empty(
                self.cache_shape, self.cache_sharding, env=self
            )
        )
      else:
        caches.append(
            cache_manager.KVCacheGenerate.empty(
                self.cache_shape, self.cache_sharding, env=self
            )
        )
    return caches

  def sharding_by_name(self, name):
    """Create sharding specified in the config."""
    if self.shard_on_batch:
      return self.sharding_by_axis(0)  # batch dimension

    if name in self._sharding_config:
      return self.sharding_by_axis(self._sharding_config[name])

    name = process_sharding_name(name)
    if name in self._sharding_config:
      return self.sharding_by_axis(self._sharding_config[name])

    raise RuntimeError("Sharding for name: ", name, " not specified")


def process_sharding_name(name):
  """Replace integers in param name with *.

  Presumably all layers should have the same sharding.
  """

  def is_integer(t):
    try:
      int(t)
      return True
    # pylint: disable-next=all
    except:  # noqa: E722
      return False

  tokens = name.split(".")
  for i, t in enumerate(tokens):
    if is_integer(t):
      tokens[i] = "*"
  return ".".join(tokens)
