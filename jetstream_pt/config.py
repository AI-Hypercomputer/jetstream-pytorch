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


from absl import flags
from jetstream_pt.environment import QuantizationConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("tokenizer_path", None, "The tokenizer model path")
flags.DEFINE_string("model_name", None, "model type")
flags.DEFINE_string("checkpoint_path", None, "Directory for .pth checkpoints")
flags.DEFINE_bool("bf16_enable", True, "Whether to enable bf16")
flags.DEFINE_integer("context_length", 1024, "The context length")
flags.DEFINE_integer("batch_size", 32, "The batch size")
flags.DEFINE_string("size", "tiny", "size of model")
flags.DEFINE_integer("max_cache_length", 1024, "kv_cache_quantize")
flags.DEFINE_integer("max_decode_length", 1024, "max length of generated text")
flags.DEFINE_string("sharding_config", "", "config file for sharding")
flags.DEFINE_bool(
    "shard_on_batch",
    False,
    "whether to shard on batch dimension"
    "If set true, sharding_config will be ignored.",
)
flags.DEFINE_string("profiling_output", "", "The profiling output")

# Quantization related flags
flags.DEFINE_bool("quantize_weights", False, "weight quantization")
flags.DEFINE_bool(
    "quantize_activation",
    False,
    "Quantize Q,K,V projection and FeedForward activation. Defaults to False",
)
flags.DEFINE_string(
    "quantize_type", "int8_per_channel", "Type of quantization."
)
flags.DEFINE_bool(
    "quantize_kv_cache", None, "defaults to the same value as quantize_weights"
)
flags.DEFINE_bool(
    "internal_quantize_embedding_layer",
    True,
    "Whether to quantize embedding layer or not. Defaults to true",
)


_VALID_QUANTIZATION_TYPE = {
    "int8_per_channel",
    "int4_per_channel",
    "int8_blockwise",
    "int4_blockwise",
}

flags.register_validator(
    "quantize_type",
    lambda value: value in _VALID_QUANTIZATION_TYPE,
    f"quantize_type is invalid, supported quantization types are {_VALID_QUANTIZATION_TYPE}",
)
flags.DEFINE_bool(
    "profiling_prefill",
    False,
    "Whether to profile the prefill, "
    "if set to false, profile generate function only",
    required=False,
)
flags.DEFINE_bool(
    "ragged_mha",
    False,
    "Whether to enable Ragged multi head attention",
    required=False,
)
flags.DEFINE_integer(
    "starting_position",
    512,
    "The starting position of decoding, "
    "for performance tuning and debugging only",
    required=False,
)
flags.DEFINE_bool(
    "ring_buffer",
    True,
    "Whether to enable ring buffer",
    required=False,
)
flags.DEFINE_bool(
    "flash_attention",
    False,
    "Whether to enable flas attention. Only takes effect at test mode",
    required=False,
)
flags.DEFINE_bool(
    "generate_cache_stacked",
    False,
    "Whether to stack the generate cache to the layer dimension. Only takes effect at test mode",
    required=False,
)
flags.DEFINE_bool(
    "new_cache_stacked",
    False,
    "Whether to stack the generate cache to the layer dimension. Only takes effect at test mode",
    required=False,
)
flags.DEFINE_bool(
    "lazy_cache_update",
    False,
    "Whether to update the cache during attention or delayed until all the layers are done. "
    "Only takes effect at test mode",
    required=False,
)
flags.DEFINE_float(
    "temperature",
    1.0,
    "temperature parameter for scaling probability."
    "Only invoked when sampling algorithm is set to"
    "weighted or topk",
)
flags.DEFINE_string(
    "sampling_algorithm",
    "greedy",
    "sampling algorithm to use. Options:"
    "('greedy', 'weighted', 'neucleus', 'topk')",
)
flags.DEFINE_float(
    "nucleus_topp",
    0.0,
    "restricting to p probability mass before sampling",
)
flags.DEFINE_integer(
    "topk",
    0,
    "size of top k used when sampling next token",
)

flags.DEFINE_integer(
    "paged_attention_total_num_pages",
    0,
    "total number of pages per layer for page attention",
)

flags.DEFINE_integer(
    "paged_attention_page_size",
    64,
    "page size per page",
)
flags.DEFINE_string(
    "jax_compilation_cache_dir",
    "~/jax_cache",
    "Jax compilation cache directory",
)
flags.DEFINE_integer(
    "jax_persistent_cache_min_entry_size_bytes",
    0,
    "Minimum size (in bytes) of an entry that will be cached in the persistent compilation cache",
)
flags.DEFINE_integer(
    "jax_persistent_cache_min_compile_time_secs",
    1,
    "Minimum compilation time for a computation to be written to persistent cache",
)


def create_quantization_config_from_flags():
  """Create Quantization Config from cmd flags"""
  config = QuantizationConfig()
  quantize_weights = FLAGS.quantize_weights
  quantize_type = FLAGS.quantize_type
  if not quantize_weights:
    return config
  config.enable_weight_quantization = True

  config.num_bits_weight = 8 if "int8" in quantize_type else 4
  config.is_blockwise_weight = "blockwise" in quantize_type

  config.enable_activation_quantization = FLAGS.quantize_activation
  config.enable_embedding_quantization = FLAGS.internal_quantize_embedding_layer
  config.enable_kv_quantization = (
      FLAGS.quantize_kv_cache
      if FLAGS.quantize_kv_cache is not None
      else FLAGS.quantize_weights
  )
  return config
