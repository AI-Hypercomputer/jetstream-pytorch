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

import os
import time

import jax
from absl import flags
from jetstream_pt.engine import create_pytorch_engine
from jetstream_pt.environment import QuantizationConfig

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tokenizer_path",
    None,
    "The tokenizer model path",
    required=False,
)
flags.DEFINE_string("model_name", None, "model type", required=False)
flags.DEFINE_string(
    "checkpoint_path", None, "Directory for .pth checkpoints", required=False
)
flags.DEFINE_bool("bf16_enable", True, "Whether to enable bf16", required=False)
flags.DEFINE_integer(
    "context_length", 1024, "The context length", required=False
)
flags.DEFINE_integer("batch_size", 32, "The batch size", required=False)
flags.DEFINE_string("size", "tiny", "size of model")
# flags.DEFINE_bool("quantize_weights", False, "weight quantization")
# flags.DEFINE_string(
#     "quantize_type", "int8_per_channel", "Type of quantization."
# )
flags.DEFINE_bool("quantize_kv_cache", False, "kv_cache_quantize")
flags.DEFINE_integer("max_cache_length", 1024, "kv_cache_quantize")
flags.DEFINE_string("sharding_config", "", "config file for sharding")
flags.DEFINE_bool(
    "shard_on_batch",
    False,
    "whether to shard on batch dimension"
    "If set true, sharding_config will be ignored.",
    required=False,
)
flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)


_VALID_QUANTIZATION_TYPE = {
    "int8_per_channel",
    "int4_per_channel",
    "int8_blockwise",
    "int4_blockwise",
}

flags.DEFINE_string("quantize_type", "", "Type of quantization.")
flags.register_validator(
    "quantize_type",
    lambda value: value in _VALID_QUANTIZATION_TYPE,
    f"quantize_type is invalid, supported quantization types are {_VALID_QUANTIZATION_TYPE}",
)


def create_quantization_config_from_flags():
  """Create Quantization Config from cmd flags"""
  config = QuantizationConfig()
  quantize_type = FLAGS.quantize_type
  if quantize_type == "":
    return config
  config.enable_weight_quantization = True
  config.num_bits_weight = 8 if "int8" in quantize_type else 4
  config.is_blockwise_weight = "blockwise" in quantize_type
  config.enable_kv_quantization = FLAGS.quantize_kv_cache
  return config


def create_engine_from_config_flags():
  """create a pytorch engine from cmd flag"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()

  # Create quant config.
  quant_config = create_quantization_config_from_flags()
  # Derive sharding_config_path if it's not given by user.
  sharding_config_path = FLAGS.sharding_config
  if not sharding_config_path:
    sharding_config_name = FLAGS.model_name
    if (
        quant_config.enable_weight_quantization
        and quant_config.is_blockwise_weight
    ):
      sharding_config_name += "-blockwise-quant"
    sharding_config_path = os.path.join(
        "default_shardings", sharding_config_name + ".yaml"
    )

  engine = create_pytorch_engine(
      model_name=FLAGS.model_name,
      devices=devices,
      tokenizer_path=FLAGS.tokenizer_path,
      ckpt_path=FLAGS.checkpoint_path,
      bf16_enable=FLAGS.bf16_enable,
      param_size=FLAGS.size,
      context_length=FLAGS.context_length,
      batch_size=FLAGS.batch_size,
      quant_config=quant_config,
      max_cache_length=FLAGS.max_cache_length,
      sharding_config=sharding_config_path,
      shard_on_batch=FLAGS.shard_on_batch,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine
