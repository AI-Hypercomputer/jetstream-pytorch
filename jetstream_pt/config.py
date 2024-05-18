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

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "tokenizer_path",
    None,
    "The tokenizer model path",
    required=True,
)
flags.DEFINE_string("model_name", None, "model type", required=False)
flags.DEFINE_string(
    "checkpoint_path", None, "Directory for .pth checkpoints", required=False
)
flags.DEFINE_bool(
    "bf16_enable", True, "Whether to enable bf16", required=False
)
flags.DEFINE_integer(
    "context_length", 1024, "The context length", required=False
)
flags.DEFINE_integer("batch_size", 32, "The batch size", required=False)
flags.DEFINE_string("size", "tiny", "size of model")
flags.DEFINE_bool("quantize_weights", False, "weight quantization")
flags.DEFINE_bool("quantize_kv_cache", False, "kv_cache_quantize")
flags.DEFINE_integer("max_cache_length", 1024, "kv_cache_quantize")
flags.DEFINE_string("sharding_config", "", "config file for sharding")
flags.DEFINE_bool(
    "shard_on_batch",
    False,
    "whether to shard on batch dimension"
    "If set true, sharding_config will be ignored.",
)
flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)


def define_profiling_flags():
  """Add profiling related config flags to global FLAG."""
  


def create_engine_from_config_flags():
  """create a pytorch engine from config flag"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()
  engine = create_pytorch_engine(
      model_name=FLAGS.model_name,
      devices=devices,
      tokenizer_path=FLAGS.tokenizer_path,
      ckpt_path=FLAGS.checkpoint_path,
      bf16_enable=FLAGS.bf16_enable,
      param_size=FLAGS.size,
      context_length=FLAGS.context_length,
      batch_size=FLAGS.batch_size,
      quantize_weights=FLAGS.quantize_weights,
      quantize_kv=FLAGS.quantize_kv_cache,
      max_cache_length=FLAGS.max_cache_length,
      sharding_config=FLAGS.sharding_config,
      shard_on_batch=FLAGS.shard_on_batch,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine
