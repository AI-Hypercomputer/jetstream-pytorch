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

"""Runs a pytorch server."""
import os
from typing import Sequence

import jetstream_pt
from absl import app, flags
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig
from jetstream_pt.config import (
    FLAGS,
    create_engine_from_config_flags,
    define_common_flags,
)

define_common_flags()

flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)
flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)
flags.DEFINE_string(
    "platform",
    "tpu=4",
    "The platform that the engine runs on",
    required=False,
)
# _PORT = flags.DEFINE_integer("port", 9000, "port to listen on")
# _THREADS = flags.DEFINE_integer(
#     "threads", 64, "number of worker threads in thread pool"
# )
# _CONFIG = flags.DEFINE_string(
#     "config",
#     "InterleavedCPUTestServer",
#     "available servers",
# )
# _PROFILING_OUTPUT = flags.DEFINE_string(
#     "profiling_output",
#     "",
#     "The profiling output",
#     required=False,
# )
# _PLATFORM = flags.DEFINE_string(
#     "platform",
#     "tpu=4",
#     "The platform that the engine runs on",
#     required=False,
# )

# Common flags
# _TOKENIZER_PATH = flags.DEFINE_string(
#     "tokenizer_path",
#     "tokenizer.model",
#     "The tokenizer model path",
#     required=False,
# )
# _CKPT_PATH = flags.DEFINE_string(
#     "checkpoint_path", None, "Directory for .pth checkpoints", required=False
# )
# _BF16_ENABLE = flags.DEFINE_bool(
#     "bf16_enable", True, "Whether to enable bf16", required=False
# )
# _CONTEXT_LENGTH = flags.DEFINE_integer(
#     "context_length", 1024, "The context length", required=False
# )
# _BATCH_SIZE = flags.DEFINE_integer(
#     "batch_size", 32, "The batch size", required=False
# )

# _PARAM_SIZE = flags.DEFINE_string(
#     "param_size",
#     "7b",
#     "The model size the server runs on.",
#     required=False,
# )
# _MODEL_NAME = flags.DEFINE_string(
#     "model",
#     "llama-2",
#     "name of the model. Supported options are llama-2 and llama-3",
# )

# _QUANTIZE_WEIGHTS = flags.DEFINE_bool(
#     "quantize_weights", False, "weight quantization"
# )
# _QUANTIZE_KV_CACHE = flags.DEFINE_bool(
#     "quantize_kv_cache", False, "kv_cache_quantize"
# )
# _MAX_CACHE_LENGTH = flags.DEFINE_integer(
#     "max_cache_length", 1024, "kv_cache_quantize"
# )
# _SHARDING_CONFIG = flags.DEFINE_string(
#     "sharding_config", "", "config file for sharding"
# )
# _SHARD_ON_BATCH = flags.DEFINE_bool(
#     "shard_on_batch",
#     False,
#     "whether to shard on batch dimension"
#     "If set true, sharding_config will be ignored.",
# )


# pylint: disable-next=all
def main(argv: Sequence[str]):
  del argv
  os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_logs --xla_dump_hlo_as_text"
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  sharding_config_path = _SHARDING_CONFIG.value
  engine = jetstream_pt.create_pytorch_engine(
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
  server_config = ServerConfig(
      interleaved_slices=(FLAGS.platform,),
      interleaved_engine_create_fns=(lambda a: engine,),
  )
  print(f"server_config: {server_config}")

  # We separate credential from run so that we can unit test it with local credentials.
  # We would like to add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=FLAGS.threads,
      port=FLAGS.port,
      config=server_config,
      devices=devices,
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)
