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

import functools
import os
import time

import humanize
import jax
import numpy as np
# pylint: disable-next=all
from absl import app, flags
from jetstream_pt import engine as je
from jetstream_pt.config import (
    FLAGS,
    create_engine_from_config_flags,
    define_common_flags,
    define_profiling_flags,
)

define_common_flags()
define_profiling_flags()

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
#     "bf16_enable", False, "Whether to enable bf16", required=False
# )
# _CONTEXT_LENGTH = flags.DEFINE_integer(
#     "context_length", 1024, "The context length", required=False
# )
# _BATCH_SIZE = flags.DEFINE_integer(
#     "batch_size", 32, "The batch size", required=False
# )
# _PROFILING_OUTPUT = flags.DEFINE_string(
#     "profiling_output",
#     "",
#     "The profiling output",
#     required=False,
# )

# _SIZE = flags.DEFINE_string("size", "tiny", "size of model")

# _QUANTIZE_WEIGHTS = flags.DEFINE_bool(
#     "quantize_weights", False, "weight quantization"
# )
# _QUANTIZE_KV_CACHE = flags.DEFINE_bool(
#     "quantize_kv_cache", False, "kv_cache_quantize"
# )
# _MAX_CACHE_LENGTH = flags.DEFINE_integer(
#     "max_cache_length", 1024, "kv_cache_quantize"
# )


def create_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()
  engine = je.create_pytorch_engine(
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


def delete_pytree(p):
  """delete jax pytree"""

  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_map(delete_leaf, p)


def print_mem_usage():
  """Print current mem usage"""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(
        f"memory using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}"
    )


def create_prefill_tokens():
  """create list of prefill tokens"""
  prefill_lengths = [
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096,
      8192,
      16384,
      32768,
      # 65536,
      # 131072,
  ]
  tokens_list = []
  for length in prefill_lengths:
    tokens = np.random.randint(1, 32000, length)
    tokens_list.append(tokens)
  return tokens_list


def prefill_benchmark(tokens_list, engine, params, warmup):
  """prefill bechmark function"""
  for prefill_tokens in tokens_list:
    # pylint: disable-next=all
    warmup_text = "warmup" if warmup else "execute"
    it = time.time()
    prefill_result = engine.prefill(
        params=params,
        padded_tokens=prefill_tokens,
        true_length=len(prefill_tokens),
    )
    print(f"---- {warmup_text} First Token: {prefill_result.token}")
    elapsed = time.time() - it
    print(
        f"---- {warmup_text} time: {elapsed} for token_len: {len(prefill_tokens)}"
    )
    if warmup:
      print_mem_usage()
    delete_pytree(prefill_result)
    print("\n\n")


# pylint: disable-next=all
def main(argv):

  engine = create_engine_from_config_flags()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  profiling_output = FLAGS.profiling_output
  if profiling_output:
    jax.profiler.start_trace(profiling_output)

  print_mem_usage()
  tokens_list = create_prefill_tokens()
  for _ in range(3):
    prefill_benchmark(
        tokens_list=tokens_list, engine=engine, params=params, warmup=True
    )
    prefill_benchmark(
        tokens_list=tokens_list, engine=engine, params=params, warmup=True
    )

  for _ in range(5):
    prefill_benchmark(
        tokens_list=tokens_list, engine=engine, params=params, warmup=False
    )
    prefill_benchmark(
        tokens_list=tokens_list, engine=engine, params=params, warmup=False
    )

  if profiling_output:
    jax.profiler.stop_trace()


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
