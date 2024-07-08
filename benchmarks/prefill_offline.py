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

# import torch_xla2 first!
# pylint: disable-next=all
import torch_xla2
import humanize
import jax
import numpy as np
# pylint: disable-next=all
from absl import app, flags
from jetstream_pt.config import FLAGS, create_engine_from_config_flags


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
    prefill_result, _ = engine.prefill(
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
