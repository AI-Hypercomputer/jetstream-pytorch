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

import logging
import os
import time

# import torch_xla2 first!
# pylint: disable-next=all
import torch_xla2
import jax
import jax.numpy as jnp
# pylint: disable-next=all
from absl import app, flags
# pylint: disable-next=all
from benchmarks import analyze_sharegpt
from jetstream_pt.config import FLAGS, create_engine_from_config_flags

logging.getLogger().setLevel(logging.ERROR)

flags.DEFINE_string("sharegpt_path", "", "path to sharegpt json file")


def run_prefill_time(engine, params, decode_state, seqlen, profiler_started):
  """Run prefill and measure time."""
  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)

  text = "This is a beautiful day"
  tokens, true_length = tokenizer.encode(
      text, is_bos=True, prefill_lengths=[seqlen]
  )

  for _ in range(3):
    prefill_result, _ = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=jnp.int32(1)
    )

  nums = 5
  start = time.perf_counter()
  for i in range(nums):
    if i == nums - 1 and FLAGS.profiling_prefill and not profiler_started:
      jax.profiler.start_trace(FLAGS.profiling_output)
      profiler_started = True

    prefill_result, _ = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=jnp.int32(i)
    )
  jax.block_until_ready(decode_state)

  end = time.perf_counter()
  return (end - start) / nums, decode_state, profiler_started


MAXTEXT_PREFILL = {
    16: 0,
    32: 0,
    64: 14.02,
    128: 18.29,
    256: 23.59,
    512: 35.28,
    1024: 60.28,
}


def main(argv):
  """Main function to run engine offline."""
  engine = create_engine_from_config_flags()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  prefill_times = {}

  decode_state = engine.init_decode_state()
  profiler_started = False
  for exp in range(4, 11):
    batch = 2**exp
    runtime, decode_state, profiler_started = run_prefill_time(
        engine, params, decode_state, batch, profiler_started
    )
    prefill_times[batch] = runtime

  sampled_tokens_list = []

  for i in range(3):  # warm up
    # pylint: disable-next=all
    decode_state, sampled_tokens = engine.generate(
        params=params, decode_state=decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  profiling_output = FLAGS.profiling_output
  print("======= decode starting ===")

  dec_times = []
  for i in range(10):
    if profiling_output and i == 7 and not profiler_started:
      jax.profiler.start_trace(profiling_output)
      profiler_started = True
    start = time.perf_counter()
    # pylint: disable-next=all
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    dec_times.append(end - start)
    print(i, "decode time", (end - start))

  if profiler_started:
    jax.profiler.stop_trace()

  print("prefill ", prefill_times)
  avg_decode_times = sum(dec_times[2:]) / len(dec_times[2:])
  print("decode", avg_decode_times)

  prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
  decode_time_ms = sum(dec_times[2:]) * 1000 / 8

  largest_prefill = max(prefill_times.items())
  print("MAX tokens:", FLAGS.batch_size / avg_decode_times)

  time2 = (FLAGS.batch_size * FLAGS.max_decode_length) / (
      FLAGS.batch_size * largest_prefill[1]
      + FLAGS.max_decode_length * avg_decode_times
  )
  print("MAX tokens 2:", time2)

  sharegpt_path = FLAGS.sharegpt_path
  if sharegpt_path:
    analyze_sharegpt.do_simulation(
        sharegpt_path, prefill_times_ms, decode_time_ms
    )


if __name__ == "__main__":

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
