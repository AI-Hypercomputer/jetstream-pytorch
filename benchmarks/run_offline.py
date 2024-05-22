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

import jax
import jax.numpy as jnp
# pylint: disable-next=all
from absl import app, flags
# pylint: disable-next=all
from benchmarks import analyze_sharegpt
from jetstream_pt.config import FLAGS, create_engine_from_config_flags

logging.getLogger().setLevel(logging.ERROR)

flags.DEFINE_string("sharegpt_path", "", "path to sharegpt json file")


def run_prefill_time(engine, params, decode_state, seqlen):
  """Run prefill and measure time."""
  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)

  text = "This is a beautiful day"
  tokens, true_length = tokenizer.encode(
      text, is_bos=True, prefill_lengths=[seqlen]
  )

  for _ in range(3):
    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=jnp.int32(1)
    )

  nums = 5
  start = time.perf_counter()
  for i in range(nums):
    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=jnp.int32(i)
    )
  jax.block_until_ready(decode_state)
  end = time.perf_counter()
  return (end - start) / nums, decode_state


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
  for batch, _ in MAXTEXT_PREFILL.items():
    runtime, decode_state = run_prefill_time(
        engine, params, decode_state, batch
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
    if i == 7 and profiling_output:
      jax.profiler.start_trace(profiling_output)
    start = time.perf_counter()
    # pylint: disable-next=all
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    dec_times.append(end - start)
    print(i, "decode time", (end - start))

  if profiling_output:
    jax.profiler.stop_trace()

  print("prefill ", prefill_times)
  print("decode", sum(dec_times) / 10)

  prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
  decode_time_ms = sum(dec_times) * 1000 / 10 / FLAGS.batch_size

  sharegpt_path = FLAGS.sharegpt_path
  if sharegpt_path:
    analyze_sharegpt.do_simulation(
        sharegpt_path, prefill_times_ms, decode_time_ms
    )


if __name__ == "__main__":

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
