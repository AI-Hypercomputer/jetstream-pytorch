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
from absl import app
from absl import flags

import jax
import jax.numpy as jnp

from jetstream_pt import engine as je
# pylint: disable-next=all
from benchmarks import analyze_sharegpt


logging.getLogger().setLevel(logging.ERROR)


FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    "tokenizer_path",
    "tokenizer.model",
    "The tokenizer model path",
    required=False,
)
_CKPT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Directory for .pth checkpoints", required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    "bf16_enable", False, "Whether to enable bf16", required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    "context_length", 1024, "The context length", required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 32, "The batch size", required=False
)
_PROFILING_OUTPUT = flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)

_SIZE = flags.DEFINE_string("size", "tiny", "size of model")

_QUANTIZE_WEIGHTS = flags.DEFINE_bool(
    "quantize_weights", False, "weight quantization"
)
_QUANTIZE_KV_CACHE = flags.DEFINE_bool(
    "quantize_kv_cache", False, "kv_cache_quantize"
)
_MAX_CACHE_LENGTH = flags.DEFINE_integer(
    "max_cache_length", 1024, "kv_cache_quantize"
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "", "model_name"
)
_SHARDING_CONFIG = flags.DEFINE_string(
  "sharding_config", "", "path to sharding config"
)
_SHAREGPT_PATH = flags.DEFINE_string(
  "sharegpt_path", "", "path to sharegpt json file"
)


def create_engine():
  """Create a pytorch engine."""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()
  engine = je.create_pytorch_engine(
      devices=devices,
      tokenizer_path=_TOKENIZER_PATH.value,
      ckpt_path=_CKPT_PATH.value,
      bf16_enable=True,
      param_size=_SIZE.value,
      context_length=_CONTEXT_LENGTH.value,
      batch_size=_BATCH_SIZE.value,
      quantize_weights=_QUANTIZE_WEIGHTS.value,
      quantize_kv=_QUANTIZE_KV_CACHE.value,
      max_cache_length=_MAX_CACHE_LENGTH.value,
      model_name=_MODEL_NAME.value,
      sharding_config=_SHARDING_CONFIG.value,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


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
  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  prefill_times = {}

  if _PROFILING_OUTPUT.value:
    jax.profiler.start_trace(_PROFILING_OUTPUT.value)
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

  print("======= decode starting ===")
  dec_times = []
  for i in range(10):
    start = time.perf_counter()
    # pylint: disable-next=all
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    dec_times.append(end - start)
    print(i, "decode time", (end - start))

  if _PROFILING_OUTPUT.value:
    jax.profiler.stop_trace()

  print("prefill ", prefill_times)
  print("decode", sum(dec_times) / 10)

  prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
  decode_time_ms = sum(dec_times) * 1000 / 10 / _BATCH_SIZE.value

  if _SHAREGPT_PATH.value:
    analyze_sharegpt.do_simulation(_SHAREGPT_PATH.value, prefill_times_ms, decode_time_ms)


if __name__ == "__main__":

  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
