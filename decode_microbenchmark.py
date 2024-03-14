import datetime
import sys
import jax
import jax.numpy as jnp
import json
import numpy as np
import time


from jetstream.engine import token_utils
from absl.testing import absltest

import os
import sys

from petstream import jet_engine2

import gc
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path',
    'petstream/pets/tokenizer.model',
    'The tokenizer model path',
    required=False,
)
_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path', None, 'Directory for .pth checkpoints', required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    'bf16_enable', False, 'Whether to enable bf16', required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    'context_length', 1024, 'The context length', required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32, 'The batch size', required=False
)
_PROFILING_OUTPUT =flags.DEFINE_string(
    'profiling_output',
    '',
    'The profiling output',
    required=False,
)

_SIZE = flags.DEFINE_string('size', 'tiny', 'size of model')

_QUANTIZE_WEIGHTS = flags.DEFINE_bool('quantize_weights', False, 'weight quantization')
_QUANTIZE_KV_CACHE = flags.DEFINE_bool('quantize_kv_cache', False, 'kv_cache_quantize')
_MAX_CACHE_LENGTH = flags.DEFINE_integer('max_cache_length', 1024, 'kv_cache_quantize')

def profile(func):
  def wrapper(*args, **kwargs):
    jax.profiler.start_trace(_PROFILING_OUTPUT.value) 
    start = time.perf_counter()
    res = func(*args, **kwargs)
    end = time.perf_counter()
    jax.profiler.stop_trace() 
    return (end - start), res
  return wrapper

# def print_objects():
#   print(f"Objects {len(gc.get_objects())}")

@profile
def prefill_benchmark_loop(engine, decode_state, params, tokens, true_length, steps=10):
  for i in range(steps + 1):
    slot = i % _BATCH_SIZE.value
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, decode_state, slot=int(slot))
  jax.block_until_ready(decode_state)
  return decode_state

def prefill_benchmark(
    engine, decode_state, params, tokens, true_length, steps=10, num_model_params=None): 

  # warmup
  for _ in range(3):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, decode_state, slot=0)
    jax.block_until_ready(decode_state)

  steps = 10
  time_in_s, decode_state = prefill_benchmark_loop(
    engine, decode_state, params, tokens, true_length)
  prefill_average_ms = 1000 * time_in_s / steps

  print(f"Prefill results:\n"
        f"\tPrefill step average time: {prefill_average_ms:.2f}ms\n"
        f"\tPrefill total TFLOPs: NA \n"
        f"\tPrefill TFLOPs/sec/device: \n")
  return decode_state, {"prefill_time_in_ms": prefill_average_ms, 
          "prefill_total_tflops": "" , 
          "prefill_tflops_per_sec_per_device": ""}

@profile
def ar_benchmark_loop(engine, decode_state, params, tokens, true_length, global_batch_size, steps=10):
  steps = 10
  for i in range(steps):
    decode_state, sampled_tokens = engine.generate(params, decode_state)
  jax.block_until_ready(decode_state)
  return decode_state


def calc_cache_size(decode_state):
    res = 0
    for k, v in decode_state.caches:
        res += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    for k, v in decode_state.cache_scales:
        res += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    return res


def ar_benchmark(engine, decode_state, params, tokens, true_length, steps=10, model_size=None): 
  global_batch_size = _BATCH_SIZE.value

  # Warmup
  for _ in range(3):
    decode_state, sampled_tokens = engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)

  time_in_s, decode_state = ar_benchmark_loop(engine, decode_state, params, tokens, true_length, global_batch_size, steps=steps)
  seconds_per_step = time_in_s / steps
  ar_average_ms = seconds_per_step*1000
  total_throughput = _BATCH_SIZE.value / seconds_per_step

  cache_size = calc_cache_size(decode_state)
  GB_per_step_per_device = (model_size + cache_size) / 2**30 / jax.device_count()
  bw_per_device = GB_per_step_per_device/seconds_per_step
  print(f"AutoRegressive results:\n"
        f"\tAR step average time: {ar_average_ms:.2f}ms\n"
        f"\tAR global batch size: {global_batch_size}\n"
        f"\tAR throughput: {total_throughput:.2f} tokens/second\n"
        f"\tAR memory bandwidth per device: {bw_per_device:.2f} GB/s")
  return decode_state, {"ar_step_in_ms": ar_average_ms, 
          "ar_global_batch_size": global_batch_size, 
          "ar_total_throughput_tokens_per_second": total_throughput,
          "ar_device_bandwidth_GB_per_second": bw_per_device}


def create_engine():
  max_prefill_predict_length = 1024
  max_target_length = max_prefill_predict_length + 1024
  devices = jax.devices()
  start = time.perf_counter()
  engine = jet_engine2.create_pytorch_engine(
        devices=devices,
        tokenizer_path=_TOKENIZER_PATH.value,
        ckpt_path=_CKPT_PATH.value,
        bf16_enable=True,
        param_size=_SIZE.value,
        context_length=_CONTEXT_LENGTH.value,
        batch_size=_BATCH_SIZE.value,
        quantize_weights=_QUANTIZE_WEIGHTS.value,
        quantize_kv=_QUANTIZE_KV_CACHE.value,
        max_cache_length = _MAX_CACHE_LENGTH.value,
  )

  print('Initialize engine', time.perf_counter() - start)
  return engine

def main(argv):
  del argv
  engine = create_engine()
  params = engine.load_params()
  prefill_lengths = [1024]#, 1024] #[128, 256#, 512, 1024, 2048, 4096]
  num_steps = 10
  text = 'today is a beautiful day isnt'

  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  decode_state = engine.init_decode_state()

  results = {}
  for prefill_length in prefill_lengths:
    tokens, true_length = token_utils.tokenize_and_pad(
        text, vocab, is_bos=True, prefill_lengths=[prefill_length])
    print(f"Prompt tokenized to size {tokens.size}")
    decode_state, prefill_results = prefill_benchmark(
        engine, decode_state, params, tokens, 
        true_length, num_steps, num_model_params=None)
    decode_state, ar_results = ar_benchmark(
        engine, decode_state, params, tokens, 
        true_length, num_steps, 0)
    prefill_results.update(ar_results)
    results[prefill_length] = prefill_results

  with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)


if __name__ == "__main__":
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)