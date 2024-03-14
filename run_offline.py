from absl import app
from absl import flags
from absl import logging
import sys
import jax
import jax.numpy as jnp
import numpy as np

from jetstream.engine import token_utils
from absl.testing import absltest

import os
import sys

from petstream import jet_engine2 as je
import time
import logging

logging.getLogger().setLevel(logging.ERROR)


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


def create_engine():
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  max_prefill_predict_length = 1024
  max_target_length = max_prefill_predict_length + 256
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
        max_cache_length = _MAX_CACHE_LENGTH.value,
  )

  print('Initialize engine', time.perf_counter() - start)
  return engine


def run_prefill_time(engine, params, decode_state, seqlen):
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(
    metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer

  text = 'This is a beautiful day'
  tokens, true_length = token_utils.tokenize_and_pad(
    text, vocab, is_bos=True, prefill_lengths=[seqlen])

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


MAXTEXT_PREFILL = {64 : 14.02, 128:18.29, 256:23.59, 512:35.28, 1024: 60.28}

def main(argv):

  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print('Load params ', time.perf_counter() - start)

  prefill_times = {}
  slot = jnp.int32(1)

  decode_state = engine.init_decode_state()
  for batch in MAXTEXT_PREFILL.keys():
    runtime, decode_state = run_prefill_time(engine, params, decode_state, batch)
    prefill_times[batch] = runtime

  sampled_tokens_list = []

  for i in range(3): # warm up
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  print('======= decode starting ===')
  if _PROFILING_OUTPUT.value:
    jax.profiler.start_trace(_PROFILING_OUTPUT.value)
  for i in range(10):
    start = time.perf_counter()
    decode_state, sampled_tokens = engine.generate(
      params, decode_state
    )
    jax.block_until_ready(decode_state)
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    print(i, 'decode time', (end - start))

  if _PROFILING_OUTPUT.value:
    jax.profiler.stop_trace()

  print(prefill_times)


if __name__ == "__main__":
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
