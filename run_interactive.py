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

from jetstream_pt import engine as je
import time
import logging

logging.getLogger().setLevel(logging.ERROR)


FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path',
    'tokenizer.model',
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


def main(argv):

  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print('Load params ', time.perf_counter() - start)

  prefill_times = {}
  slot = jnp.int32(0)
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(
    metadata.path, metadata.extra_ids)
  tokenizer = vocab.tokenizer

  while True:
    # text = input('Text >>>> ')
    text = 'I believe the meaning of life is'
    decode_state = engine.init_decode_state()
    tokens, true_length = token_utils.tokenize_and_pad(text, vocab, is_bos=True)
    # tokens = tokenizer.encode(text)
    # tokens = [tokenizer.bos_id()] + tokens
    print('Encoded tokens are: ', tokens)

    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = engine.insert(
        prefill_result, decode_state, slot=slot
    )
    sampled_tokens_list = []
    for i in range(100):
      decode_state, sampled_tokens = engine.generate(
        params, decode_state
      )
      tstart, end = sampled_tokens.tokens_idx
      sampled_tokens_list.append(sampled_tokens.data[0, 0].item())

    print('---- ans ----')
    print(sampled_tokens_list)
    print(tokenizer.decode(sampled_tokens_list))
    break



  if _PROFILING_OUTPUT.value:
    jax.profiler.stop_trace()



if __name__ == "__main__":
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
