import functools
from absl import app
from absl import flags
from absl import logging
import sys
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
import numpy as np

from jetstream.engine import token_utils
from absl.testing import absltest

import os
import sys

from petstream import jet_engine as je
import time
import logging

from petstream.pets.llama2 import model_utils
from petstream.pets.llama2 import model_exportable
import torch_xla2
import torch_xla2.extra
import torch
from torch.utils import _pytree as pytree

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
    'batch_size', 1, 'The batch size', required=False
)
_PROFILING_OUTPUT =flags.DEFINE_string(
    'profiling_output',
    '',
    'The profiling output',
    required=False,
)

_SIZE = flags.DEFINE_string('size', 'tiny', 'size of model')

def make_state_dict_jax(model_args_meta):
    def make_array(t):
        return jnp.ones(
            t.shape, dtype=torch_xla2.tensor.t2j_dtype(t.dtype))
    return pytree.tree_map_only(torch.Tensor, make_array, model_args_meta)


import jax.sharding as jsharding
from jax.experimental import mesh_utils

P = jsharding.PartitionSpec
num_devices = len(jax.devices())
mesh = jsharding.Mesh(
    mesh_utils.create_device_mesh((num_devices, 1)),
    axis_names=("x", "y"),
)
y_sharding = jsharding.NamedSharding(mesh, P(None, "x"))
x_sharding = jsharding.NamedSharding(mesh, P("x"))
replicated = jsharding.NamedSharding(mesh, P())
def sharding_by_name(name):
  if "tok_embeddings." in name:
      return x_sharding 
  if "attention." in name:
      if "wo" in name:
          return x_sharding 
      else:
          return y_sharding 
  if "feed_forward." in name:
      if "w2" in name:
          return x_sharding 
      else:
          return y_sharding 
  if "output" in name:
      return y_sharding 
  return replicated 


def main(argv):
  del argv
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  model_exportable.hanq_flag = True

  max_prefill_predict_length = _CONTEXT_LENGTH.value
  max_target_length = max_prefill_predict_length + 256

  devices = jax.devices()

  model_arg = model_utils.get_arg(
    param_size=_SIZE.value,
    seqlen=max_prefill_predict_length,
    batch_size=_BATCH_SIZE.value,
    vocab_size=32000,
    bf16_enable=True,
  )
  model_arg.device = 'meta'  # don't actually allocate tensosr
  

  start = time.perf_counter()
  model = model_exportable.Transformer(model_arg)
  model.to(dtype=torch.bfloat16)
  end = time.perf_counter()
  print('init time', end - start)

  # run forward once in jax
  cpu_device = jax.devices('cpu')[0]
  with jax.default_device(cpu_device):
    jax_weights = make_state_dict_jax(model.state_dict())

  jax_weights = {
    key: jax.device_put(value, sharding_by_name(key))
    for key, value in jax_weights.items()}
  #import pdb; pdb.set_trace()

  jax_weights['freqs_cis'] = jax.device_put(torch_xla2.tensor.t2j(model.freqs_cis), replicated)
  jax_weights['mask'] = jax.device_put(torch_xla2.tensor.t2j(model.mask), replicated)

  model_arg.n_kv_heads = model_arg.n_kv_heads or model_arg.n_heads

  tokens = jnp.arange( 
    0, model_arg.max_batch_size).reshape((model_arg.max_batch_size, 1))

  input_indexes = jnp.full((1, ), 1024)
  cache_indexes = jnp.arange(0, model_arg.max_seq_len)
  caches_torch = model_utils.make_cache(model_arg, model_arg.max_batch_size)
  caches = pytree.tree_map_only(
    torch.Tensor, torch_xla2.tensor.t2j, caches_torch)

  cache_sharding = jsharding.NamedSharding(mesh, P(None, None, "x", None))
  caches = jax.device_put(caches, cache_sharding)

  prefill = False
  #todo need to shard inputs
  @functools.partial(
    jax.jit, 
    static_argnums=(5, ),
    donate_argnums=(4, ),
  )
  def call_model(
    params, 
    tokens, input_indexes, cache_indexes, caches, prefill
  ):
    args = (
      tokens, input_indexes, cache_indexes, caches, prefill
    )
    paramst, argst = pytree.tree_map(torch_xla2.tensor.wrap, (params, args))
    with torch_xla2.tensor.XLADispatchMode():
      res = torch.func.functional_call(model, paramst, argst)
    return pytree.tree_map(torch_xla2.tensor.unwrap, res)

  args = (
    tokens, input_indexes, cache_indexes, caches, prefill
  )

  key = jax.random.PRNGKey(0)
  
  for i in range(20):
    key, subkey = jax.random.split(key)
    start = time.perf_counter()
    tokens = jax.random.randint(subkey, (model_arg.max_batch_size, 1), 0, 32000)
    #tokens = jax.device_put(tokens, x_sharding)
    args = (
      tokens, input_indexes, cache_indexes, caches, prefill
    )
    res = call_model(jax_weights, tokens, input_indexes, cache_indexes, caches, prefill)
    caches = res[1]
    jax.tree_map(lambda r: r.block_until_ready(), res)
    end = time.perf_counter()
    print(i, 'decode step', end - start)

if __name__ == "__main__":
  app.run(main)
