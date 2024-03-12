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

import time
import logging

from petstream.pets.llama2 import model_utils
from petstream.pets.llama2 import model_exportable
from petstream.pets import cache_manager
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
    'batch_size', 32, 'The batch size', required=False
)
_PROFILING_OUTPUT =flags.DEFINE_string(
    'profiling_output',
    '',
    'The profiling output',
    required=False,
)

_SIZE = flags.DEFINE_string('size', 'tiny', 'size of model')

_QUANTIZE=flags.DEFINE_bool(
    'quantize',
    False,
    '',
    required=False,
)

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
  if 'weight_scaler' in name:
    return x_sharding
  if "tok_embeddings." in name:
      return x_sharding 
  if "attention." in name:
      # wo = row
      if "wo" in name:
          return y_sharding 
      else:
          return x_sharding 
  if "feed_forward." in name:
      # w1 = col; w2 = row; w3 = col
      if "w2" in name:
          return y_sharding 
      else:
          return x_sharding 
  if "output" in name:
      return y_sharding 
  return replicated 

def make_cache(args, batch_size, sharding):
  """Creates a cache for each layer."""

  head_dim = args.dim // args.n_heads
  res = []
  kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
  for _ in range(args.n_layers):
    k_size = (batch_size, kv_heads, args.max_seq_len, head_dim)
    v_size = k_size
    res.append((
        jnp.zeros(
            k_size, dtype=jnp.int8,
            device=sharding
        ),
        jnp.zeros(
            v_size, dtype=jnp.int8, device=sharding
        ),
    ))
  return res


def main(argv):
  del argv
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  # See issue b/309529778 if it's turned on.
  jax.config.update('jax_dynamic_shapes', False)
  # Pytorch exports has int64 constants.
  # jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_traceback_filtering', 'off')
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  model_exportable.hanq_flag = True
  torch.set_default_dtype(torch.bfloat16)

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
  
  model_arg.quantize = _QUANTIZE.value

  start = time.perf_counter()
  model = model_exportable.Transformer(model_arg, None)
  #model.to(dtype=torch.bfloat16)
  end = time.perf_counter()
  print('init time', end - start)


  # run forward once in jax
  cpu_device = jax.devices('cpu')[0]
  with jax.default_device(cpu_device):
    jax_weights = make_state_dict_jax(model.state_dict())

  jax_weights = {
    key: jax.device_put(value, sharding_by_name(key))
    for key, value in jax_weights.items()}

  jax_weights['freqs_cis'] = jax.device_put(torch_xla2.tensor.t2j(model.freqs_cis), replicated)
  jax_weights['mask'] = jax.device_put(torch_xla2.tensor.t2j(model.mask), replicated)

  model_arg.n_kv_heads = model_arg.n_kv_heads or model_arg.n_heads

  tokens = jnp.arange( 
    0, model_arg.max_seq_len).reshape((1, model_arg.max_seq_len))

  input_indexes = jnp.full((1, ), 1024)
  cache_indexes = jnp.arange(0, model_arg.max_seq_len)
  
  input_indexes = cache_indexes
  # shard on num_heads
  cache_sharding = jsharding.NamedSharding(mesh, P(None, "x", None, None))
  caches = make_cache(model_arg, 1, cache_sharding)


  prefill = True 
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
      tokens, input_indexes, caches, prefill
    )
    paramst, argst = pytree.tree_map(torch_xla2.tensor.wrap, (params, args))
    if _QUANTIZE.value:
      caches_obj = [
        cache_manager.Int8KVCache(
          k, v, prefill, torch_xla2.tensor.wrap(input_indexes), 
          cache_sharding) for k, v in caches 
      ]
    else:
      if prefill:
        caches_obj = [
          cache_manager.KVCachePrefill() for k, v in argst[2]
        ]
      else:
        caches_obj = [
          cache_manager.KVCacheGenerate(
            k, v, cache_indexes, cache_sharding) for k, v in argst[2]
          
        ]

    argst = argst[:2] + (caches_obj, ) + argst[3:]
    with torch_xla2.tensor.XLADispatchMode():
      res = torch.func.functional_call(model, paramst, argst)
    res = pytree.tree_map(torch_xla2.tensor.unwrap, res), 
    new_caches = [torch_xla2.tensor.unwrap((m.cache_k, m.cache_v)) for m in caches_obj]
    return res[0], new_caches

  args = (
    tokens, input_indexes, cache_indexes, caches, prefill
  )

  key = jax.random.PRNGKey(0)

  for i in range(10):
    key, subkey = jax.random.split(key)
    start = time.perf_counter()
    tokens = jax.random.randint(subkey, (1, model_arg.max_seq_len), 0, 32000)
    #tokens = jax.device_put(tokens, x_sharding)
    res = call_model(jax_weights, tokens, input_indexes, cache_indexes, caches, True)
    caches = res[1]
    jax.tree_map(lambda r: r.block_until_ready(), res)
    end = time.perf_counter()
    print(i, 'prefill step', end - start)

  input_indexes = jnp.full((1, ), 1024)
  caches = make_cache(model_arg, model_arg.max_batch_size, cache_sharding)
  
  gcs_profiler_path = 'gs://hanq-random/single_decode'
  jax.profiler.start_trace(gcs_profiler_path)
  for i in range(10):
    key, subkey = jax.random.split(key)
    tokens = jax.random.randint(subkey, (model_arg.max_batch_size, 1), 0, 32000)
    #tokens = jax.device_put(tokens, x_sharding)
    tokens.block_until_ready()
    start = time.perf_counter()
    cache_indexes = torch_xla2.tensor.wrap(jnp.array([3], dtype=jnp.int32))
    res = call_model(jax_weights, tokens, input_indexes, cache_indexes, caches, False)
    caches = res[1]
    jax.tree_map(lambda r: r.block_until_ready(), res)
    end = time.perf_counter()
    print(i, 'decode step', end - start)
  jax.profiler.stop_trace()

  for k, v in model.layers[0].state_dict().items():
    print(k, v.shape, v.dtype)

  num_params = 0
  for k, v in jax_weights.items():
    num_params += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
  print('Number of param Gbytes', num_params / (1 << 30))

if __name__ == "__main__":
  app.run(main)
