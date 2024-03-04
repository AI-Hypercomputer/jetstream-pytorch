"""Generate Jax wrapper of the imported LLM model."""

from typing import Any, Optional

from flax import struct
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental import pjit
import jax.sharding as jsharding
from torch.fx import _pytree as fx_pytree
from torch.utils import _pytree as pytree

from petstream.pets.imported_model import Llama2ImportedModel
from petstream.pets.llama2.model_args import ModelArgs

P = jsharding.PartitionSpec


@struct.dataclass
class DecodeState:
  caches: Any
  pos: Any
  context_pos: Any
  gen_len: Any


@struct.dataclass
class LoopState:
  tokens: Any
  res: Any
  decode_state: DecodeState


def init_cache(
    model_args: ModelArgs,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
  """Initializes the all 0 cache value for decoding."""
  caches = []

  for _ in range(model_args.n_layers):
    k_size = (
        model_args.max_batch_size,
        model_args.n_kv_heads,
        model_args.max_seq_len,
        model_args.head_dim,
    )
    v_size = k_size
    cache_k = jnp.zeros(
        k_size,
        dtype=jnp.bfloat16 if model_args.bf16_enable else jnp.float32,
    )
    cache_v = jnp.zeros(
        v_size,
        dtype=jnp.bfloat16 if model_args.bf16_enable else jnp.float32,
    )
    caches.append((cache_k, cache_v))
  return caches


def init_random_cache(
    model_args: ModelArgs,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
  """Initializes the random cache value, for debugging purpose."""
  caches = []

  for _ in range(model_args.n_layers):
    k_size = (
        model_args.max_batch_size,
        model_args.n_kv_heads,
        model_args.max_seq_len,
        model_args.head_dim,
    )
    v_size = k_size
    cache_k = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=k_size,
        dtype=jnp.bfloat16 if model_args.bf16_enable else jnp.float32,
    )
    cache_v = jax.random.uniform(
        jax.random.PRNGKey(1),
        shape=v_size,
        dtype=jnp.bfloat16 if model_args.bf16_enable else jnp.float32,
    )
    caches.append((cache_k, cache_v))
  return caches


def init_prefill_state(
    model_args: ModelArgs, init_cache_fn=init_cache
) -> DecodeState:
  """Initializes the state.

  Args:
    model_args: model specific arguments
    init_cache_fn: The cache initialization function

  Returns:
    The data class passes inside the while loop for each iteration
  """
  caches = init_cache_fn(model_args)
  return DecodeState(
      caches,
      jnp.arange(
          0, model_args.max_seq_len, dtype=jnp.int32
      ),  # input_pos_tensor, only size matters
      jnp.arange(
          0, model_args.max_seq_len, dtype=jnp.int32
      ),  # context_pos_tensor, only size matters
      jnp.zeros((model_args.max_batch_size, 1), dtype=jnp.int32),
      # gen_len, the number of generated results
  )


def init_decode_state(
    model_args: ModelArgs, init_cache_fn: Any = init_cache
) -> DecodeState:
  """Initializes the state.

  Args:
    model_args: model specific arguments
    init_cache_fn: The cache initialization function

  Returns:
    The data class passes inside the while loop for each iteration
  """
  caches = init_cache_fn(model_args)
  return DecodeState(
      caches,
      jnp.array([0], dtype=jnp.int32),
      # input_pos_tensor, only size matters
      update_context_pos(
          model_args, jnp.arange(0, model_args.max_seq_len, dtype=jnp.int32)
      ),  # context_pos_tensor, only size matters
      jnp.zeros(
          (model_args.max_batch_size, 1), dtype=jnp.int32
      ),  # gen_len, the number of generated results
  )


def initial_result(model_args: ModelArgs) -> jnp.ndarray:
  return jnp.zeros(
      (
          model_args.max_batch_size,
          model_args.max_seq_len,
      ),
      dtype=jnp.int32,
  )  # initial_results


def sampling(logits: Any, batch_size: int) -> jnp.ndarray:
  return (
      jnp.argmax(logits[:, -1], axis=-1)
      .reshape(batch_size, -1)
      .astype(jnp.int32)
  )


@struct.dataclass
class Prefix:
  caches: Any
  token: Any


@struct.dataclass
class PrefillResult:
  res: Any


def prefill(
    prefill_states: LoopState,
    model: Llama2ImportedModel,
    model_args: ModelArgs,
    weights: ...,
) -> tuple[Prefix, PrefillResult]:
  """Prefills the KV cache."""
  in_spec, out_spec = model.input_output_spec()
  # in_spec = model.jax_program.exported_program.call_spec.in_spec
  # out_spec = model.jax_program.exported_program.call_spec.out_spec
  inputs = fx_pytree.tree_flatten_spec(
      (
          prefill_states.tokens,
          prefill_states.decode_state.pos,
          prefill_states.decode_state.context_pos,
          prefill_states.decode_state.caches,
          True,
      ),
      in_spec,
  )
  prefill_result = model.prefill(weights, inputs)

  # The initial KV cache
  logits, caches = pytree.tree_unflatten(prefill_result, out_spec)

  print(
      'Prefill inputs: %s \nresult: %s \nlogits: %s'
      % (inputs, prefill_result, logits)
  )
  # Calculates the next token, prefill have batch size of 1
  next_token = sampling(logits, model_args.max_batch_size)

  # The update target position of next tokens should be [0, ...] for batch
  # dimension
  current_pos = prefill_states.decode_state.context_pos[0].reshape(
      1,
  )
  insert_pos = jnp.concatenate(
      (
          jnp.zeros((1,), dtype=jnp.int32),
          current_pos,
      ),
      axis=0,
  )

  print(
      'Prefill res: %s\n next_token: %s\n insert_pos: %s'
      % (prefill_states.res, next_token, insert_pos)
  )
  res = jax.lax.dynamic_update_slice(prefill_states.res, next_token, insert_pos)
  print('Updated res: ', res)
  return Prefix(caches, next_token), PrefillResult(res)


# starting from (0 1 2 3) -> (3 0 1 2) -> (2 3 0 1) -> (1 2 3 0) -> (0 1 2 3)
def update_context_pos(
    model_args: ModelArgs, context_pos: jnp.ndarray
) -> jnp.ndarray:
  return (context_pos - 1) % (model_args.max_seq_len + model_args.infer_length)


# starting from 0, 1, 2, ... 7, loop over to 0
def update_pos(model_args: ModelArgs, pos: jnp.ndarray) -> jnp.ndarray:
  return (pos + 1) % (model_args.max_seq_len + model_args.infer_length)


def generate_shlo(
    generate_states: LoopState,
    model: Llama2ImportedModel,
    model_args: ModelArgs,
    weights: ...,
) -> tuple[jnp.ndarray, Any]:
  """Generates the KV cache for each iteration.

  For non Jet API, all the pointers will be updated in this function. For Jet
  API, the pointer increment will be handled in the Jet implementation.

  Args:
    generate_states: the decode state for generate step
    model: the stable HLO model
    model_args: odel parameters
    weights: model weights

  Returns:
    The next token
    The updated kv cache
  """
  in_spec, out_spec = model.input_output_spec()
  inputs = fx_pytree.tree_flatten_spec(
      (
          generate_states.tokens,
          generate_states.decode_state.pos,
          generate_states.decode_state.context_pos,
          generate_states.decode_state.caches,
          False,
      ),
      in_spec,
  )
  result = model.decode(weights, inputs)
  print('Jet Prefill result: ', result)
  logits, caches = pytree.tree_unflatten(result, out_spec)

  next_token = sampling(logits, model_args.max_batch_size)

  return next_token, caches


def generate(
    generate_states: LoopState,
    model: Llama2ImportedModel,
    model_args: ModelArgs,
    weights: ...,
) -> LoopState:
  next_token, caches = generate_shlo(
      generate_states, model, model_args, weights
  )
  return LoopState(
      next_token,
      jax.lax.dynamic_update_slice(
          generate_states.res,
          next_token,
          [jnp.int32(0), generate_states.decode_state.gen_len[0][0] + 1],
      ),
      DecodeState(
          caches,
          generate_states.decode_state.pos + 1,
          generate_states.decode_state.context_pos + 1,
          generate_states.decode_state.gen_len + 1,
      ),
  )


def _shard_variables(name_to_pos, num_of_partitions):
  """Shard variables.

  Args:
    name_to_pos: the weight name to its position in the state dictionary
    num_of_partitions: the number of sharding partitions

  Returns:
    position_to_sharding: dictionary of state dictionary position to sharding
  """
  position_to_sharding = {}
  for name, i in name_to_pos.items():
    if 'tok_embeddings.' in name:
      position_to_sharding[i] = (num_of_partitions, 1)
      continue
    if 'attention.' in name:
      if 'wo' in name:
        position_to_sharding[i] = (num_of_partitions, 1)
      else:
        position_to_sharding[i] = (1, num_of_partitions)
      continue
    if 'feed_forward.' in name:
      if 'w2' in name:
        position_to_sharding[i] = (num_of_partitions, 1)
      else:
        position_to_sharding[i] = (1, num_of_partitions)
      continue
    if 'output' in name:
      position_to_sharding[i] = (1, num_of_partitions)
      continue
    position_to_sharding[i] = (-1, -1)
  return position_to_sharding


def _replicated_sharding(num_of_partitions):
  mesh = jsharding.Mesh(
      mesh_utils.create_device_mesh((num_of_partitions, 1)),
      axis_names=('x', 'y'),
  )
  return jsharding.NamedSharding(mesh, P())


def _input_sharding_spec(input_size, num_of_partitions):
  replicated = _replicated_sharding(num_of_partitions)
  return replicated if input_size == 1 else [replicated] * input_size


def _variables_sharding_spec(name_to_pos, weight_size, num_of_partitions):
  """Shard weights."""
  mesh = jsharding.Mesh(
      mesh_utils.create_device_mesh((num_of_partitions, 1)),
      axis_names=('x', 'y'),
  )
  y_sharding = jsharding.NamedSharding(mesh, P(None, 'x'))
  x_sharding = jsharding.NamedSharding(mesh, P('x'))
  replicated = jsharding.NamedSharding(mesh, P())
  weights_sharding_mapping = _shard_variables(name_to_pos, num_of_partitions)
  weight_sharding = [replicated] * weight_size

  for i, sharding in weights_sharding_mapping.items():
    if sharding[0] > 1:
      weight_sharding[i] = x_sharding
    elif sharding[1] > 1:
      weight_sharding[i] = y_sharding
    else:
      weight_sharding[i] = replicated

  return weight_sharding


# For Jet API implementation only
def shard_actual_variables(
    name_to_pos: dict[str, int], weights: ..., num_of_partitions: int
):
  weight_sharding = _variables_sharding_spec(
      name_to_pos, len(weights), num_of_partitions
  )
  print('Live array:', jax.live_arrays())
  print('Number of weights:', len(weights))
  for w, sharding in zip(weights, weight_sharding):
    print('Shape: %s dtype: %s sharding %s:' % (w.shape, w.dtype, sharding))
  return jax.lax.with_sharding_constraint(weights, weight_sharding)


def shard_weights(keys, values, num_of_partitions: int):
  """Shard weights."""

  mesh = jsharding.Mesh(
      mesh_utils.create_device_mesh((num_of_partitions, 1)),
      axis_names=('x', 'y'),
  )
  y_sharding = jsharding.NamedSharding(mesh, P(None, 'x'))
  x_sharding = jsharding.NamedSharding(mesh, P('x'))
  replicated = jsharding.NamedSharding(mesh, P())

  weight_sharding = [replicated] * len(keys)

  for name in keys:
    if 'tok_embeddings.' in name:
      weight_sharding.append(x_sharding)
      continue
    if 'attention.' in name:
      if 'wo' in name:
        weight_sharding.append(x_sharding)
      else:
        weight_sharding.append(y_sharding)
      continue
    if 'feed_forward.' in name:
      if 'w2' in name:
        weight_sharding.append(x_sharding)
      else:
        weight_sharding.append(y_sharding)
      continue
    if 'output' in name:
      weight_sharding.append(y_sharding)
      continue

  print('Live array:', jax.live_arrays())
  print('Number of weights:', len(values))
  for name, w, sharding in zip(keys, values, weight_sharding):
    print(
        'Name: %s Shape: %s dtype: %s sharding %s:'
        % (name, w.shape, w.dtype, sharding)
    )
  return jax.lax.with_sharding_constraint(values, weight_sharding)
