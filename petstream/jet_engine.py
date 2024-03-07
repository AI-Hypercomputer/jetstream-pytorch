"""Implement Jet Engine API."""

import copy
from typing import Any, List, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
import jax.sharding as jsharding

from jetstream.engine import engine_api, tokenizer_pb2, token_utils
import petstream.jax_wrapper as jw
from .pets.imported_model import ImportedModel
from .pets.llama2.model_args import ModelArgs
from .pets.llama2 import model_exportable, model_utils
from .pets import tokenizer

DecodeState = jw.LoopState


Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array

Prefix = jw.Prefix
PrefillInputs = jax.Array

class PyTorchEngine(engine_api.Engine):
  """Wraps functions to the Jet Engine API format."""

  def __init__(
      self,
      devices: Union[List[Any], Mesh],
      imported_model: ImportedModel,
      tokenizer: Any,
      samples_per_slot: int,
      max_decode_length: int,
  ):
    self.devices = devices
    self.num_of_partitions = len(devices)
    self.imported_model = imported_model
    self.param: ModelArgs = self.imported_model.model_args
    self.tokenizer = tokenizer
    self.samples_per_slot_input = samples_per_slot
    self._max_decode_length = max_decode_length

    # TODO(lancewang): Test may not work, need to verify.
    self.init_decode_state = jax.jit(
        self.init_decode_state, out_shardings=self.get_decode_state_sharding()
    )
    self.prefill = jax.jit(self.prefill)
    self.insert = jax.jit(self.insert)
    self.generate = jax.jit(self.generate)

  def init_decode_state(
      self,
  ) -> DecodeState:
    return DecodeState(
        jnp.zeros((self.param.max_batch_size, 1), dtype=jnp.int32),
        jw.initial_result(self.param),
        jw.init_decode_state(self.param),
    )

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      prefill_inputs: PrefillInputs,  # PrefillInputs[jax.Array],
      true_length: int
  ) -> Prefix:
    # ) -> Prefix:
    model_args = copy.deepcopy(self.param)
    # Prefill generates 1 cache entry a time
    model_args.max_batch_size = 1

    if isinstance(prefill_inputs, jax.Array):
      batched_token = prefill_inputs.reshape(1, -1)
    else:
      raise TypeError(
          'Input tokens should be of type Jax Array, but receiving:'
          ' {prefill_inputs}'
      )

    # Pytorch model needs the whole decode state.
    init_decode_state = DecodeState(
        batched_token,
        jw.initial_result(model_args),
        jw.init_prefill_state(model_args),
    )
    prefix, prefill_results = jw.prefill(
        init_decode_state, self.imported_model, model_args, params
    )
    return prefix

  def shrink_prefix(
      self,
      prefix: Prefix,
      new_length: int,
  ) -> Prefix:
    return prefix

  def _update_prefill_slot(self, operand, update, slot, pos):
    return jax.tree.map(
        lambda cache, new_entry: jax.lax.dynamic_update_slice(
            cache,
            new_entry,
            [slot, pos, jnp.int32(0), jnp.int32(0)],
        ),
        operand,
        update,
    )

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    logging.info(
        'Jet input prefix: %s, decode state before insert: %s',
        prefix,
        decode_state,
    )
    return DecodeState(
        tokens=jax.lax.dynamic_update_slice(
            decode_state.tokens, prefix.token, [slot, jnp.int32(0)]
        ),
        res=jax.lax.dynamic_update_slice(
            decode_state.res, prefix.token, [slot, jnp.int32(0)]
        ),
        decode_state=jw.DecodeState(
            caches=self._update_prefill_slot(
                decode_state.decode_state.caches,
                prefix.caches,
                slot,
                decode_state.decode_state.pos[0],
            ),
            pos=decode_state.decode_state.pos,
            context_pos=decode_state.decode_state.context_pos,
            gen_len=jax.lax.dynamic_update_slice(
                decode_state.decode_state.gen_len,
                jnp.array([[1]], dtype=jnp.int32),
                [slot, jnp.int32(0)],
            ),
        ),
    )

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    logging.info('Jet decode state before generate: %s', decode_state)
    next_token, caches_kv = jw.generate_shlo(
        decode_state, self.imported_model, self.param, params
    )
    logging.info(
        'Jet generate next_token: %s, \ncaches: %s', next_token, caches_kv
    )
    data = jnp.concatenate(
        [
            next_token,
            jnp.ones_like(next_token),
            decode_state.decode_state.gen_len,
        ],
        axis=-1,
    )

    # [0] is the batch dimension, [1] normally should be 1
    length = next_token.shape[1]
    result_tokens = engine_api.ResultTokens(
        data=data,
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )

    logging.info(
        'Jet decode state after generate: %s \nresult token: %s',
        decode_state,
        result_tokens,
    )

    def _update_result(results, new_result, pos):
      row_indices = jnp.arange(self.param.max_batch_size)[:, None]
      pos = pos.astype(jnp.int32)
      update_indices = (
          jnp.zeros(results.shape, dtype=bool).at[row_indices, pos].set(True)
      )
      return jnp.where(update_indices, new_result, results)

    return (
        DecodeState(
            tokens=next_token,
            res=_update_result(
                decode_state.res,
                next_token,
                jnp.squeeze(decode_state.decode_state.gen_len),
            ),
            decode_state=jw.DecodeState(
                caches=caches_kv,
                pos=jw.update_pos(
                    self.param,
                    decode_state.decode_state.pos,
                ),
                context_pos=jw.update_context_pos(
                    self.param,
                    decode_state.decode_state.context_pos,
                ),
                gen_len=decode_state.decode_state.gen_len + 1,
            ),
        ),
        result_tokens,
    )

  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    tokenizer = tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)
    return tokenizer

  def join_prefixes(
      self,
      prefix1: engine_api.Prefix,
      length1: int,
      prefix2: engine_api.Prefix,
      length2: int,
  ) -> tuple[engine_api.Prefix, int]:
    raise NotImplementedError('join_prefixes not supported')

  def load_params(self) -> Params:
    weights = self.imported_model.load_weights()
    return self.imported_model.place_weights(weights)

  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices()[0]

  @property
  def replicated_sharding(self) -> jax.sharding.NamedSharding:
    """Returns sharding to specify replication of a single object."""
    return jax.sharding.NamedSharding(self.mesh, P())

  def get_cache_sharding(self) -> jw.DecodeState:
    """Returns the shardings necessary to transfer data between engines."""
    return jw.DecodeState(
        jsharding.NamedSharding(self.mesh, P('x', None, None, None)),
        self.replicated_sharding,
        self.replicated_sharding,
        self.replicated_sharding,
    )

  # The best sharding is in dimension 2.
  def get_prefix_destination_sharding(self) -> Prefix:
    """Returns the shardings necessary to transfer data between engines."""
    return Prefix(
        jsharding.NamedSharding(self.mesh, P(None, None, 'x', None)),
        self.replicated_sharding,
    )

  # The best sharding is in dimension 0.
  def get_decode_state_sharding(self) -> DecodeState:
    """Gets the shardings corresponding to the decode state."""
    return DecodeState(
        self.replicated_sharding,
        self.replicated_sharding,
        self.get_cache_sharding(),
    )

  def get_prefix_sequence_ddim(self) -> Any:
    """Returns the index of the sequence dim in the prefix type."""
    return self.get_prefix_destination_sharding()

  @property
  def max_concurrent_decodes(self) -> int:
    return self.param.max_batch_size

  @property
  def samples_per_slot(self) -> int:
    return self.samples_per_slot_input

  @property
  def max_prefill_length(self) -> int:
    return self.param.max_seq_len

  @property
  def max_decode_length(self) -> int:
    """Maximum decode length."""
    return self._max_decode_length

  @property
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""
    return jsharding.Mesh(
        mesh_utils.create_device_mesh((self.num_of_partitions, 1)),
        axis_names=('x', 'y'),
    )


def create_pytorch_engine(
    devices: list[Any],
    tokenizer_path: str,
    ckpt_path: Optional[str] = None,
    samples_per_slot: int = 1,
    bf16_enable: bool = False,
    param_size: str = '7b',
    context_length: int = 1024,
    batch_size: int = 1,
    max_decode_length: int = 4096,
    model_name = "llama"
) -> PyTorchEngine:
  """Returns: The pytorch engine."""

  # See issue b/309529778 if it's turned on.
  jax.config.update('jax_dynamic_shapes', False)
  # Pytorch exports has int64 constants.
  # jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_traceback_filtering', 'off')

  sample_input_prefill = None
  sample_input_decode = None
  tokenizer = token_utils.load_vocab(tokenizer_path)
  pt_model = None
  shard_weights_fn = None
  if model_name == "llama":
    model_args = model_utils.get_model_args(param_size, context_length, batch_size, tokenizer.vocab_size, bf16_enable)
    pt_model = model_exportable.Transformer(model_args)
    prefill_caches = model_utils.make_cache(model_args, 1)
    decode_caches = model_utils.make_cache(model_args, batch_size)
    sample_input_prefill =  model_utils.make_prefill_input(context_length, prefill_caches)
    sample_input_decode = model_utils.make_decode_input(context_length, decode_caches, batch_size)
    shard_weights_fn = model_utils.shard_weights

  return PyTorchEngine(
      devices=devices,
      imported_model=ImportedModel(
          param_size=param_size,
          context_length=context_length,
          ckpt_path=ckpt_path,
          batch_size=batch_size,
          bf16_enable=bf16_enable,
          num_of_partitions=len(devices),
          pt_model = pt_model,
          input_prefill = sample_input_prefill,
          input_decode = sample_input_decode,
          model_args = model_args,
          shard_weights_fn = shard_weights_fn,
      ),
      tokenizer=tokenizer,
      samples_per_slot=samples_per_slot,
      max_decode_length=max_decode_length,
  )
