"""Implement Jet Engine API."""

import copy
from typing import Any, List, Optional, Tuple, Union
import functools

from flax import struct
from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import mesh_utils
import torch
import jax.sharding as jsharding

from jetstream.engine import engine_api, tokenizer_pb2, token_utils
import torch_xla2
from .pets.llama2 import model_exportable, model_utils
from .pets import tokenizer

from torch.utils import _pytree as pytree



Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array
PrefillInputs = jax.Array

@struct.dataclass
class Prefix:
  token: jax.Array  # [1, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  seq_len: int

@struct.dataclass
class DecodeState:
  tokens: jax.Array   # [batch_size, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  current_position: int


# NOTE model specific


class PyTorchEngine(engine_api.Engine):
  """Wraps functions to the Jet Engine API format."""

  def __init__(
      self,
      devices: Union[List[Any], Mesh],
      pt_model: torch.nn.Module,
      tokenizer: Any,
      tokenizer_path: str,
      samples_per_slot: int,
      max_decode_length: int,
  ):
    self.devices = devices
    self.num_of_partitions = len(devices)
    self.pt_model = pt_model

    # NOTE: this is llama2 specific now.
    self.param = pt_model.params
    self.tokenizer = tokenizer
    self.tokenizer_path = tokenizer_path
    self.samples_per_slot_input = samples_per_slot
    self._max_decode_length = max_decode_length

    # TODO do we really need Named? seems we are not taking adv of the name at all
    P = jsharding.PartitionSpec
    self._mesh = jsharding.Mesh(
        mesh_utils.create_device_mesh((self.num_of_partitions, 1)),
        axis_names=("x", "y"),
    )
    self.y_sharding = jsharding.NamedSharding(self._mesh, P(None, "x"))
    self.x_sharding = jsharding.NamedSharding(self._mesh, P("x"))
    self.replicated = jsharding.NamedSharding(self._mesh, P())
    #self.cache_sharding = jsharding.NamedSharding(self._mesh, P("x", None, None, None))
    self.cache_sharding = jsharding.NamedSharding(self._mesh, P(None, None, "x", None))

    self.prefill = jax.jit(self.prefill, out_shardings=self.get_prefix_destination_sharding())
    self.insert = jax.jit(self.insert, donate_argnums=(0, 1, ), out_shardings=self.get_decode_state_sharding())
    self.generate = jax.jit(self.generate, donate_argnums=(1, ), out_shardings=(self.get_decode_state_sharding(), None))

  def sharding_by_name(self, name, reverse = False):
    if "tok_embeddings." in name:
        return self.x_sharding 
    if "attention." in name:
        if "wo" in name:
            return self.y_sharding if reverse else self.x_sharding
        else:
            return self.x_sharding if reverse else self.y_sharding
    if "feed_forward." in name:
        if "w2" in name:
            return self.y_sharding if reverse else self.x_sharding
        else:
            return self.x_sharding if reverse else self.y_sharding
    if "output" in name:
        return self.x_sharding if reverse else self.y_sharding 
    return self.replicated 

  def init_decode_state(
      self,
  ) -> DecodeState:
    caches = self._make_cache(self.param, self.param.max_batch_size)
    return DecodeState(
        jnp.zeros((self.param.max_batch_size, 1), dtype=jnp.int32),
        caches,
        0,
    )

  #@functools.partial(
  #  jax.jit,
  #  static_argnums=(0, 6),
  #  donate_argnums=(5, ),
  #)
  def _call_model(self, 
    weights, 
    tokens, 
    input_indexes, 
    cache_indexes, 
    caches, 
    prefill
  ):
    args = (
      tokens, input_indexes, cache_indexes, caches, prefill
    )
    paramst, argst = torch_xla2.tensor.wrap((weights, args))
    with torch_xla2.tensor.XLADispatchMode():
      res = torch.func.functional_call(self.pt_model, paramst, argst)
    return torch_xla2.tensor.unwrap(res)

  def _make_cache(self, args, batch_size):
    head_dim = args.dim // args.n_heads
    res = []
    kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
    for _ in range(args.n_layers):
      k_size = (batch_size, kv_heads, args.max_seq_len, head_dim)
      v_size = k_size
      res.append((
          jnp.zeros(
              k_size, dtype=jnp.bfloat16,
              device=self.cache_sharding
          ),
          jnp.zeros(
              v_size, dtype=jnp.bfloat16,
              device=self.cache_sharding
          ),
      ))
    return res

  def _sampling(self, logits: Any, batch_size: int) -> jnp.ndarray:
    return (
        jnp.argmax(logits[:, -1], axis=-1)
        .reshape(batch_size, -1)
        .astype(jnp.int32)
    )

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[jax.Array],
      true_length: int
  ) -> Prefix:
    if isinstance(padded_tokens, jax.Array):
      batched_token = padded_tokens.reshape(1, -1)
    else:
      raise TypeError(
          'Input tokens should be of type Jax Array, but receiving:'
          ' {prefill_inputs}'
      )
    seq_len = padded_tokens.shape[0]
    initial_caches = self._make_cache(self.pt_model.params, 1)
    input_indexes = jnp.arange(0, seq_len)
    cache_indexes = input_indexes
    logits, updated_caches = self._call_model(
      params, 
      batched_token, 
      input_indexes, 
      cache_indexes, 
      initial_caches, 
      True)
    token = self._sampling(logits, 1)
    token = token.squeeze(0) # drop batch
    return Prefix(token, updated_caches, seq_len) 

  def shrink_prefix(
      self,
      prefix: Prefix,
      new_length: int,
  ) -> Prefix:
    return prefix

  #@functools.partial(
  #  jax.jit,
  #  static_argnums=(0,),
  #  donate_argnums=(2,3),
  #)
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
    tokens = decode_state.tokens.at[slot].set(prefix.token)
    # TODO this doest wrap around
    pos = decode_state.current_position
    def insert(cache, new_entry):
        return jax.lax.dynamic_update_slice(
            cache,
            new_entry,
            [slot, pos, jnp.int32(0), jnp.int32(0)],
        )
    caches = [
      (insert(k, newk), insert(v, newv))
      for (k, v), (newk, newv) in zip(decode_state.caches, prefix.caches)
    ]
    position = prefix.seq_len
    return DecodeState(tokens, caches, position)

  #@functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2, ))
  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    #logging.info('Jet decode state before generate: %s', decode_state)
    #seq_len = padded_tokens.shape[0]
    #initial_caches = self._make_cache(self.pt_model.params, 1)
    pos = decode_state.current_position
    input_indexes = jnp.full((1,), pos) 
    cache_indexes = jnp.arange(0, 1024) + pos
    logits, new_caches = self._call_model(
      params, 
      decode_state.tokens, 
      input_indexes, 
      cache_indexes, 
      decode_state.caches, 
      False)
    next_token = self._sampling(logits, self.param.max_batch_size)
    #logging.info(
    #    'Jet generate next_token: %s, \ncaches: %s', next_token, caches_kv
    #)
    data = jnp.concatenate(
        [
            next_token,
            jnp.ones_like(next_token),
            jnp.ones_like(next_token),  # TODO gen len
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

    
    # logging.info(
    #     'Jet decode state after generate: %s \nresult token: %s',
    #     decode_state,
    #     result_tokens,
    # )
    new_decode_state = DecodeState(
      next_token, 
      new_caches,
      decode_state.current_position + 1
    )
    return new_decode_state, result_tokens


  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    return tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)

  def join_prefixes(
      self,
      prefix1: engine_api.Prefix,
      length1: int,
      prefix2: engine_api.Prefix,
      length2: int,
  ) -> tuple[engine_api.Prefix, int]:
    raise NotImplementedError('join_prefixes not supported')

  def _make_state_dict_jax(self, model_args_meta):
      def make_array(t):
          return jnp.ones(
              t.shape, dtype=torch_xla2.tensor.t2j_dtype(t.dtype))
      return pytree.tree_map_only(torch.Tensor, make_array, model_args_meta)

  def load_params(self) -> Params:
    # TODO load from files
    with jax.default_device(self.colocated_cpus()):
      jax_weights = self._make_state_dict_jax(self.pt_model.state_dict())
    jax_weights = {
      key: jax.device_put(value, self.sharding_by_name(key))
      for key, value in jax_weights.items()
    }
    jax.tree_map(lambda k, v: print(f'Name: {k}, shape: {v.shape}'), list(jax_weights.keys()), list(jax_weights.values()))
    return jax_weights

  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices('cpu')[0]

  def get_prefix_destination_sharding(self) -> Prefix:
    """Returns the shardings necessary to transfer data between engines."""
    return Prefix(
        self.replicated,
        self.cache_sharding,
        #self.replicated,
        self.replicated,
    )

  def get_decode_state_sharding(self) -> DecodeState:
    """Gets the shardings corresponding to the decode state."""
    return DecodeState(
        self.replicated,
        #self.x_sharding, # Sharding on batch dim of next token
        self.cache_sharding,
        self.replicated,
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
  def mesh(self):
    return self._mesh


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

  tokenizer = token_utils.load_vocab(tokenizer_path)
  pt_model = None
  shard_weights_fn = None
  if model_name == "llama":
    model_args = model_utils.get_model_args(param_size, context_length, batch_size, tokenizer.vocab_size, bf16_enable)
    model_args.device = 'meta'
    pt_model = model_exportable.Transformer(model_args)

  if bf16_enable:
    pt_model = pt_model.to(torch.bfloat16)

  return PyTorchEngine(
      devices=devices,
      pt_model=pt_model,
      tokenizer=tokenizer,
      samples_per_slot=samples_per_slot,
      max_decode_length=max_decode_length,
      tokenizer_path=tokenizer_path,
  )
