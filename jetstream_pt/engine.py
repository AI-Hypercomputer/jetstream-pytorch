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
import numpy as np

from jetstream.engine import engine_api, tokenizer_pb2, token_utils
import torch_xla2
from .pets.llama2 import model_exportable, model_utils
from .pets import tokenizer

from .pets import cache_manager
from .environment import JetEngineEnvironment, JetEngineEnvironmentData
from .pets import quantize

from torch.utils import _pytree as pytree
import torch



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
  cache_scales: List[Tuple[jax.Array, jax.Array]]  # only present in quantized kv
  current_position: int
  lens: jax.Array # [batch_size, 1]


# NOTE model specific


class PyTorchEngine(engine_api.Engine):
  """Wraps functions to the Jet Engine API format."""

  def __init__(
      self,
      pt_model: torch.nn.Module,
      env: JetEngineEnvironment,
  ):
    self.pt_model = pt_model
    self.env = env

    # NOTE: this is llama2 specific now.
    self.param = pt_model.params

    self.y_sharding = env.sharding_by_axis(1)
    self.x_sharding = env.sharding_by_axis(0)
    self.replicated = env.sharding_by_axis(-1) # replicated
    self.cache_sharding = self.y_sharding

    self.prefill = jax.jit(self.prefill, out_shardings=self.get_prefix_destination_sharding())
    # self.insert = jax.jit(self.insert, donate_argnums=(0, 1), out_shardings=self.get_decode_state_sharding())
    self.generate = jax.jit(self.generate, donate_argnums=(1, ), out_shardings=(self.get_decode_state_sharding(), None))
    self._insert_wrap = jax.jit(self._insert_wrap, donate_argnums=(0, 1, 2, 3, 4),
                                out_shardings=(self.cache_sharding, self.replicated)
                               )

    self._insert_no_wrap = jax.jit(
        self._insert_no_wrap, 
        donate_argnums=(0, 1), 
        out_shardings=self.get_decode_state_sharding())

  def sharding_by_name(self, name):

    # This allows easier way to edit shardings
    """
    for key, val in self.env._data.experimental_sharding_axis_override.items():
      if name.endswith(key): 
        return self.env.sharding_by_axis(val)
    """

    if 'weight_scaler' in name:
      return self.x_sharding
    if "tok_embeddings." in name:
        return self.y_sharding
    if "attention." in name:
        if "wo" in name:
            return self.y_sharding
        else:
            return self.x_sharding
    if "feed_forward." in name:
        if "w2" in name:
            return self.y_sharding
        else:
            return self.x_sharding
    if "output" in name:
        return self.x_sharding 
    return self.replicated 

  def init_decode_state(
      self,
  ) -> DecodeState:
    caches_obj = self.env.make_caches_generate()
    caches = [c.state() for c in caches_obj]
    scalers = []
    if self.env.enable_kv_quantization:
      scalers = [c.scalers() for c in caches_obj]
    return DecodeState(
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
        caches,
        scalers,
        self.env.max_input_sequence_length,
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
    )

  def _call_model_generate(
    self, 
    weights, 
    tokens, 
    input_indexes, 
    caches, 
    cache_scales,
  ):
    if self.env.enable_kv_quantization:
      caches_obj = [
        cache_manager.Int8KVCacheGenerate(
          k, v, ks, vs, input_indexes) for (k, v), (ks, vs) 
          in torch_xla2.tensor.wrap(list(zip(caches, cache_scales)))
      ]
    else:
      caches_obj = [
        cache_manager.KVCacheGenerate(k, v, input_indexes, self.cache_sharding) for k, v in torch_xla2.tensor.wrap(caches)
      ]
    args = (
      tokens, input_indexes, caches_obj, None
    )
    paramst, argst = torch_xla2.tensor.wrap((weights, args))
    with torch_xla2.tensor.XLADispatchMode():
      res = torch.func.functional_call(self.pt_model, paramst, argst)
    updated_caches = [c.state() for c in caches_obj]
    scales = []
    if self.env.enable_kv_quantization:
      scales = [c.scalers() for c in caches_obj]
    return torch_xla2.tensor.unwrap((res, updated_caches, scales))


  @functools.partial(
    jax.jit,
    static_argnums=(0, ),
  )
  def _call_model_prefill(self, weights, tokens, input_indexes):
    caches = [
      cache_manager.KVCachePrefill(self.env.enable_kv_quantization) for _ in self.pt_model.layers
    ]
    mask = jnp.full((1, 1, tokens.shape[1], tokens.shape[1]), 
                     float('-inf'), dtype=jnp.bfloat16)
    mask = jnp.triu(mask, k=1)
    args = (
      tokens, input_indexes, caches, mask
    )

    paramst, argst = torch_xla2.tensor.wrap((weights, args))
    with torch_xla2.tensor.XLADispatchMode():
      res = torch.func.functional_call(self.pt_model, paramst, argst)[0]
    caches_res = [c.state() for c in caches]
    return torch_xla2.tensor.unwrap((res, caches_res))

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
    input_indexes = jnp.arange(0, seq_len)
    logits, updated_caches = self._call_model_prefill(
      params, 
      batched_token, 
      input_indexes,
    )
    token = self._sampling(logits, 1)
    token = token.squeeze(0) # drop batch
    # truncate to true_length didnt work need to be out side of jit
    # caches = [
    #   (jax.lax.dynamic_slice_in_dim(
    #       k, seq_len - true_length, true_length, axis=2),
    #    jax.lax.dynamic_slice_in_dim(
    #       v, seq_len - true_length, true_length, axis=2))
    #   for k, v in updated_caches
    # ]
    return Prefix(token, updated_caches, seq_len) 

  def shrink_prefix(
      self,
      prefix: Prefix,
      new_length: int,
  ) -> Prefix:
    return prefix

  def _insert_no_wrap(
    self,
    prefix: Prefix,
    decode_state: DecodeState,
    slot: int,
  ):
    scales = []
    caches = []
    pos = decode_state.current_position
    tokens = decode_state.tokens.at[slot].set(prefix.token)
    if not self.env.enable_kv_quantization:
      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, new_entry):
          res = jax.lax.dynamic_update_slice(
              cache,
              new_entry,
              [slot, 0, pos, 0],
          )
          res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
          return res
      caches = [
        (insert(k, newk), insert(v, newv))
        for (k, v), (newk, newv) in zip(decode_state.caches, prefix.caches)
      ]
    else:
      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, scaler, new_entry):
          reduce_axis = (1, 3)
          vals, scales = torch_xla2.extra.call_torch(
            quantize.quantize_torch_int8, new_entry, reduce_axis)
          new_scaler = jax.lax.dynamic_update_slice(
              scaler,
              scales,
              [slot, 0, pos, 0],
          )
          new_scaler = jax.lax.with_sharding_constraint(new_scaler, self.replicated)
          res = jax.lax.dynamic_update_slice(
              cache,
              vals,
              [slot, 0, pos, 0],
          )
          res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
          return res, new_scaler

      for (k, v), (kscaler, vscaler), (newk, newv) in zip(
          decode_state.caches,
          decode_state.cache_scales,
          prefix.caches):
        kcache, kscale = insert(k, kscaler, newk)
        vcache, vscale = insert(v, vscaler, newv)
        caches.append((kcache, vcache))
        scales.append((kscale, vscale))

    lens = decode_state.lens.at[slot].set(1)
    return DecodeState(tokens, caches, scales, pos, lens)


  def _insert_wrap(
    self,
    old_caches,
    old_scales,
    cache_inserts,
    update_indexes,
    slot
  ):  # returns caches and scales
    scales = []
    caches = []
    if not self.env.enable_kv_quantization:
      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, new_entry):
          new_entry = jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
          res = cache.at[slot, :, update_indexes, :].set(new_entry)
          res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
          return res
      caches = [
        (insert(k, newk), insert(v, newv))
        for (k, v), (newk, newv) in zip(old_caches, cache_inserts)
      ]
    else:
      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, scaler, new_entry):
          new_entry = jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
          reduce_axis = (0, )
          vals, scales = torch_xla2.extra.call_torch(
            quantize.quantize_torch_int8, new_entry, reduce_axis)
          new_scaler = scaler.at[slot, head_indexes, update_indexes, :].set(scales)
          new_scaler = jax.lax.with_sharding_constraint(new_scaler, self.replicated)
          res = cache.at[slot, :, update_indexes, :].set(vals)
          res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
          return res, new_scaler

      caches = []
      for (k, v), (kscaler, vscaler), (newk, newv) in zip(
          old_caches,
          old_scales,
          cache_inserts):
        kcache, kscale = insert(k, kscaler, newk)
        vcache, vscale = insert(v, vscaler, newv)
        caches.append((kcache, vcache))
        scales.append((kscale, vscale))
    return caches, scales


  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    # logging.info(
    #     'Jet input prefix: %s, decode state before insert: %s',
    #     prefix,
    #     decode_state,
    # )
    if decode_state.current_position - prefix.seq_len >= 0:
      # Will not wrap around:
      return self._insert_no_wrap(prefix, decode_state, slot)
    else:
      # Wraps around
      tokens = decode_state.tokens.at[slot].set(prefix.token)

      pos = decode_state.current_position
      # Create indexes
      update_indexes = (jnp.arange(-prefix.seq_len, 0) + pos) % self.env.cache_sequence_length
      update_indexes = update_indexes.reshape(1, -1)

      caches, scales = self._insert_wrap(
        decode_state.caches, 
        decode_state.cache_scales,
        prefix.caches,
        update_indexes,
        slot
      )

      position = prefix.seq_len
      lens = decode_state.lens.at[slot].set(0)
      return DecodeState(tokens, caches, scales, position, lens)

  #@functools.partial(jax.jit, static_argnums=(0,), donate_argnums=(2, ))
  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    #logging.info('Jet decode state before generate: %s', decode_state)
    #seq_len = padded_tokens.shape[0]
    pos = decode_state.current_position
    input_indexes = jnp.full((1,), pos) 
    cache_indexes = jnp.arange(0, 1024) + pos
    logits, new_caches, new_scales = self._call_model_generate(
      params, 
      decode_state.tokens, 
      input_indexes, 
      decode_state.caches, 
      decode_state.cache_scales
    )
    next_token = self._sampling(logits, self.param.max_batch_size)
    lens = decode_state.lens + 1
    data = jnp.concatenate(
        [
            next_token,
            jnp.ones_like(next_token),
            lens,
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
    
    new_decode_state = DecodeState(
      next_token, 
      new_caches,
      new_scales,
      decode_state.current_position + 1,
      lens,
    )
    return new_decode_state, result_tokens


  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    return tokenizer_pb2.TokenizerParameters(path=self.env.tokenizer_path)

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
        res = jax.random.normal(
          jax.random.key(0), shape=t.shape, dtype=jnp.bfloat16)
        res = res.astype(torch_xla2.tensor.t2j_dtype(t.dtype))
        return res
      return pytree.tree_map_only(torch.Tensor, make_array, model_args_meta)

  def _load_from_safetensors(self, path):
    from safetensors import safe_open
    weights = {}
    with safe_open(path, framework='flax', device='cpu') as f:
      for key, model_weights in self.pt_model.state_dict().items():
        if key == 'freqs_cis': 
          continue
        arr = jax.device_put(f.get_tensor(key), self.sharding_by_name(key))
        assert tuple(model_weights.shape) == tuple(arr.shape), key
        weights[key] = arr
    freqs_cis = torch_xla2.tensor.t2j(self.pt_model.freqs_cis)
    weights['freqs_cis'] = jax.device_put(freqs_cis, self.replicated)
    return weights

  def load_params(self) -> Params:
    # TODO load from files
    with jax.default_device(self.colocated_cpus()):
      if self.env.checkpoint_path:
        if self.env.checkpoint_format == 'safetensors':
          return self._load_from_safetensors(self.env.checkpoint_path)
      else:
        jax_weights = self._make_state_dict_jax(self.pt_model.state_dict())
    jax_weights = {
      key: jax.device_put(value, self.sharding_by_name(key))
      for key, value in jax_weights.items()
    }
    for k, v in jax_weights.items():
      if k.startswith('layers') and not k.startswith('layers.0'):
        continue
      print(f'Name: {k}, shape: {v.shape} x {v.dtype}')
    return jax_weights

  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices('cpu')[0]

  def get_prefix_destination_sharding(self) -> Prefix:
    """Returns the shardings necessary to transfer data between engines."""
    return Prefix(
        self.replicated,
        self.cache_sharding,
        self.replicated,
    )

  def get_decode_state_sharding(self) -> DecodeState:
    """Gets the shardings corresponding to the decode state."""
    return DecodeState(
        self.replicated,
        self.cache_sharding,
        self.replicated,
        self.replicated,
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
    return 1
    # return self.samples_per_slot_input

  @property
  def max_prefill_length(self) -> int:
    return self.param.max_seq_len

  @property
  def max_decode_length(self) -> int:
    """Maximum decode length."""
    return self.env._data.max_decode_length

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
    model_name = "llama",
    quantize_weights = False,
    quantize_kv = False,
    max_cache_length = 1024,
) -> PyTorchEngine:
  """Returns: The pytorch engine."""

  # See issue b/309529778 if it's turned on.
  jax.config.update('jax_dynamic_shapes', False)
  # Pytorch exports has int64 constants.
  # jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_traceback_filtering', 'off')
  torch.set_default_dtype(torch.bfloat16)

  env_data = JetEngineEnvironmentData(
    tokenizer_path=tokenizer_path,
    checkpoint_path = ckpt_path, 
    checkpoint_format = 'safetensors',
    model_type = 'llama-2-' + param_size,
    batch_size = batch_size,
    max_decode_length = max_decode_length,
    max_input_sequence_length = context_length,
    enable_weight_quantization = quantize_weights,
    enable_kv_quantization = quantize_kv,
    cache_sequence_length = max_cache_length,
  )

  env = JetEngineEnvironment(env_data)

  tokenizer = token_utils.load_vocab(tokenizer_path)
  pt_model = None
  shard_weights_fn = None
  if model_name == "llama":
    model_args = model_utils.get_model_args(param_size, context_length, batch_size, tokenizer.vocab_size, bf16_enable)
    model_args.device = 'meta'
    model_args.quantize = quantize_weights
    pt_model = model_exportable.Transformer(model_args, env)

    num_params_size = 0
    num_params = 0
    for k, v in pt_model.state_dict().items():
      num_params += 1
      num_params_size += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    print('Number of param Gbytes:', num_params_size / (1 << 30))
    print('Number of param: ', num_params)

  return PyTorchEngine(
      pt_model=pt_model,
      env=env
  )
