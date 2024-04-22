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

"""Implement Jet Engine API."""

import copy
from typing import Any, List, Optional, Tuple, Union
import threading
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
from jetstream_pt.third_party.llama2 import model_exportable, model_args

from jetstream_pt import cache_manager
from jetstream_pt import quantize
from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData

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
  seq_len: int  # true seqlen front pad

@struct.dataclass
class DecodeState:
  tokens: jax.Array   # [batch_size, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  cache_scales: List[Tuple[jax.Array, jax.Array]]  # only present in quantized kv
  current_position: int
  lens: jax.Array # [batch_size, 1]
  input_pos: jax.Array # [batch_size, 1] input pos for each slot
  mask: jax.Array # [batch_size, seqlen] -inf for invalid; 0 for valid


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
    self.insert = jax.jit(self.insert, donate_argnums=(0, 1), out_shardings=self.get_decode_state_sharding())
    self.generate = jax.jit(self.generate, donate_argnums=(1, ), out_shardings=(self.get_decode_state_sharding(), None))
    # self._insert_wrap = jax.jit(self._insert_wrap, donate_argnums=(0, 1),
    #                              out_shardings=self.get_decode_state_sharding())

    # self._insert_no_wrap = jax.jit(
    #      self._insert_no_wrap, 
    #      donate_argnums=(0, 1), 
    #      out_shardings=self.get_decode_state_sharding())
    self._lock = threading.RLock()

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
        jnp.zeros((self.env.batch_size, ), dtype=jnp.int32),  # input pos 
        jnp.full((self.env.batch_size, self.env.cache_sequence_length), float('-inf'), 
                  dtype=jnp.bfloat16),
    )

  def _call_model_generate(
    self, 
    weights, 
    tokens, 
    input_indexes, 
    caches, 
    cache_scales,
    mask,
    input_pos,
  ):
    if self.env.enable_kv_quantization:
      caches_obj = [
        cache_manager.Int8KVCacheGenerate(
          k, v, ks, vs, input_indexes) for (k, v), (ks, vs) 
          in torch_xla2.tensor.wrap(list(zip(caches, cache_scales)))
      ]
    else:
      caches_obj = [
        cache_manager.KVCacheGenerate(
          k, v, input_indexes, self.cache_sharding) for k, v in torch_xla2.tensor.wrap(caches)
      ]
    mask = jnp.expand_dims(mask, (1, 2))
    
    args = (
      tokens, input_pos, caches_obj, mask 
    )
    paramst, argst = torch_xla2.tensor.wrap((weights, args))
    with self._lock:
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
    with self._lock:
      with torch_xla2.tensor.XLADispatchMode():
        res = torch.func.functional_call(self.pt_model, paramst, argst)[0]
    caches_res = [c.state() for c in caches]
    return torch_xla2.tensor.unwrap((res, caches_res))

  def _sampling(self, logits: Any, batch_size: int) -> jnp.ndarray:
    if len(logits.shape) == 2:
      logits = jnp.expand_dims(logits, 0)
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
    if len(logits.shape) == 3: # b, seqlen, num words
      logits = logits[0]

    token = jnp.argmax(logits[true_length-1])
    
    # truncate to true_length didnt work need to be out side of jit
    # caches = [
    #   (jax.lax.dynamic_slice_in_dim(
    #       k, seq_len - true_length, true_length, axis=2),
    #    jax.lax.dynamic_slice_in_dim(
    #       v, seq_len - true_length, true_length, axis=2))
    #   for k, v in updated_caches
    # ]
    return Prefix(token, updated_caches, true_length) 

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
    pos = decode_state.current_position - prefix.seq_len
    tokens = decode_state.tokens.at[slot].set(prefix.token)

    x = jnp.arange(0, self.env.cache_sequence_length)
    cond = jnp.logical_and(x <= decode_state.current_position, x >= pos)
    mask_insert = jnp.where(cond, 0, float('-inf'))
    mask = decode_state.mask.at[slot].set(mask_insert)
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)
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
    return DecodeState(
      tokens, 
      caches, 
      scales, 
      decode_state.current_position, 
      lens, 
      input_pos,
      mask)


  def _insert_wrap(
    self,
    prefix: Prefix,
    decode_state: DecodeState,
    slot: int,
  ):  # returns Decode State

    start_insert = decode_state.current_position - prefix.seq_len
    end_insert = start_insert + prefix.caches[0][0].shape[2] # padded seclen
    tokens = decode_state.tokens.at[slot].set(prefix.token)

    start_insert = start_insert % self.env.cache_sequence_length
    # pos < 0
    update_indexes = (jnp.arange(0, prefix.caches[0][0].shape[2]) + start_insert) % self.env.cache_sequence_length
    update_indexes = update_indexes.reshape(1, -1)

    x = jnp.arange(0, self.env.cache_sequence_length)
    cond = jax.lax.cond(decode_state.current_position > start_insert, 
      lambda x, start_insert, current_position: jnp.logical_and(x >= start_insert, x <= current_position),
      lambda x, start_insert, current_position: jnp.logical_or(x >= start_insert, x <= current_position),
      x, start_insert, decode_state.current_position
    )

    mask_insert = jnp.where(cond, 0, float('-inf'))
    mask = decode_state.mask.at[slot].set(mask_insert)
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)

    old_caches = decode_state.caches
    old_scales = decode_state.cache_scales
    cache_inserts = prefix.caches

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
          reduce_axis = (1, 2)
          vals, scales = torch_xla2.extra.call_torch(
            quantize.quantize_torch_int8, new_entry, reduce_axis)
          new_scaler = scaler.at[slot, :, update_indexes, :].set(scales)
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

    lens = decode_state.lens.at[slot].set(1)
    return DecodeState(
      tokens, 
      caches, 
      scales, 
      decode_state.current_position, 
      lens,
      input_pos,
      mask)

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
    start_insert = decode_state.current_position - prefix.seq_len
    end_insert = start_insert + prefix.caches[0][0].shape[2] # padded seclen
    return jax.lax.cond(
        jnp.logical_and(start_insert >= 0, end_insert < self.env.cache_sequence_length),
        self._insert_no_wrap,
        self._insert_wrap,
        prefix, decode_state, slot)

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    #seq_len = padded_tokens.shape[0]
    pos = decode_state.current_position
    input_indexes = jnp.full((1,), pos) 

    # fill mask first
    mask = decode_state.mask.at[:, decode_state.current_position].set(0)
    logits, new_caches, new_scales = self._call_model_generate(
      params, 
      decode_state.tokens, 
      input_indexes, 
      decode_state.caches, 
      decode_state.cache_scales,
      mask,
      decode_state.input_pos,
    )
    next_token = self._sampling(logits, self.param.max_batch_size)
    lens = decode_state.lens + 1
    data = jnp.concatenate(
        [
            decode_state.tokens,
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
      (decode_state.current_position + 1) % self.env.cache_sequence_length,
      lens,
      decode_state.input_pos + 1,
      mask,
    )
    print('new_pos', (decode_state.current_position + 1) % self.env.cache_sequence_length)
    print('cache_seq_len', self.env.cache_sequence_length)

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
        assert tuple(model_weights.shape) == tuple(arr.shape), f"key: {key} error: {model_weights.shape} != {arr.shape}"
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

  checkpoint_format = ''
  checkpoint_path = ''

  if not ckpt_path or ckpt_path == None:
      print('WARNING: Using random weights instead of checkpoints.')
  elif ".safetensors" in ckpt_path:
      checkpoint_format = 'safetensors'
      checkpoint_path = ckpt_path
  elif ".pth" in ckpt_path:
      raise NotImplementedError('Loading from Pytorch raw checkpoint is not supported!')
  else:
      from etils import epath
      path = epath.Path(ckpt_path) if ckpt_path and ckpt_path != None else ''
      if not path.exists():
          raise ValueError(f'Checkpoint path {ckpt_path} not exists!')
      paths = list(path.glob('*.safetensors'))
      assert len(paths) == 1, f'Expects 1 *.safetensors in the checkpoint dir, see {len(paths)}'
      checkpoint_format = 'safetensors'
      checkpoint_path = paths[0]
  env_data = JetEngineEnvironmentData(
    tokenizer_path=tokenizer_path,
    checkpoint_path = checkpoint_path,
    checkpoint_format = checkpoint_format,
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
    args = model_args.get_model_args(param_size, context_length, batch_size, tokenizer.vocab_size, bf16_enable)
    args.device = 'meta'
    args.quantize = quantize_weights
    pt_model = model_exportable.Transformer(args, env)

    num_params_size = 0
    num_params = 0
    for k, v in pt_model.state_dict().items():
      num_params += 1
      num_params_size += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    print('Number of param Gbytes:', num_params_size / (1 << 30))
    print('Number of param: ', num_params)
  elif model_name == "gemma":
    args = model_args.get_model_args(param_size, context_length, batch_size, tokenizer.vocab_size, bf16_enable)
    args.device = 'google'
    args.quantize = quantize_weights
    pt_model = model_exportable.Transformer(args, env)

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
