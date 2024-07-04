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

from typing import Any, List, Optional, Tuple, Union
import threading
import functools
import os

import glob
from etils import epath
from flax import struct
import jax
from jax import numpy as jnp
from safetensors import safe_open
import torch
import numpy as np

from jetstream.engine import engine_api, tokenizer_api, tokenizer_pb2, token_utils
from jetstream.engine import sampling_utils
import torch_xla2
from torch.utils import _pytree as pytree

from jetstream_pt import cache_manager
from jetstream_pt import quantize
from jetstream_pt import torchjax
from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData, QuantizationConfig
from jetstream_pt.third_party.llama import model_exportable as llama_model, model_args
from jetstream_pt.third_party.gemma import config as gemma_config, model as gemma_model
from jetstream_pt.third_party.mixtral import config as mixtral_config, model as mixtral_model

from absl import flags

FLAGS = flags.FLAGS

Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array
PrefillInputs = jax.Array


@struct.dataclass
# pylint: disable-next=all
class Prefix:
  token: jax.Array  # [1, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  seq_len: int  # true seqlen front pad


@struct.dataclass
# pylint: disable-next=all
class DecodeState:
  tokens: jax.Array  # [batch_size, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  cache_scales: List[
      Tuple[jax.Array, jax.Array]
  ]  # only present in quantized kv
  current_position: int
  lens: jax.Array  # [batch_size, 1], the output token length
  start: jax.Array  # [batch_size, 1], the starting pos for each slot
  input_pos: jax.Array  # [batch_size, 1] input pos for each slot
  mask: jax.Array  # [batch_size, seqlen] -inf for invalid; 0 for valid


# NOTE model specific


# pylint: disable-next=all
class PyTorchEngine(engine_api.Engine):
  """Wraps functions to the Jet Engine API format."""

  def __init__(
      self,
      pt_model: torch.nn.Module,
      env: JetEngineEnvironment,
      weights=None,
  ):
    self.pt_model = pt_model
    self.env = env
    self.default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32
    self.rng = jax.random.PRNGKey(0)
    self.weights = weights

    self.y_sharding = env.sharding_by_axis(1)
    self.x_sharding = env.sharding_by_axis(0)
    self.replicated = env.sharding_by_axis(-1)  # replicated

    self.cache_sharding = self.env.cache_sharding

    jax.config.update("jax_enable_x64", False)

    self.prefill = jax.jit(
        self.prefill,
        out_shardings=(self.get_prefix_destination_sharding(), None),
    )
    self.insert = jax.jit(
        self.insert,
        donate_argnums=(0, 1),
        out_shardings=self.get_decode_state_sharding(),
    )
    self.generate = jax.jit(
        self.generate,
        donate_argnums=(1,),
        out_shardings=(self.get_decode_state_sharding(), None),
    )
    # self._insert_wrap = jax.jit(self._insert_wrap, donate_argnums=(0, 1),
    #                              out_shardings=self.get_decode_state_sharding())

    # self._insert_no_wrap = jax.jit(
    #      self._insert_no_wrap,
    #      donate_argnums=(0, 1),
    #      out_shardings=self.get_decode_state_sharding())
    self._lock = threading.RLock()

  def init_decode_state(
      self,
  ) -> DecodeState:
    caches_obj = self.env.make_caches_generate()
    caches = [c.state() for c in caches_obj]
    scalers = []
    if self.env.quant_config.enable_kv_quantization:
      scalers = [c.scalers() for c in caches_obj]
    return DecodeState(
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
        caches,
        scalers,
        self.env.starting_position,
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),  # lens
        jnp.zeros((self.env.batch_size,), dtype=jnp.int32),  # start pos
        jnp.zeros((self.env.batch_size,), dtype=jnp.int32),  # input pos
        jnp.full(
            (self.env.batch_size, self.env.cache_sequence_length),
            float("-inf"),
            dtype=self.default_dtype,
        ),
    )

  # pylint: disable-next=all
  def _call_model_generate(
      self,
      weights,
      tokens,
      input_indexes,
      caches,
      cache_scales,
      mask,
      start,
      input_pos,
      ragged_batch_index,
      ragged_block_index,
  ):
    if self.env.quant_config.enable_kv_quantization:
      caches_obj = [
          cache_manager.Int8KVCacheGenerate(
              k, v, ks, vs, input_indexes, env=self.env
          )
          for (k, v), (ks, vs) in torchjax.to_torch(
              list(zip(caches, cache_scales))
          )
      ]
    else:
      caches_obj = [
          cache_manager.KVCacheGenerate(
              k, v, input_indexes, self.cache_sharding, env=self.env
          )
          for k, v in torchjax.to_torch(caches)
      ]
    mask = jnp.expand_dims(mask, (1, 2))

    args = (
        tokens,
        input_pos,
        caches_obj,
        mask,
        start,
        ragged_batch_index,
        ragged_block_index,
    )
    paramst, argst = torchjax.to_torch((weights, args))
    with self._lock:
      with torch_xla2.default_env():
        # The mode is needed so that tensors created inside of
        # the model (such as via torch.ones etc) also have the right type
        res = torch.func.functional_call(self.pt_model, paramst, argst)
    updated_caches = []
    for c in caches_obj:
      c.finalize()
      updated_caches.append(c.state())
    scales = []
    if self.env.quant_config.enable_kv_quantization:
      scales = [c.scalers() for c in caches_obj]
    return torchjax.from_torch((res, updated_caches, scales))

  @functools.partial(
      jax.jit,
      static_argnums=(0,),
  )
  def _call_model_prefill(self, weights, tokens, input_indexes):
    caches = [
        cache_manager.KVCachePrefill(
            self.env.quant_config.enable_kv_quantization
        )
        for _ in self.pt_model.layers
    ]
    mask = jnp.full(
        (1, 1, tokens.shape[1], tokens.shape[1]),
        float("-inf"),
        dtype=self.default_dtype,
    )
    mask = jnp.triu(mask, k=1)
    args = (tokens, input_indexes, caches, mask)

    paramst, argst = torchjax.to_torch((weights, args))
    with self._lock:
      with torch_xla2.default_env():
        res = torch.func.functional_call(self.pt_model, paramst, argst)[0]
    caches_res = [c.state() for c in caches]
    return torchjax.from_torch((res, caches_res))

  def _sampling(self, logits: Any, batch_size: int) -> jnp.ndarray:
    if len(logits.shape) == 2:
      logits = jnp.expand_dims(logits, 0)
    return (
        sampling_utils.sampling(
            logits[:, -1],
            self.rng,
            self.env.sampling_algorithm,
            self.env.topk,
            self.env.nucleus_topp,
            self.env.temperature,
        )
        .reshape(batch_size, -1)
        .astype(jnp.int32)
    )

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[jax.Array],
      true_length: int,
  ) -> Tuple[Prefix, engine_api.ResultTokens]:
    if isinstance(padded_tokens, jax.Array):
      batched_token = padded_tokens.reshape(1, -1)
    else:
      raise TypeError(
          "Input tokens should be of type Jax Array, but receiving:"
          " {prefill_inputs}"
      )
    seq_len = padded_tokens.shape[0]
    input_indexes = jnp.arange(0, seq_len)
    logits, updated_caches = self._call_model_prefill(
        params,
        batched_token,
        input_indexes,
    )
    if len(logits.shape) == 3:  # b, seqlen, num words
      logits = logits[0]  # seqlen, num words
    token = sampling_utils.sampling(
        logits[true_length - 1],
        self.rng,
        self.env.sampling_algorithm,
        self.env.topk,
        self.env.nucleus_topp,
        self.env.temperature,
    )
    token_out = jnp.reshape(token, (1, 1))
    data = jnp.concatenate(
        [
            token_out,  # First token
            jnp.ones_like(token_out),  # validity of first token
            jnp.zeros((1, 1), dtype=jnp.int32),  # length = 0
        ],
        axis=-1,
    )
    length = token_out.shape[1]
    result = engine_api.ResultTokens(
        data=data,
        tokens_idx=(0, length),
        valid_idx=(length, 2 * length),
        length_idx=(2 * length, 2 * length + 1),
        samples_per_slot=1,
    )
    # truncate to true_length didnt work need to be out side of jit
    # caches = [
    #   (jax.lax.dynamic_slice_in_dim(
    #       k, seq_len - true_length, true_length, axis=2),
    #    jax.lax.dynamic_slice_in_dim(
    #       v, seq_len - true_length, true_length, axis=2))
    #   for k, v in updated_caches
    # ]
    return Prefix(token, updated_caches, true_length), result

  def shrink_prefix(
      self,
      prefix: Prefix,
      new_length: int,  # pylint: disable=unused-argument
  ) -> Prefix:
    """shrink prefix"""
    return prefix

  # pylint: disable-next=all
  def _insert_no_wrap(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ):
    scales = []
    caches = []
    if self.env.ring_buffer:
      current_pos = decode_state.current_position
    else:
      current_pos = prefix.seq_len

    pos = current_pos - prefix.seq_len
    tokens = decode_state.tokens.at[slot].set(prefix.token)

    x = jnp.arange(0, self.env.cache_sequence_length)
    cond = jnp.logical_and(x < current_pos, x >= pos)
    mask_insert = jnp.where(cond, 0, float("-inf"))
    mask = decode_state.mask.at[slot].set(mask_insert)
    start = decode_state.start.at[slot].set(
        pos % self.env.cache_sequence_length
    )
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)
    if not self.env.quant_config.enable_kv_quantization:

      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, new_entry, update_index):
        res = jax.lax.dynamic_update_slice(
            cache,
            new_entry,
            update_index,
        )
        res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
        return res

      if self.env.generate_cache_stacked:
        caches = decode_state.caches
        for idx, (newk, newv) in enumerate(prefix.caches):
          update_index = [idx, slot, 0, pos, 0]
          newk = jnp.expand_dims(newk, 0)
          newv = jnp.expand_dims(newv, 0)
          caches = [(insert(caches[0][0], newk, update_index),insert(caches[0][1], newv, update_index))]
      else:
        update_index = [slot, 0, pos, 0]
        caches = [
            (insert(k, newk, update_index), insert(v, newv, update_index))
            for (k, v), (newk, newv) in zip(decode_state.caches, prefix.caches)
        ]
    else:

      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, scaler, new_entry):
        reduce_axis = (1, 3)
        vals, scales, _ = torchjax.call_torch(
            quantize.quantize_tensor, new_entry, reduce_axis
        )
        new_scaler = jax.lax.dynamic_update_slice(
            scaler,
            scales,
            [slot, 0, pos, 0],
        )
        new_scaler = jax.lax.with_sharding_constraint(
            new_scaler, self.replicated
        )
        res = jax.lax.dynamic_update_slice(
            cache,
            vals,
            [slot, 0, pos, 0],
        )
        res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
        return res, new_scaler

      for (k, v), (kscaler, vscaler), (newk, newv) in zip(
          decode_state.caches, decode_state.cache_scales, prefix.caches
      ):
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
        start,
        input_pos,
        mask,
    )

  # pylint: disable-next=all
  def _insert_wrap(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ):  # returns Decode State

    start_insert = decode_state.current_position - prefix.seq_len
    tokens = decode_state.tokens.at[slot].set(prefix.token)
    start_insert = start_insert % self.env.cache_sequence_length
    # pos < 0
    update_indexes = (
        jnp.arange(0, prefix.caches[0][0].shape[2]) + start_insert
    ) % self.env.cache_sequence_length
    update_indexes = update_indexes.reshape(1, -1)

    x = jnp.arange(0, self.env.cache_sequence_length)
    cond = jax.lax.cond(
        decode_state.current_position > start_insert,
        lambda x, start_insert, current_position: jnp.logical_and(
            x >= start_insert, x < current_position
        ),
        lambda x, start_insert, current_position: jnp.logical_or(
            x >= start_insert, x < current_position
        ),
        x,
        start_insert,
        decode_state.current_position,
    )

    mask_insert = jnp.where(cond, 0, float("-inf"))
    mask = decode_state.mask.at[slot].set(mask_insert)
    start = decode_state.start.at[slot].set(start_insert)
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)

    old_caches = decode_state.caches
    old_scales = decode_state.cache_scales
    cache_inserts = prefix.caches

    # print(f"YY old_caches: {len(decode_state.caches)} cache_inserts: {len(cache_inserts)}")
    scales = []
    caches = []
    if not self.env.quant_config.enable_kv_quantization:

      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, new_entry, layer_id):
        new_entry = jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
        if self.env.generate_cache_stacked:
          res = cache.at[layer_id, slot, :, update_indexes, :].set(new_entry)
        else:
          res = cache.at[slot, :, update_indexes, :].set(new_entry)
        res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
        return res

      for idx, (newk, newv) in enumerate(prefix.caches):
        caches = [
          (insert(old_caches[0][0], newk, idx), insert(old_caches[0][1], newv, idx))
        ]
    else:

      @functools.partial(jax.jit, donate_argnums=(0, 1), inline=True)
      def insert(cache, scaler, new_entry):
        new_entry = jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
        reduce_axis = (1, 2)
        vals, scales, _ = torchjax.call_torch(
            quantize.quantize_tensor, new_entry, reduce_axis
        )
        new_scaler = scaler.at[slot, :, update_indexes, :].set(scales)
        new_scaler = jax.lax.with_sharding_constraint(
            new_scaler, self.replicated
        )
        res = cache.at[slot, :, update_indexes, :].set(vals)
        res = jax.lax.with_sharding_constraint(res, self.cache_sharding)
        return res, new_scaler

      caches = []
      for (k, v), (kscaler, vscaler), (newk, newv) in zip(
          old_caches, old_scales, cache_inserts
      ):
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
        start,
        input_pos,
        mask,
    )

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
    if self.env.ring_buffer:
      start_insert = decode_state.current_position - prefix.seq_len
      end_insert = start_insert + prefix.caches[0][0].shape[2]  # padded seclen
      return jax.lax.cond(
          jnp.logical_and(
              start_insert >= 0, end_insert < self.env.cache_sequence_length
          ),
          self._insert_no_wrap,
          self._insert_wrap,
          prefix,
          decode_state,
          slot,
      )
    # Left aligned, starts from 0, guaranteed no wrap
    else:
      return self._insert_no_wrap(prefix, decode_state, slot)

  def precompute_ragged_block_indices(self, decode_state: DecodeState):
    """Precompute the ragged attention block indices. Ragged attention iterates the grid
    and relies on the computed grid index to skip the unnecessary blocks. The basic idea
    is to use input_pos, which is the length of each slot to determine if we should
    work on the next block of the slot or move to the next slot."""
    start = decode_state.start
    end = (start + decode_state.input_pos) % self.env.cache_len
    batch_size = start.shape[0]
    bk = self.env.block_size
    # The batch index
    b = jnp.arange(batch_size).reshape((batch_size, 1))
    num_bk = self.env.cache_len // self.env.block_size
    # The block index
    i = jnp.arange(num_bk).reshape((1, num_bk))
    i = jnp.broadcast_to(i, (batch_size, num_bk))

    start = start.reshape((batch_size, 1))
    end = end.reshape((batch_size, 1))

    am_last_batch = b == batch_size - 1
    last_good_block = jnp.where(
        start < end,
        jax.lax.div(end - 1, bk),
        jax.lax.div(self.env.cache_len - 1, bk),
    )

    next_b = jnp.where(am_last_batch, b, b + 1)
    next_i = jnp.where(am_last_batch, last_good_block, 0)

    # start < end, continue work on the block is there is overlap with the [start, end)
    def true_comp(b, i, bk, start, end, next_b, next_i):
      b_next = jnp.where(i * bk >= end, next_b, b)
      i_next = jnp.where(i * bk >= end, next_i, i)
      i_next = jnp.where((i + 1) * bk <= start, jax.lax.div(start, bk), i_next)
      return b_next, i_next

    # start > end, continue work on the block is there is no overlap with [end, start)
    def false_comp(b, i, bk, start, end):
      b_next = b
      i_next = jnp.where(
          jnp.logical_and(i * bk >= end, (i + 1) * bk <= start),
          jax.lax.div(start, bk),
          i,
      )
      return b_next, i_next

    true_comp_b, true_comp_i = true_comp(b, i, bk, start, end, next_b, next_i)
    false_comp_b, false_comp_i = false_comp(b, i, bk, start, end)

    b_next = jnp.where(
        start < end, true_comp_b, jnp.where(start == end, next_b, false_comp_b)
    )
    i_next = jnp.where(
        start < end, true_comp_i, jnp.where(start == end, next_i, false_comp_i)
    )
    return b_next, i_next

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:
    # seq_len = padded_tokens.shape[0]
    pos = decode_state.current_position
    if self.env.ring_buffer:
      input_indexes = jnp.full((1,), pos)
    else:
      input_indexes = decode_state.input_pos

    ragged_batch_index, ragged_block_index = (
        self.precompute_ragged_block_indices(decode_state)
    )
    ragged_batch_index, ragged_block_index = ragged_batch_index.reshape(
        (-1)
    ), ragged_block_index.reshape((-1))


    def update_mask():
      if self.env.ring_buffer:
        return decode_state.mask.at[:, decode_state.current_position].set(0)

      batch = jnp.arange(self.env.batch_size)
      return decode_state.mask.at[batch, decode_state.input_pos].set(0)

    mask = decode_state.mask
    if not self.env.flash_attention:
      mask = update_mask(mask, decode_state.current_position)
    logits, new_caches, new_scales = self._call_model_generate(
        params,
        decode_state.tokens,
        input_indexes,
        decode_state.caches,
        decode_state.cache_scales,
        mask,
        decode_state.start,
        decode_state.input_pos,
        ragged_batch_index,
        ragged_block_index,
    )

    if self.env.flash_attention:
      # fill mask later, now use flash attention
      mask = update_mask()

    next_token = self._sampling(logits, self.env.batch_size)
    if self.env.ring_buffer:
      input_pos = decode_state.input_pos + 1
      lens = decode_state.lens + 1
    else:
      input_pos = jnp.where(
          decode_state.input_pos == 0,
          0,
          decode_state.input_pos + 1 % self.env.cache_len,
      )
      lens = jnp.where(
          decode_state.lens == 0, 0, decode_state.lens + 1 % self.env.cache_len
      )

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
        decode_state.start,
        input_pos,
        mask,
    )
    # print(
    #     "new_pos",
    #     (decode_state.current_position + 1) % self.env.cache_sequence_length,
    # )
    # print(f"new_token: {jnp.squeeze(next_token)}")
    return new_decode_state, result_tokens

  # pylint: disable-next=all
  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    # pylint: disable-next=all
    return tokenizer_pb2.TokenizerParameters(path=self.env.tokenizer_path)

  def build_tokenizer(
      self, metadata: tokenizer_pb2.TokenizerParameters  # pylint: disable=all
  ) -> tokenizer_api.Tokenizer:
    if "llama-3" in self.env.model_type:
      return token_utils.TikToken(metadata)

    return token_utils.SentencePieceTokenizer(metadata)

  def join_prefixes(
      self,
      prefix1: engine_api.Prefix,
      length1: int,
      prefix2: engine_api.Prefix,
      length2: int,
  ) -> tuple[engine_api.Prefix, int]:
    """join prefixes"""
    raise NotImplementedError("join_prefixes not supported")

  def _make_state_dict_jax(self, model_args_meta):
    def make_array(t):
      res = jax.random.normal(
          jax.random.key(0), shape=t.shape, dtype=self.default_dtype
      )
      res = res.astype(torch_xla2.tensor.t2j_dtype(t.dtype))
      return res

    return pytree.tree_map_only(torch.Tensor, make_array, model_args_meta)

  def _load_from_safetensors(self, path):
    weights = {}
    with safe_open(path, framework="flax", device="cpu") as f:
      for key, model_weights in self.pt_model.state_dict().items():
        if key == "freqs_cis":
          continue
        weights[key] = f.get_tensor(key)
        assert tuple(model_weights.shape) == tuple(
            weights[key].shape
        ), f"key: {key} error: {model_weights.shape} != {weights[key].shape}"
    weights["freqs_cis"] = torch_xla2.tensor.t2j(self.pt_model.freqs_cis)
    return weights

  def _load_from_state_dict(self, path):
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    weights = {}
    print(f"Loaded keys are : {state_dict.keys()}")
    for key, model_weights in self.pt_model.state_dict().items():
      if key == "freqs_cis":
        continue
      assert key in state_dict, f"key: {key} not found"
      weights[key] = torch_xla2.tensor.t2j(state_dict[key])
      assert tuple(model_weights.shape) == tuple(
          weights[key].shape
      ), f"key: {key} error: {model_weights.shape} != {weights[key].shape}"

    weights["freqs_cis"] = torch_xla2.tensor.t2j(self.pt_model.freqs_cis)
    return weights

  # pylint: disable-next=all
  def load_params(self) -> Params:
    if self.weights is not None:
      return self.weights
    # We want to fix this: load from files
    with jax.default_device(self.colocated_cpus):
      if self.env.checkpoint_path:
        if self.env.checkpoint_format == "safetensors":
          jax_weights = self._load_from_safetensors(self.env.checkpoint_path)
        elif self.env.checkpoint_format == "state_dict":
          jax_weights = self._load_from_state_dict(self.env.checkpoint_path)
      else:
        jax_weights = self._make_state_dict_jax(self.pt_model.state_dict())

      if self.env.quant_config.num_bits_weight == 4:
        assert (
            "gemma" not in self.env.model_type
        ), "int-4 is not supported in Gemma model yet."
        quantize_linear_weights_scaler_map = (
            self.pt_model.get_quantized_linear_weight_to_scaler_map()
        )
        with jax.default_device(jax.devices("cpu")[0]):
          for key, val in jax_weights.items():
            for qname in quantize_linear_weights_scaler_map.keys():
              if key.endswith(qname):
                val = val.astype(jnp.int4)
                jax_weights[key] = val

      jax_weights = {
          key: jax.device_put(value, self.env.sharding_by_name(key))
          for key, value in jax_weights.items()
      }
      for k, v in jax_weights.items():
        if k.startswith("layers") and not k.startswith("layers.0"):
          continue
        print(f"Name: {k}, shape: {v.shape} x {v.dtype}")
      return jax_weights

  @property
  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices("cpu")[0]

  def get_prefix_destination_sharding(self) -> Prefix:
    """Returns the shardings necessary to transfer data between engines."""
    return Prefix(
        self.replicated,
        self.replicated if self.env.shard_on_batch else self.cache_sharding,
        self.replicated,
    )

  def get_decode_state_sharding(self) -> DecodeState:
    """Gets the shardings corresponding to the decode state."""
    return DecodeState(
        self.x_sharding if self.env.shard_on_batch else self.replicated,
        self.cache_sharding,
        self.replicated,
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
    return self.env.batch_size

  @property
  def samples_per_slot(self) -> int:
    return 1
    # return self.samples_per_slot_input

  @property
  def max_prefill_length(self) -> int:
    return self.env.max_input_sequence_length

  @property
  def max_decode_length(self) -> int:
    """Maximum decode length."""
    # pylint: disable-next=all
    return self.env._data.max_decode_length

  @property
  def mesh(self):
    return self.mesh


# pylint: disable-next=all
def create_pytorch_engine(
    # pylint: disable-next=all
    devices: list[Any],
    tokenizer_path: str,
    ckpt_path: Optional[str] = None,
    samples_per_slot: int = 1,  # pylint: disable=unused-argument
    bf16_enable: bool = False,
    param_size: str = "7b",
    context_length: int = 1024,
    batch_size: int = 1,
    max_decode_length: int = 4096,
    model_name="llama-2",
    quant_config: QuantizationConfig = QuantizationConfig(),
    max_cache_length=1024,
    sharding_config=None,
    shard_on_batch=False,
    ragged_mha=False,
    starting_position=512,
    temperature=None,
    sampling_algorithm="greedy",
    nucleus_topp=None,
    topk=None,
    ring_buffer=True,
    flash_attention=False,
    generate_cache_stacked=False,
    new_cache_stacked=False,
    lazy_cache_update=False,
) -> PyTorchEngine:
  """Returns: The pytorch engine."""

  supported_models = ["llama-2", "llama-3", "gemma", "mixtral"]
  if model_name not in supported_models:
    raise NotImplementedError(
        f"Model name should be one of{','.join(supported_models)}"
    )
  # See issue b/309529778 if it's turned on.
  jax.config.update("jax_dynamic_shapes", False)
  # Pytorch exports has int64 constants.
  # jax.config.update('jax_enable_x64', True)
  jax.config.update("jax_traceback_filtering", "off")
  torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
  torch.set_default_dtype(torch_dtype)
  checkpoint_format = ""
  checkpoint_path = ""

  if not ckpt_path or ckpt_path is None:
    print("WARNING: Using random weights instead of checkpoints.")
  elif ".safetensors" in ckpt_path:
    checkpoint_format = "safetensors"
    checkpoint_path = ckpt_path
  elif ".pth" in ckpt_path or ".ckpt" in ckpt_path:
    checkpoint_format = "state_dict"
    checkpoint_path = ckpt_path
  else:
    path = epath.Path(ckpt_path) if ckpt_path and ckpt_path is not None else ""
    if not path.exists():
      raise ValueError(f"Checkpoint path {ckpt_path} not exists!")
    paths = list(path.glob("*.safetensors"))
    assert (
        len(paths) == 1
    ), f"Expects 1 *.safetensors in the checkpoint dir, see {len(paths)}"
    checkpoint_format = "safetensors"
    checkpoint_path = paths[0]

  pt_model = None

  sharding_file_name = ""
  if not sharding_config:
    if model_name.startswith("llama"):
      sharding_file_name = "llama"
    elif model_name.startswith("gemma"):
      sharding_file_name = "gemma"
    elif model_name.startswith("mixtral"):
      sharding_file_name = "mixtral"
    sharding_config = os.path.join(
        "default_shardings", sharding_file_name + ".yaml"
    )

  env_data = JetEngineEnvironmentData(
      tokenizer_path=tokenizer_path,
      checkpoint_path=checkpoint_path,
      checkpoint_format=checkpoint_format,
      batch_size=batch_size,
      max_decode_length=max_decode_length,
      max_input_sequence_length=context_length,
      quant_config=quant_config,
      cache_sequence_length=max_cache_length,
      bf16_enable=bf16_enable,
      sharding_config_path=sharding_config,
      shard_on_batch=shard_on_batch,
      ragged_mha=ragged_mha,
      starting_position=starting_position,
      temperature=temperature,
      sampling_algorithm=sampling_algorithm,
      nucleus_topp=nucleus_topp,
      topk=topk,
      ring_buffer=ring_buffer,
      flash_attention=flash_attention,
      generate_cache_stacked=generate_cache_stacked,
      new_cache_stacked=new_cache_stacked,
      lazy_cache_update=lazy_cache_update,
  )

  if shard_on_batch and sharding_config:
    print("WARNING: with sharding_on_batch sharding config is ignored.")

  if model_name.startswith("llama"):

    args = model_args.get_model_args(
        model_name + "-" + param_size, context_length, batch_size, bf16_enable
    )
    args.device = "meta"
    env_data.cache_shape = (
        batch_size,
        args.n_kv_heads,
        max_cache_length,
        args.dim // args.n_heads,
    )
    env_data.model_type = model_name + "-" + param_size
    env_data.num_layers = args.n_layers
    env = JetEngineEnvironment(env_data)
    pt_model = llama_model.Transformer(args, env)
  elif model_name == "gemma":
    args = gemma_config.get_model_config(param_size)
    env_data.cache_shape = (
        batch_size,
        args.num_key_value_heads,
        max_cache_length,
        args.head_dim,
    )
    env_data.model_type = model_name + "-" + param_size
    env_data.num_layers = args.num_hidden_layers
    env = JetEngineEnvironment(env_data)
    print(f"Enviroment variables: {vars(env)}")
    pt_model = gemma_model.GemmaModel(args, env)
  elif model_name == "mixtral":
    args = mixtral_config.ModelArgs.from_name("Mixtral-8x7B-v0.1")
    args.device = "meta"
    env_data.cache_shape = (
        batch_size,
        args.n_local_heads,
        max_cache_length,
        args.dim // args.n_head,
    )
    env_data.num_layers = args.n_layer
    env = JetEngineEnvironment(env_data)
    pt_model = mixtral_model.Transformer(args, env)
  else:
    raise RuntimeError(f"Model with name {model_name} not found")

  num_params_size = 0
  num_params = 0
  for _, v in pt_model.state_dict().items():
    num_params += 1
    num_params_size += np.prod(v.shape) * (1 if v.dtype == torch.int8 else 2)
  print("Number of param Gbytes:", num_params_size / (1 << 30))
  print("Number of param: ", num_params)

  return PyTorchEngine(pt_model=pt_model, env=env)
