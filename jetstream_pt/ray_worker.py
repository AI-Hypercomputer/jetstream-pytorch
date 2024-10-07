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
import functools
import os
import queue
import threading
from typing import Any, List, Optional, Tuple, Union

import humanize
import jax
import numpy as np
import ray
from ray.util.accelerators import tpu
import safetensors
import torch
import torch_xla2
from etils import epath
from flax import struct
from jax import numpy as jnp
from jax.experimental import multihost_utils
from torch.utils import _pytree as pytree
from jetstream.engine import engine_api, tokenizer_pb2
from jetstream_pt import cache_manager, quantize, torchjax
from jetstream_pt.environment import (
    JetEngineEnvironment,
    JetEngineEnvironmentData,
    QuantizationConfig,
)
from jetstream_pt.third_party.gemma import config as gemma_config
from jetstream_pt.third_party.gemma import model as gemma_model
from jetstream_pt.third_party.llama import model_args, model_exportable

Mesh = jax.sharding.Mesh
P = jax.sharding.PartitionSpec

Params = jax.Array
PrefillInputs = np.ndarray


@struct.dataclass
# pylint: disable-next=all
class Prefix:
  token: jax.Array  # [1, seqlen]
  caches: List[Tuple[jax.Array, jax.Array]]
  seq_len: int  # true seqlen front pad


@struct.dataclass
# pylint: disable-next=all
class NpPrefix:
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
  lens: jax.Array  # [batch_size, 1]
  input_pos: jax.Array  # [batch_size, 1] input pos for each slot
  mask: jax.Array  # [batch_size, seqlen] -inf for invalid; 0 for valid


# NOTE model specific


@ray.remote
# pylint: disable-next=all
class PyTorchRayWorker:
  """Ray actor representation for a PyTorch engine worker.

  PyTorchRayWorker enables multi-host serving for models that exceed the memory
  capabilities of single-host TPU VMs with Ray. PyTorchRayWorker should be used
  with a "leader" engine, e.g. `PyTorchRayEngine`.

  Note: For `PyTorchRayEngine` to return consistent results, it's important that
  `PyTorchEngineRayWorker` is able to maintain its state within process and only
  return results once its transferred to CPU device.

  """

  # pylint: disable-next=all
  def __init__(
      self,
      tokenizer_path: str,
      ckpt_path: Optional[str] = None,
      samples_per_slot: int = 1,
      bf16_enable: bool = False,
      param_size: str = "7b",
      context_length: int = 1024,
      batch_size: int = 1,
      max_decode_length: int = 4096,
      model_name="llama-2",
      quantize_weights=False,
      quantize_kv=False,
      max_cache_length=1024,
      sharding_config=None,
      enable_jax_profiler: bool = False,
      jax_profiler_port: int = 9999,
  ):

    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    jax.config.update("jax_dynamic_shapes", False)
    # Pytorch exports has int64 constants.
    # jax.config.update('jax_enable_x64', True)
    jax.config.update("jax_traceback_filtering", "off")
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    self.devices = jax.devices()
    device_count = jax.device_count()
    local_device_count = jax.local_device_count()
    print(
        f"---Jax device_count:{device_count}, local_device_count{local_device_count} "
    )

    if enable_jax_profiler:
      jax.profiler.start_server(jax_profiler_port)
      print(f"Started JAX profiler server on port {jax_profiler_port}")

    checkpoint_format = ""
    checkpoint_path = ""

    if not ckpt_path or ckpt_path is None:
      print("WARNING: Using random weights instead of checkpoints.")
    elif ".safetensors" in ckpt_path:
      checkpoint_format = "safetensors"
      checkpoint_path = ckpt_path
    elif ".pth" in ckpt_path:
      raise NotImplementedError(
          "Loading from Pytorch raw checkpoint is not supported!"
      )
    else:
      path = (
          epath.Path(ckpt_path) if ckpt_path and ckpt_path is not None else ""
      )
      if not path.exists():
        raise ValueError(f"Checkpoint path {ckpt_path} not exists!")
      paths = list(path.glob("*.safetensors"))
      assert (
          len(paths) == 1
      ), f"Expects 1 *.safetensors in the checkpoint dir, see {len(paths)}"
      checkpoint_format = "safetensors"
      checkpoint_path = paths[0]

    if not sharding_config:
      sharding_config = os.path.join("default_shardings", model_name + ".yaml")

    quant_config = QuantizationConfig(
        enable_weight_quantization=quantize_weights,
        enable_kv_quantization=quantize_kv,
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
    )

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
      env_data.model_type = "llama-2-" + param_size
      env_data.num_layers = args.n_layers
      env = JetEngineEnvironment(env_data)
      pt_model = model_exportable.Transformer(args, env)
    elif model_name == "gemma":
      args = gemma_config.get_model_config(param_size)
      env_data.cache_shape = (
          batch_size,
          args.num_key_value_heads,
          max_cache_length,
          args.head_dim,
      )
      env_data.model_type = "gemma-" + param_size
      env_data.num_layers = args.num_hidden_layers
      env = JetEngineEnvironment(env_data)
      pt_model = gemma_model.GemmaModel(args, env)
    else:
      raise RuntimeError(f"Model with name {model_name} not found")

    num_params_size = 0
    num_params = 0
    for _, v in pt_model.state_dict().items():
      num_params += 1
      num_params_size += np.prod(v.shape) * (1 if v.dtype == jnp.int8 else 2)
    print("Number of param Gbytes:", num_params_size / (1 << 30))
    print("Number of param: ", num_params)

    self.decode_state = None
    self.prefix_queue = queue.Queue()
    self.pt_model = pt_model
    self.env = env
    self.default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32

    self.y_sharding = env.sharding_by_axis(1)
    self.x_sharding = env.sharding_by_axis(0)
    self.replicated = env.sharding_by_axis(-1)  # replicated
    self.cache_sharding = self.y_sharding

    self._compiled_call_model_prefill = jax.jit(
        self._call_model_prefill,
        donate_argnums=(1, 2),
        out_shardings=(self.replicated, self.cache_sharding),
    )
    self._compiled_insert = jax.jit(
        self._insert,
        donate_argnums=(0, 1),
        out_shardings=(
            self.replicated,
            self.cache_sharding,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
        ),
    )

    self._compiled_call_model_generate = jax.jit(
        self._call_model_generate,
        donate_argnums=(2, 3, 4, 5, 6, 7),
        out_shardings=(
            self.replicated,
            self.cache_sharding,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
            self.replicated,
        ),
    )
    self._lock = threading.RLock()

  # pylint: disable-next=all
  def sharding_by_name(self, name):

    if "weight_scaler" in name:
      return self.x_sharding
    if "tok_embeddings." in name:
      return self.y_sharding
    if "attention." in name:
      if "wo" in name:
        return self.y_sharding
      return self.x_sharding
    if "feed_forward." in name:
      if "w2" in name:
        return self.y_sharding
      return self.x_sharding
    if "output" in name:
      return self.x_sharding
    return self.replicated

  # pylint: disable-next=all
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
        self.env.max_input_sequence_length,
        jnp.zeros((self.env.batch_size, 1), dtype=jnp.int32),
        jnp.zeros((self.env.batch_size,), dtype=jnp.int32),  # input pos
        jnp.full(
            (self.env.batch_size, self.env.cache_sequence_length),
            float("-inf"),
            dtype=self.default_dtype,
        ),
    )

  def print_mem_usage(self):
    """Print current mem usage"""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)

    for d in jax.local_devices():
      stats = d.memory_stats()
      used = stats["bytes_in_use"]
      limit = stats["bytes_limit"]
      print(
          f"memory using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}"
      )

  def init_decode_state_ray(
      self,
  ) -> None:
    """Init decode state in ray worker"""
    self.decode_state = self.init_decode_state()

  # pylint: disable-next=all
  def _call_model_generate(
      self,
      weights,
      tokens,
      caches,
      cache_scales,
      mask,
      current_position,
      input_pos,
      lens,
  ):
    pos = current_position
    input_indexes = jnp.full((1,), pos)
    new_mask = mask.at[:, current_position].set(0)
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
    mask = jnp.expand_dims(new_mask, (1, 2))

    args = (tokens, input_pos, caches_obj, mask)
    paramst, argst = torchjax.to_torch((weights, args))
    with self._lock:
      with torch_xla2.default_env():
        res = torch.func.functional_call(self.pt_model, paramst, argst)
      updated_caches = [c.state() for c in caches_obj]
    scales = []
    if self.env.quant_config.enable_kv_quantization:
      scales = [c.scalers() for c in caches_obj]
    new_current_position = (
        current_position + 1
    ) % self.env.cache_sequence_length
    return torchjax.from_torch(
        (
            res,
            updated_caches,
            scales,
            input_pos + 1,
            lens + 1,
            new_current_position,
            new_mask,
        )
    )

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

  def _sampling(self, logits: Any, batch_size: int) -> np.ndarray:
    if len(logits.shape) == 2:
      logits = np.expand_dims(logits, 0)
    return (
        np.argmax(logits[:, -1], axis=-1)
        .reshape(batch_size, -1)
        .astype(np.int32)
    )

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> Any:
    """Do prefill"""
    padded_tokens = jnp.asarray(padded_tokens)
    if isinstance(padded_tokens, jax.Array):
      batched_token = padded_tokens.reshape(1, -1)
    else:
      raise TypeError(
          "Input tokens should be of type Jax Array, but receiving:"
          " {padded_tokens}"
      )
    seq_len = padded_tokens.shape[0]
    input_indexes = jnp.arange(0, seq_len)
    logits, updated_caches = self._compiled_call_model_prefill(
        self.params,
        batched_token,
        input_indexes,
    )

    logits = multihost_utils.process_allgather(logits, tiled=True)
    return logits, updated_caches

  def prefill_ray(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> tuple[Prefix, engine_api.ResultTokens]:
    """Do prefill in ray worker"""
    logits, updated_caches = self.prefill(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    if len(logits.shape) == 3:  # b, seqlen, num words
      logits = logits[0]

    token = np.argmax(logits[true_length - 1])
    prefix = Prefix(token, updated_caches, true_length)
    self.prefix_queue.put(prefix, block=False)

    token_out = np.reshape(token, (1, 1))
    data = np.concatenate(
        [
            token_out,  # First token
            np.ones_like(token_out),  # validity of first token
            np.zeros((1, 1), dtype=np.int32),  # length = 0
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

    return None, result

  def _convert_to_np_caches(
      self, caches: List[Tuple[jax.Array, jax.Array]]
  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(np.asarray(tup[0]), np.asarray(tup[1])) for tup in caches]

  def _convert_to_jax_caches(
      self, np_caches: List[Tuple[np.ndarray, np.ndarray]]
  ) -> List[Tuple[jax.Array, jax.Array]]:
    return [(jnp.asarray(tup[0]), jnp.asarray(tup[1])) for tup in np_caches]

  def prefill_ray_disaggregation(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: PrefillInputs,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> tuple[NpPrefix, engine_api.ResultTokens]:
    """Do prefill in ray worker"""
    logits, updated_caches = self.prefill(
        params=params,
        existing_prefix=existing_prefix,
        padded_tokens=padded_tokens,
        true_length=true_length,
    )
    if len(logits.shape) == 3:  # b, seqlen, num words
      logits = logits[0]

    token = np.argmax(logits[true_length - 1])
    updated_caches = multihost_utils.process_allgather(
        updated_caches, tiled=True
    )
    np_update_caches = self._convert_to_np_caches(updated_caches)
    np_prefix = NpPrefix(token, np_update_caches, true_length)

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

    return np_prefix, result

  def transfer(self, np_prefix: NpPrefix) -> Any:
    """Transfer prefill result from object store to HBM"""
    updated_caches = self._convert_to_jax_caches(np_prefix.caches)
    prefix = Prefix(np_prefix.token, updated_caches, np_prefix.seq_len)
    self.prefix_queue.put(prefix, block=False)
    return True

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
    pos = decode_state.current_position - prefix.seq_len
    tokens = decode_state.tokens.at[slot].set(prefix.token)

    x = jnp.arange(0, self.env.cache_sequence_length)
    cond = jnp.logical_and(x <= decode_state.current_position, x >= pos)
    mask_insert = jnp.where(cond, 0, float("-inf"))
    mask = decode_state.mask.at[slot].set(mask_insert)
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)
    if not self.env.quant_config.enable_kv_quantization:

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
        reduce_axis = (-3, -1)
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
    return (
        tokens,
        caches,
        scales,
        decode_state.current_position,
        lens,
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
            x >= start_insert, x <= current_position
        ),
        lambda x, start_insert, current_position: jnp.logical_or(
            x >= start_insert, x <= current_position
        ),
        x,
        start_insert,
        decode_state.current_position,
    )

    mask_insert = jnp.where(cond, 0, float("-inf"))
    mask = decode_state.mask.at[slot].set(mask_insert)
    input_pos = decode_state.input_pos.at[slot].set(prefix.seq_len)

    old_caches = decode_state.caches
    old_scales = decode_state.cache_scales
    cache_inserts = prefix.caches

    scales = []
    caches = []
    if not self.env.quant_config.enable_kv_quantization:

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
    return (
        tokens,
        caches,
        scales,
        decode_state.current_position,
        lens,
        input_pos,
        mask,
    )

  def _insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ):
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

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """insert prefix to decode state"""
    tokens, caches, scales, current_position, lens, input_pos, mask = (
        self._compiled_insert(prefix, decode_state, slot)
    )
    return DecodeState(
        tokens, caches, scales, current_position, lens, input_pos, mask
    )

  def insert_ray(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """insert prefix to decode state in ray worker"""
    prefix = self.prefix_queue.get_nowait()
    self.decode_state = self.insert(prefix, self.decode_state, slot)

  # pylint: disable-next=all
  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[DecodeState, engine_api.ResultTokens]:

    # pylint: disable-next=all
    (
        logits,
        new_caches,
        new_scales,
        new_input_pos,
        new_lens,
        new_current_position,
        new_mask,
    ) = self._compiled_call_model_generate(
        self.params,
        decode_state.tokens,
        decode_state.caches,
        decode_state.cache_scales,
        decode_state.mask,
        decode_state.current_position,
        decode_state.input_pos,
        decode_state.lens,
    )

    logits = multihost_utils.process_allgather(logits, tiled=True)
    next_token = self._sampling(logits, self.env.batch_size)

    data = np.concatenate(
        [
            decode_state.tokens,
            np.ones_like(next_token),
            new_lens,
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
    next_token = jax.lax.with_sharding_constraint(next_token, self.replicated)
    new_decode_state = DecodeState(
        next_token,
        new_caches,
        new_scales,
        new_current_position,
        new_lens,
        new_input_pos,
        new_mask,
    )

    return new_decode_state, result_tokens

  # pylint: disable-next=all
  def generate_ray(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[None, engine_api.ResultTokens]:
    decode_state, result_tokens = self.generate(self.params, self.decode_state)
    self.decode_state = decode_state
    return None, result_tokens

  # pylint: disable-next=all
  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    """get tokenizer"""
    # pylint: disable-next=all
    return tokenizer_pb2.TokenizerParameters(path=self.env.tokenizer_path)

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

  def _weight_sharding(self, weight, sharding):
    return jax.make_array_from_callback(
        weight.shape, sharding, lambda idx: weight[idx]
    )

  def _load_from_safetensors(self, path):

    weights = {}
    # pylint: disable-next=all
    with safetensors.safe_open(path, framework="flax", device="cpu") as f:
      for key, model_weights in self.pt_model.state_dict().items():
        if key == "freqs_cis":
          continue
        tensor = f.get_tensor(key)
        arr = self._weight_sharding(tensor, self.sharding_by_name(key))

        assert tuple(model_weights.shape) == tuple(
            arr.shape
        ), f"key: {key} error: {model_weights.shape} != {arr.shape}"
        weights[key] = arr

    freqs_cis = torch_xla2.tensor.t2j(self.pt_model.freqs_cis)
    weights["freqs_cis"] = self._weight_sharding(freqs_cis, self.replicated)

    for k, v in weights.items():
      if k.startswith("layers") and not k.startswith("layers.0"):
        continue
      print(f"Name: {k}, shape: {v.shape} x {v.dtype}")

    return weights

  # pylint: disable-next=all
  def load_params(self) -> Params:
    # We want to fix this: load from files
    with jax.default_device(self.colocated_cpus()):
      if self.env.checkpoint_path:
        if self.env.checkpoint_format == "safetensors":
          return self._load_from_safetensors(self.env.checkpoint_path)
      else:
        jax_weights = self._make_state_dict_jax(self.pt_model.state_dict())
    jax_weights = {
        key: jax.device_put(value, self.sharding_by_name(key))
        for key, value in jax_weights.items()
    }
    for k, v in jax_weights.items():
      if k.startswith("layers") and not k.startswith("layers.0"):
        continue
      print(f"Name: {k}, shape: {v.shape} x {v.dtype}")
    return jax_weights

  def load_params_ray(self):
    """load params in ray worker"""
    print("--- mem_usage before load params")
    self.print_mem_usage()
    self.params = self.load_params()  # pylint: disable=attribute-defined-outside-init
    print("--- mem_usage after load params")
    self.print_mem_usage()

  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    """cpu device"""
    return jax.devices("cpu")[0]

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
    """Max batch size for decodes"""
    return self.env.batch_size

  @property
  def samples_per_slot(self) -> int:
    """Samples per slot"""
    return 1

  @property
  def max_prefill_length(self) -> int:
    """Maximum prefill length"""
    return self.env.max_input_sequence_length

  @property
  def max_decode_length(self) -> int:
    """Maximum decode length"""
    # pylint: disable-next=all
    return self.env._data.max_decode_length

  @property
  def mesh(self):
    """return mesh"""
    return None

  def pod_slice_name(self):
    """pod slice name"""
    return tpu.get_current_pod_name()
