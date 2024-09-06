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

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
import torch
import torch_xla2

from jetstream_pt import torchjax
from jetstream_pt.page_attention_manager import PageAttentionManager


# pylint: disable-next=all
class CacheInterface:
  """Kv cache interface"""

  # cache for ONE layer

  def update(self, key, value):
    """Update the cache for this key and value.

    The key, and val will have shape (Batch, Heads, Seqlen, Head dim)
    The cache is free to store them in a different format.
    Return the full cache after update.
    This cache instance need to know which position / layer is
    the update for.
    """


class KVCachePrefill:
  """Prefill kv cache"""

  def __init__(self, kv_quantize=False, stacked=False):
    self.kv_quantize = kv_quantize
    self.cache_k = None
    self.cache_v = None
    self.stacked = stacked

  def update(self, key, value, layer_id):
    """This cache just remembers the stuff."""
    self.cache_k = key
    self.cache_v = value
    if self.kv_quantize:  # pretend to be quantized
      bsz, _, seq, _ = key.shape
      ones = torchjax.to_torch(jnp.ones((bsz, 1, seq, 1), dtype=jnp.bfloat16))
      return key, value, None, None, ones, ones, None, None

    return key, value

  def state(self):
    """Get prefill cache state"""
    return self.cache_k, self.cache_v

  # Placeholder, to match with GenerateCache
  def finalize(self):
    """Finalize the cache operation and updates the cache."""
    return


# pylint: disable-next=all
def KVCachePrefill_flatten(cache):
  return (
      torchjax.from_torch((cache.cache_k, cache.cache_v)),
      cache.kv_quantize,
  )


# pylint: disable-next=all
def KVCachePrefill_unflatten(auxdata, data):
  cache = KVCachePrefill(auxdata)
  cache_k, cache_v = torchjax.from_torch(data)
  cache.cache_k = cache_k
  cache.cache_v = cache_v


jax.tree_util.register_pytree_node(
    KVCachePrefill, KVCachePrefill_flatten, KVCachePrefill_unflatten
)


class KVCacheGenerate:
  """Kvache generator without quantization"""

  # pylint: disable=too-many-instance-attributes
  # More than 7 is reasonable in this case.
  def __init__(
      self,
      cache_k: torch.Tensor,  # previous cache
      cache_v: torch.Tensor,  # previous cache
      position: int | torch.Tensor,  # position to store the cache
      sharding,
      env=None,
  ):
    super().__init__()
    self.cache_k = cache_k
    self.cache_v = cache_v
    self.input_pos = position
    self.sharding = sharding
    self.env = env

    self.new_ks = None
    self.new_vs = None
    self.env = env
    # Keep this one it's used in the specific model code.
    self.stacked = env.generate_cache_stacked
    self.batch = jnp.arange(self.env.batch_size)
    # The other way is to store the list and loop over to insert in finalize()
    if self.env.lazy_cache_update:
      if self.env.generate_cache_stacked:
        if self.env.new_cache_stacked:
          layer, batch, heads, _, dim = self.cache_k.shape
          new_dim = (layer, batch, heads, 1, dim)
          self.new_ks, self.new_vs = torchjax.to_torch(
              (
                  jnp.zeros(new_dim, dtype=self.env.default_type),
                  jnp.zeros(new_dim, dtype=self.env.default_type),
              )
          )
        else:
          self.new_ks, self.new_vs = [], []
      else:  # when generate cache is not stacked, new cache cannot stack
        assert not self.env.new_cache_stacked

    cache_pspec = self.env.partition_by_axis(
        self.env.cache_sharding_axis
    )  # Number of heads
    none_pspec = self.env.partition_by_axis()
    in_specs = (cache_pspec, cache_pspec, cache_pspec, cache_pspec, none_pspec)
    out_specs = (cache_pspec, cache_pspec)
    self.update_single_cache_line = jax.jit(
        shard_map(
            self.update_single_cache_line,
            self.env.mesh,
            in_specs,
            out_specs,
            check_rep=False,
        )
    )

  # pylint: disable=method-hidden
  # False alarm. The jit above doesn't hide this method.
  def update_single_cache_line(self, cache_k, cache_v, new_ks, new_vs, pos):
    """The shard map version of single cache line update."""
    b = cache_k.shape[-4]
    for bb, pp in enumerate(pos.reshape(b)):
      slice_dim = 0
      update_start_indices = (bb, 0, pp, 0)
      if self.env.generate_cache_stacked:
        if self.env.new_cache_stacked:
          slice_dim = 1
          update_start_indices = (0, bb, 0, pp, 0)
      # We are not handling generate_cache_stacked=True new_cache_stacked=False here
      new_ks_slice = jax.lax.dynamic_slice_in_dim(new_ks, bb, 1, slice_dim)
      new_vs_slice = jax.lax.dynamic_slice_in_dim(new_vs, bb, 1, slice_dim)
      cache_k = jax.lax.dynamic_update_slice(
          cache_k, new_ks_slice, update_start_indices
      )
      cache_v = jax.lax.dynamic_update_slice(
          cache_v, new_vs_slice, update_start_indices
      )
    return cache_k, cache_v

  def finalize(self):
    """Finalize the cache operation and updates the cache."""
    if not self.env.lazy_cache_update:
      return

    if self.env.ring_buffer:
      # Assume no cache stack for ring buffer
      # pylint: disable-next=all
      self.cache_k._elem = (
          self.cache_k.jax().at[..., self.input_pos, :].set(self.new_ks.jax())
      )
      # pylint: disable-next=all
      self.cache_v._elem = (
          self.cache_v.jax().at[..., self.input_pos, :].set(self.new_vs.jax())
      )
    else:
      if self.env.generate_cache_stacked:
        _, b, head, _, dim = self.cache_k.shape
        if self.env.new_cache_stacked:
          self.cache_k, self.cache_v = torch_xla2.interop.call_jax(
              self.update_single_cache_line,
              self.cache_k,
              self.cache_v,
              self.new_ks,
              self.new_vs,
              self.input_pos,
          )
        else:
          for i in range(self.env.num_layers):
            # pylint: disable-next=all
            self.cache_k._elem = (
                self.cache_k.jax()
                .at[i, self.batch, :, self.input_pos, :]
                .set(self.new_ks[i].jax().reshape(b, head, dim))
            )
            # pylint: disable-next=all
            self.cache_v._elem = (
                self.cache_v.jax()
                .at[i, self.batch, :, self.input_pos, :]
                .set(self.new_vs[i].jax().reshape(b, head, dim))
            )
      else:
        # Try to use shard_map to get rid of the data copy
        self.cache_k, self.cache_v = torch_xla2.interop.call_jax(
            self.update_single_cache_line,
            self.cache_k,
            self.cache_v,
            self.new_ks,
            self.new_vs,
            self.input_pos,
        )

  def update(self, key, value, layer_id: int):
    """Update kv cache"""
    keyj, valuej = torchjax.to_torch((key, value))
    if self.env.lazy_cache_update:
      if self.env.new_cache_stacked:
        assert (
            self.env.generate_cache_stacked
        ), "When new cache stacked, must have generate_cache_stacked!"
        self.new_ks[layer_id, ...] = keyj
        self.new_vs[layer_id, ...] = valuej
        return self.cache_k[layer_id], self.cache_v[layer_id]

      # Generate cache stacked, but new cache unstacked
      if self.env.generate_cache_stacked:
        self.new_ks.append(keyj)
        self.new_vs.append(valuej)
        return self.cache_k[layer_id], self.cache_v[layer_id]

      # all cache unstacked
      self.new_ks = keyj
      self.new_vs = valuej
      return self.cache_k, self.cache_v

    if self.env.ring_buffer:
      assert (
          not self.env.new_cache_stacked and not self.env.generate_cache_stacked
      ), "Ring buffer doesn't support stacked cache."
      # pylint: disable-next=all
      self.cache_k._elem = (
          self.cache_k.jax().at[..., self.input_pos, :].set(keyj)
      )
      # pylint: disable-next=all
      self.cache_v._elem = (
          self.cache_v.jax().at[..., self.input_pos, :].set(valuej)
      )
      return self.cache_k, self.cache_v

    # Non lazy cache update, non ring buffer, generate cache stacked
    if self.env.generate_cache_stacked:
      # pylint: disable-next=all
      self.cache_k._elem = (
          self.cache_k.jax()
          .at[layer_id, self.batch, :, self.input_pos, :]
          .set(keyj.squeeze(2))
      )
      # pylint: disable-next=all
      self.cache_v._elem = (
          self.cache_v.jax()
          .at[layer_id, self.batch, :, self.input_pos, :]
          .set(valuej.squeeze(2))
      )
      return self.cache_k[layer_id], self.cache_v[layer_id]

    # Non lazy cache update, non ring buffer, generate cache non stacked
    # pylint: disable-next=all
    self.cache_k._elem = (
        self.cache_k.jax()
        .at[self.batch, :, self.input_pos, :]
        .set(keyj.squeeze(2))
    )
    # pylint: disable-next=all
    self.cache_v._elem = (
        self.cache_v.jax()
        .at[self.batch, :, self.input_pos, :]
        .set(valuej.squeeze(2))
    )
    return self.cache_k, self.cache_v

  def state(self):
    """Get kv cache state"""
    return self.cache_k.jax(), self.cache_v.jax()

  @classmethod
  def empty(cls, shape, device, env):
    """Create empty kv caches"""
    default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32
    in_shape = shape
    if env.testing:
      key = jax.random.key(env.testing_seed)
      k_key, v_key = jax.random.split(key)
      k = jax.random.uniform(k_key, shape=in_shape, dtype=default_dtype)
      v = jax.random.uniform(v_key, shape=in_shape, dtype=default_dtype)
    else:
      k = jnp.zeros(in_shape, device=device, dtype=default_dtype)
      v = jnp.zeros(in_shape, device=device, dtype=default_dtype)
    k, v = torchjax.to_torch((k, v))
    return cls(k, v, 0, device, env=env)


# pylint: disable-next=all
def KVCacheGenerate_flatten(cache):
  return ((cache.cache_k.jax(), cache.cache_v.jax())), (
      cache.pos.jax(),
      cache.sharding.jax(),
  )


# pylint: disable-next=all
def KVCacheGenerate_unflatten(auxdata, data):
  position, sharding = auxdata
  cache_k, cache_v = torchjax.to_torch(data)
  cache = KVCacheGenerate(cache_k, cache_v, position, sharding)
  return cache


jax.tree_util.register_pytree_node(
    KVCacheGenerate, KVCacheGenerate_flatten, KVCacheGenerate_unflatten
)


class Int8KVCacheGenerate:
  """Int8 quantized kvache with scalers"""

  # pylint: disable=too-many-instance-attributes
  # More than 7 is reasonable in this case.
  def __init__(
      self,
      cache_k,
      cache_v,
      cache_k_scaler,
      cache_v_scaler,
      input_pos,  # used to write cache
      sharding=None,
      env=None,
  ):
    super().__init__()
    self.cache_k = cache_k
    self.cache_v = cache_v
    self.k_scaler = cache_k_scaler
    self.v_scaler = cache_v_scaler
    self.new_ks = None
    self.new_vs = None
    self.new_k_scaler = None
    self.new_v_scaler = None

    self.batch = jnp.arange(env.batch_size)
    self.input_pos = input_pos
    self.sharding = sharding
    self.env = env
    self.stacked = env.generate_cache_stacked

    if self.env.lazy_cache_update:
      if self.env.generate_cache_stacked:
        layer, batch, heads, _, dim = self.cache_k.shape
        new_kv_dim = (layer, batch, heads, 1, dim)
        self.new_ks, self.new_vs = torchjax.to_torch(
            (
                jnp.zeros(new_kv_dim, dtype=jnp.int8),
                jnp.zeros(new_kv_dim, dtype=jnp.int8),
            )
        )
        if self.env.new_cache_stacked:
          new_scale_dim = (layer, batch, 1, 1, 1)
          self.new_k_scaler, self.new_v_scaler = torchjax.to_torch(
              (
                  jnp.zeros(new_scale_dim, dtype=self.env.default_type),
                  jnp.zeros(new_scale_dim, dtype=self.env.default_type),
              )
          )
        else:
          self.new_ks, self.new_vs, self.new_k_scaler, self.new_v_scaler = (
              [],
              [],
              [],
              [],
          )
      else:  # when generate cache is not stacked, new cache cannot stack
        assert not self.env.new_cache_stacked

    cache_pspec = self.env.partition_by_axis(
        self.env.cache_sharding_axis
    )  # Number of heads
    new_cache_pspec = (
        self.env.partition_by_axis(2)
        if self.env.new_cache_stacked
        else self.env.partition_by_axis(1)
    )
    none_pspec = self.env.partition_by_axis()
    in_specs = (
        *([cache_pspec] * 2),
        *([new_cache_pspec] * 2),
        *([none_pspec] * 5),
    )
    out_specs = (cache_pspec, cache_pspec, none_pspec, none_pspec)
    self.update_single_cache_line = shard_map(
        self.update_single_cache_line,
        self.env.mesh,
        in_specs,
        out_specs,
        check_rep=False,
    )
    self.update_single_cache_line = jax.jit(self.update_single_cache_line)

  # pylint: disable=method-hidden
  # False alarm. The jit above doesn't hide this method.
  def update_single_cache_line(
      self,
      cache_k,
      cache_v,
      new_ks,
      new_vs,
      k_scaler,
      v_scaler,
      new_k_scaler,
      new_v_scaler,
      pos,
  ):
    """The shard map version of single cache line update."""
    b = cache_k.shape[-4]

    for bb, pp in enumerate(pos.reshape(b)):
      slice_dim = 0
      update_start_indices = (bb, 0, pp, 0)
      if self.env.generate_cache_stacked:
        if self.env.new_cache_stacked:
          slice_dim = 1
          update_start_indices = (0, bb, 0, pp, 0)
      if self.env.generate_cache_stacked and not self.env.new_cache_stacked:
        for layer in range(self.env.num_layers):
          update_start_indices = (layer, bb, 0, pp, 0)
          new_ks_slice = jax.lax.dynamic_slice_in_dim(
              new_ks[layer], bb, 1, slice_dim
          )
          new_ks_slice = jnp.expand_dims(new_ks_slice, 0)
          cache_k = jax.lax.dynamic_update_slice(
              cache_k, new_ks_slice, update_start_indices
          )

          new_vs_slice = jax.lax.dynamic_slice_in_dim(
              new_vs[layer], bb, 1, slice_dim
          )
          new_vs_slice = jnp.expand_dims(new_vs_slice, 0)
          cache_v = jax.lax.dynamic_update_slice(
              cache_v, new_vs_slice, update_start_indices
          )

          new_k_scaler_slice = jax.lax.dynamic_slice_in_dim(
              new_k_scaler[layer], bb, 1, slice_dim
          )
          new_k_scaler_slice = jnp.expand_dims(new_k_scaler_slice, 0)
          k_scaler = jax.lax.dynamic_update_slice(
              k_scaler, new_k_scaler_slice, update_start_indices
          )

          new_v_scaler_slice = jax.lax.dynamic_slice_in_dim(
              new_v_scaler[layer], bb, 1, slice_dim
          )
          new_v_scaler_slice = jnp.expand_dims(new_v_scaler_slice, 0)
          v_scaler = jax.lax.dynamic_update_slice(
              v_scaler, new_v_scaler_slice, update_start_indices
          )
      else:
        new_ks_slice = jax.lax.dynamic_slice_in_dim(new_ks, bb, 1, slice_dim)
        cache_k = jax.lax.dynamic_update_slice(
            cache_k, new_ks_slice, update_start_indices
        )

        new_vs_slice = jax.lax.dynamic_slice_in_dim(new_vs, bb, 1, slice_dim)
        cache_v = jax.lax.dynamic_update_slice(
            cache_v, new_vs_slice, update_start_indices
        )

        new_k_scaler_slice = jax.lax.dynamic_slice_in_dim(
            new_k_scaler, bb, 1, slice_dim
        )
        k_scaler = jax.lax.dynamic_update_slice(
            k_scaler, new_k_scaler_slice, update_start_indices
        )

        new_v_scaler_slice = jax.lax.dynamic_slice_in_dim(
            new_v_scaler, bb, 1, slice_dim
        )
        v_scaler = jax.lax.dynamic_update_slice(
            v_scaler, new_v_scaler_slice, update_start_indices
        )

    return cache_k, cache_v, k_scaler, v_scaler

  def state(self):
    """Get kv cache state"""
    return torchjax.from_torch((self.cache_k, self.cache_v))

  def scalers(self):
    """Get kv cache scalers"""
    return torchjax.from_torch((self.k_scaler, self.v_scaler))

  @classmethod
  # pylint: disable-next=all
  def empty(cls, shape, device, env):
    """Create empty kv caches"""
    cache_k = jnp.zeros(shape, device=device, dtype=jnp.int8)
    cache_v = jnp.zeros(shape, device=device, dtype=jnp.int8)

    if env.generate_cache_stacked:
      s_shape = (shape[0], shape[1], 1, shape[3], 1)
    else:
      s_shape = (shape[0], 1, shape[2], 1)
    kscaler = jnp.ones(s_shape, dtype=jnp.bfloat16)
    vscaler = jnp.ones(s_shape, dtype=jnp.bfloat16)

    cache_k, cache_v, kscaler, vscaler = torchjax.to_torch(
        (cache_k, cache_v, kscaler, vscaler)
    )
    return cls(cache_k, cache_v, kscaler, vscaler, 0, device, env=env)

  def quantize(self, val):
    """Quantize value"""
    # val is (batch, heads, seqlen, dim)
    scale = torch.amax(val.abs(), axis=(-3, -1), keepdim=True)
    scale = scale / 127
    return (val / scale).to(torch.int8), scale

  def update(self, xk, xv, layer_id: int):
    """Update kv cache"""
    k_quant, kscale = self.quantize(xk)
    v_quant, vscale = self.quantize(xv)

    if self.env.lazy_cache_update:
      if self.env.new_cache_stacked:
        self.new_ks[layer_id, ...] = k_quant
        self.new_vs[layer_id, ...] = v_quant
        self.new_k_scaler[layer_id, ...] = kscale
        self.new_v_scaler[layer_id, ...] = vscale
      else:
        if self.env.generate_cache_stacked:
          self.new_ks.append(k_quant)
          self.new_vs.append(v_quant)
          self.new_k_scaler.append(kscale)
          self.new_v_scaler.append(vscale)
        else:
          self.new_ks = k_quant
          self.new_vs = v_quant
          self.new_k_scaler = kscale
          self.new_v_scaler = vscale
    elif self.env.ring_buffer:
      self.cache_k[:, :, self.input_pos, :] = k_quant
      self.cache_v[:, :, self.input_pos, :] = v_quant
      self.k_scaler[:, :, self.input_pos, :] = kscale
      self.v_scaler[:, :, self.input_pos, :] = vscale
    else:
      # We don't handle left aligned but lazy_cache_update=False
      self.cache_k[self.batch, :, self.input_pos, :] = k_quant.squeeze(2)
      self.cache_v[self.batch, :, self.input_pos, :] = v_quant.squeeze(2)
      self.k_scaler[self.batch, :, self.input_pos, :] = kscale.squeeze(2)
      self.v_scaler[self.batch, :, self.input_pos, :] = vscale.squeeze(2)

    return (
        self.cache_k,
        self.cache_v,
        k_quant,
        v_quant,
        self.k_scaler,
        self.v_scaler,
        kscale,
        vscale,
    )

  def finalize(self):
    """Finalize the cache operation and updates the cache."""
    if not self.env.lazy_cache_update:
      return
    if self.env.ring_buffer:
      # Assume no cache stack for ring buffer
      # pylint: disable-next=all
      self.cache_k._elem = (
          self.cache_k.jax().at[..., self.input_pos, :].set(self.new_ks.jax())
      )
      # pylint: disable-next=all
      self.cache_v._elem = (
          self.cache_v.jax().at[..., self.input_pos, :].set(self.new_vs.jax())
      )
    else:
      if self.env.generate_cache_stacked:
        if self.env.new_cache_stacked:
          # new kv scaler also has to go through shard_map instead of indexing
          # because it needs to reshape to (batch, layer) which mess up with the data
          caches = [
              self.cache_k,
              self.cache_v,
              self.new_ks,
              self.new_vs,
              self.k_scaler,
              self.v_scaler,
              self.new_k_scaler,
              self.new_v_scaler,
          ]
          (
              self.cache_k,
              self.cache_v,
              self.k_scaler,
              self.v_scaler,
          ) = torch_xla2.interop.call_jax(
              self.update_single_cache_line, *caches, self.input_pos
          )
        else:
          caches = [
              self.cache_k,
              self.cache_v,
              self.new_ks,
              self.new_vs,
              self.k_scaler,
              self.v_scaler,
              self.new_k_scaler,
              self.new_v_scaler,
          ]
          (
              self.cache_k,
              self.cache_v,
              self.k_scaler,
              self.v_scaler,
          ) = torch_xla2.interop.call_jax(
              self.update_single_cache_line, *caches, self.input_pos
          )
      else:
        (
            self.cache_k,
            self.cache_v,
            self.k_scaler,
            self.v_scaler,
        ) = torch_xla2.interop.call_jax(
            self.update_single_cache_line,
            self.cache_k,
            self.cache_v,
            self.new_ks,
            self.new_vs,
            self.k_scaler,
            self.v_scaler,
            self.new_k_scaler,
            self.new_v_scaler,
            self.input_pos,
        )


class PageKVCacheGenerate:
  """Page attention kvache generator without quantization"""

  def __init__(
      self,
      cache_k: torch.Tensor,  # previous cache
      cache_v: torch.Tensor,  # previous cache
      page_attention_manager: PageAttentionManager,
      page_token_indices: torch.Tensor,  # page and token indices for the cache
      sharding,
      env=None,
  ):
    super().__init__()
    self.cache_k = cache_k
    self.cache_v = cache_v
    self.page_attention_manager = page_attention_manager
    self.page_token_indices = page_token_indices
    self.sharding = sharding
    self.env = env
    self.stacked = False

  def update(self, key, value, layer_id=0):
    """Update kv cache"""
    keyj, valuej, page_token_indicesj = torchjax.from_torch(
        (key, value, self.page_token_indices)
    )

    def _update(cache, x):
      x = x.squeeze(2).transpose((1, 0, 2))
      x = x[:, page_token_indicesj[2], :]
      head, _, page_size, dim = cache.shape
      selected_cache = cache[:, page_token_indicesj[0], :, :]
      selected_cache = selected_cache.reshape((head, -1, dim))

      selected_cache = selected_cache.at[:, page_token_indicesj[1], :].set(x)
      selected_cache = selected_cache.reshape((head, -1, page_size, dim))

      cache = cache.at[:, page_token_indicesj[0], :, :].set(selected_cache)
      return cache

    # pylint: disable-next=all
    self.cache_k._elem = _update(self.cache_k._elem, keyj)
    # pylint: disable-next=all
    self.cache_v._elem = _update(self.cache_v._elem, valuej)
    return self.cache_k, self.cache_v

  def state(self):
    """Get kv cache state"""
    # pylint: disable-next=all
    return torchjax.from_torch((self.cache_k, self.cache_v))
  
  def finalize(self):
    return
  
  @classmethod
  def empty(cls, shape, device, env):
    """Create empty kv caches"""
    default_dtype = jnp.bfloat16 if env.bf16_enable else jnp.float32
    k = jnp.zeros(shape, device=device, dtype=default_dtype)
    v = jnp.zeros(shape, device=device, dtype=default_dtype)
    k, v = torchjax.to_torch((k, v))
    return cls(k, v, None, None, device, env=env)
