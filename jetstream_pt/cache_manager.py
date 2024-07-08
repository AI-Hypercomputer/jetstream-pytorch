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
import torch
from jetstream_pt import torchjax
from jax.experimental.shard_map import shard_map


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
      return key, value, ones, ones

    return key, value

  def state(self):
    """Get prefill cache state"""
    return self.cache_k, self.cache_v

  # Placeholder, to match with GenerateCache
  def finalize(self):
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


# Refactor out cache management
# Easier to test for quantized kv cache
class KVCacheGenerate:
  """Kvache generator without quantization"""

  def __init__(
      self,
      cache_k: torch.Tensor,  # previous cache
      cache_v: torch.Tensor,  # previous cache
      position: int,  # position to store the cache
      sharding,
      env=None,
  ):
    super().__init__()
    self.cache_k = cache_k
    self.cache_v = cache_v
    self.pos = position
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
          layer, batch, heads, time, dim = self.cache_k.shape
          new_dim = (layer, batch, heads, 1, dim)
          self.new_ks, self.new_vs = torchjax.to_torch((jnp.zeros(new_dim), jnp.zeros(new_dim)))
        else:
          self.new_ks, self.new_vs = [], []
      else:  # when generate cache is not stacked, new cache cannot stack
        assert not self.env.new_cache_stacked

    cache_pspec = self.env.partition_by_axis(self.cache_sharding_axis)  # Number of heads
    in_specs = (cache_pspec, cache_pspec)
    out_specs = (cache_pspec, cache_pspec)
    self.update_single_cache_line = shard_map(self.update_single_cache_line, self.env.mesh, in_specs, out_specs)

  def update_single_cache_line(self, cache_k, cache_v):
    b, head, len, dim = cache_k.shape
    cache_k._elem = cache_k._elem.at[self.batch, :, self.pos, :].set(self.new_ks._elem.reshape(b, head, dim))
    cache_v._elem = cache_v._elem.at[self.batch, :, self.pos, :].set(self.new_vs._elem.reshape(b, head, dim))
    return cache_k, cache_v

  def finalize(self):
    if not self.env.lazy_cache_update:
      return
      # self.cache_k._elem = self.cache_k._elem.at[:, :, :, self.pos].set(jnp.squeeze(self.new_ks._elem, -2))
      # self.cache_v._elem = self.cache_v._elem.at[:, :, :, self.pos].set(jnp.squeeze(self.new_vs._elem, -2))
    if self.env.ring_buffer:
      # Assume no cache stack for ring buffer
      self.cache_k._elem = self.cache_k._elem.at[..., self.pos, :].set(self.new_ks._elem)
      self.cache_v._elem = self.cache_v._elem.at[..., self.pos, :].set(self.new_vs._elem)
    else:
      if self.env.generate_cache_stacked:
        layer, b, head, len, dim = self.cache_k.shape
        if self.env.new_cache_stacked:
          self.cache_k._elem = self.cache_k._elem.at[:, self.batch, :, self.pos, :].set(self.new_ks._elem.reshape(b, layer, head, dim))
          self.cache_v._elem = self.cache_v._elem.at[:, self.batch, :, self.pos, :].set(self.new_vs._elem.reshape(b, layer, head, dim))
        else:
          for i in range(self.env.num_layers):
            self.cache_k._elem = self.cache_k._elem.at[i, self.batch, :, self.pos, :].set(self.new_ks[i]._elem.reshape(b, head, dim))
            self.cache_v._elem = self.cache_v._elem.at[i, self.batch, :, self.pos, :].set(self.new_vs[i]._elem.reshape(b, head, dim))
      else:
        # Try to use shard_map to get rid of the data copy
        b, head, len, dim = self.cache_k.shape
        self.cache_k, self.cache_v = self.update_single_cache_line(self.cache_k, self.cache_v)
        # self.cache_k._elem = self.cache_k._elem.at[self.batch, :, self.pos, :].set(self.new_ks._elem.reshape(b, head, dim))
        # self.cache_v._elem = self.cache_v._elem.at[self.batch, :, self.pos, :].set(self.new_vs._elem.reshape(b, head, dim))

  def update(self, key, value, layer_id:int):
    """Update kv cache"""
    # Will process in insert() at the end of the transformer forward pass
    keyj, valuej = torchjax.to_torch((key, value))
    if self.env.lazy_cache_update:
      # When new cache stacked, must have generate_cache_stacked
      if self.env.new_cache_stacked:
        self.new_ks[layer_id, ...] = keyj
        self.new_vs[layer_id, ...] = valuej
        return self.cache_k[layer_id], self.cache_v[layer_id]
      else:
        if self.env.generate_cache_stacked:
          self.new_ks.append(keyj)
          self.new_vs.append(valuej)
          return self.cache_k[layer_id], self.cache_v[layer_id]
        else:
          self.new_ks = keyj
          self.new_vs = valuej
          return self.cache_k, self.cache_v
      

    if self.env.ring_buffer:
      # Assume no cache stack for ring buffer
      # pylint: disable-next=all
      self.cache_k._elem = self.cache_k._elem.at[..., self.pos, :].set(keyj)
      self.cache_v._elem = self.cache_v._elem.at[..., self.pos, :].set(valuej)
      return self.cache_k, self.cache_v
    else:
      if self.env.generate_cache_stacked:
              # pylint: disable-next=all
        self.cache_k._elem = self.cache_k._elem.at[layer_id, self.batch, :, self.pos, :].set(
            keyj.squeeze(2)
        )
        # pylint: disable-next=all
        self.cache_v._elem = self.cache_v._elem.at[layer_id, self.batch, :, self.pos, :].set(
            valuej.squeeze(2)
        )
        return self.cache_k[layer_id], self.cache_v[layer_id]
      else:
        # pylint: disable-next=all
        self.cache_k._elem = self.cache_k._elem.at[self.batch, :, self.pos, :].set(
            keyj.squeeze(2)
        )
        # pylint: disable-next=all
        self.cache_v._elem = self.cache_v._elem.at[self.batch, :, self.pos, :].set(
            valuej.squeeze(2)
        )
        return self.cache_k, self.cache_v

  def state(self):
    """Get kv cache state"""
    # pylint: disable-next=all
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

  # pylint: disable-next=all
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
    self.input_pos = input_pos
    self.sharding = sharding
    self.env = env

    if self.env.generate_cache_stacked:
      layer, batch, heads, len, dim = self.cache_k.shape
      self.new_ks, self.new_vs, self.new_k_scalers, self.new_v_scalers = torchjax.to_torch((jnp.zeros((layer, batch, heads, 1, dim)), jnp.zeros((layer, batch, heads, 1, dim)), jnp.zeros((layer, batch, 1, 1, 1)), jnp.zeros((layer, batch, 1, 1, 1))))

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
      kscaler = jnp.ones((shape[0], shape[1], 1, shape[2], 1), dtype=jnp.bfloat16)
      vscaler = jnp.ones((shape[0], shape[1], 1, shape[2], 1), dtype=jnp.bfloat16)
    else:
      kscaler = jnp.ones((shape[0], 1, shape[2], 1), dtype=jnp.bfloat16)
      vscaler = jnp.ones((shape[0], 1, shape[2], 1), dtype=jnp.bfloat16)

    cache_k, cache_v, kscaler, vscaler = torchjax.to_torch(
        (cache_k, cache_v, kscaler, vscaler)
    )
    return cls(cache_k, cache_v, kscaler, vscaler, 0, device, env=env)

  def quantize(self, val):
    """Quantize value"""
    # val is (batch, heads, seqlen, dim)
    scale = torch.amax(val.abs(), axis=(1, 3), keepdim=True)
    scale = scale / 127
    return (val / scale).to(torch.int8), scale

  def update(self, xk, xv, layer_id:int):
    """Update kv cache"""
    k_quant, kscale = self.quantize(xk)
    v_quant, vscale = self.quantize(xv)

    if self.stacked:
      self.new_ks[layer_id, ...] = k_quant
      self.new_vs[layer_id, ...] = v_quant
      self.new_k_scalers[layer_id, ...] = kscale
      self.new_v_scalers[layer_id, ...] = vscale

      return self.cache_k[layer_id], self.cache_v[layer_id], k_quant, v_quant, self.k_scaler[layer_id], self.v_scaler[layer_id], kscale, vscale

    if self.env.ring_buffer:
      self.cache_k[:, :, self.input_pos, :] = k_quant
      self.cache_v[:, :, self.input_pos, :] = v_quant
      self.k_scaler[:, :, self.input_pos, :] = kscale
      self.v_scaler[:, :, self.input_pos, :] = vscale
    else:
      batch = jnp.arange(self.env.batch_size)
      self.cache_k[batch, :, self.input_pos, :] = k_quant.squeeze(2)
      self.cache_v[batch, :, self.input_pos, :] = v_quant.squeeze(2)
      self.k_scaler[batch, :, self.input_pos, :] = kscale.squeeze(2)
      self.v_scaler[batch, :, self.input_pos, :] = vscale.squeeze(2)
    return self.cache_k, self.cache_v, k_quant, v_quant, self.k_scaler, self.v_scaler, kscale, vscale

  def finalize(self):
    if not self.stacked:
      return
      # self.cache_k._elem = self.cache_k._elem.at[:, :, :, self.pos].set(jnp.squeeze(self.new_ks._elem, -2))
      # self.cache_v._elem = self.cache_v._elem.at[:, :, :, self.pos].set(jnp.squeeze(self.new_vs._elem, -2))
    if self.env.ring_buffer:
      self.cache_k._elem = self.cache_k._elem.at[:, :, :, self.pos].set(self.new_ks._elem)
      self.cache_v._elem = self.cache_v._elem.at[:, :, :, self.pos].set(self.new_vs._elem)
      self.k_scaler._elem = self.k_scaler._elem.at[:, :, :, self.pos].set(self.new_k_scalers._elem)
      self.v_scaler._elem = self.v_scaler._elem.at[:, :, :, self.pos].set(self.new_v_scalers._elem)
    else:
      batch = jnp.arange(self.env.batch_size)
      self.cache_k._elem = self.cache_k._elem.at[:, batch, :, self.pos].set(self.new_ks._elem)
      self.cache_v._elem = self.cache_v._elem.at[:, batch, :, self.pos].set(self.new_vs._elem)
      self.k_scaler._elem = self.k_scaler._elem.at[:, batch, :, self.pos].set(self.new_k_scalers._elem)
      self.v_scaler._elem = self.v_scaler._elem.at[:, batch, :, self.pos].set(self.new_v_scalers._elem)