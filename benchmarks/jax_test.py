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
import time
import jax
import jax.numpy as jnp
import functools


def test1():
  @functools.partial(jax.jit, static_argnums=(2,))
  def f(x, i, issum):
    if issum:
      return x + i
    else:
      return x - i

  x = jnp.ones((10,))
  print(f(x, 0, True))
  print("cache", f._cache_size())
  print(f(x, 1, False))
  print("cache", f._cache_size())

  class A:

    def __init__(self, a):
      self.a = a

    def incr(self):
      self.a += 1

  @jax.jit
  def f(x):
    a = A(x)
    a.incr()
    return a.a

  print(f(x))
  print(f(x))
  print(f(x))


from jax.sharding import PositionalSharding
from jax.experimental import mesh_utils


def test2():
  batch, seq, heads, dim = 96, 2048, 40, 128
  sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
  sharding = sharding.reshape((1, 8, 1, 1))
  val_sharding = sharding.reshape((1, 8, 1, 1))
  caches_k = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )
  caches_v = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )

  def insert_cache(caches_k, caches_v, pos, key, val):
    # val is of shape b,h,d
    return caches_k.at[:, :, pos, :].set(key.squeeze(2)), caches_v.at[
        :, :, pos, :
    ].set(val.squeeze(2))
    # return caches_k.at[:, :, pos:pos+1, :].set(key), caches_v.at[:, :, pos:pos+1, :].set(val)

  def insert_cache2(caches_k, caches_v, pos, key, val):
    # val is of shape b,h,d
    seqlen = caches_k.shape[2]
    val = jnp.broadcast_to(val, caches_k.shape)
    iota = jnp.arange(0, seqlen).reshape(1, 1, seqlen, 1)
    iota = jnp.broadcast_to(iota, caches_k.shape)
    pos = jnp.broadcast_to(pos, (seqlen,))
    return (
        jnp.where(iota == pos.reshape(1, 1, seqlen, 1), caches_k, key),
        jnp.where(iota == pos.reshape(1, 1, seqlen, 1), caches_v, val),
    )

  def insert_cache3(caches_k, caches_v, pos, key, val):
    return (
        jax.lax.dynamic_update_slice(caches_k, key, (0, 0, pos, 0)),
        jax.lax.dynamic_update_slice(caches_k, key, (0, 0, pos, 0)),
    )

  insert_cache = jax.jit(insert_cache, donate_argnums=(0, 1))
  insert_cache2 = jax.jit(insert_cache2, donate_argnums=(0, 1))
  insert_cache3 = jax.jit(insert_cache3, donate_argnums=(0, 1))

  subkey = jax.random.PRNGKey(234)
  to_insert = jax.device_put(
      jax.random.normal(subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16),
      device=val_sharding,
  ).block_until_ready()
  j = jnp.int32(7).block_until_ready()

  print("====1====")
  print(
      insert_cache.lower(caches_k, caches_v, j, to_insert, to_insert).as_text()
  )

  print("====2====")
  print(
      insert_cache2.lower(caches_k, caches_v, j, to_insert, to_insert).as_text()
  )

  print("====3====")
  print(
      insert_cache3.lower(caches_k, caches_v, j, to_insert, to_insert).as_text()
  )

  rng = jax.random.PRNGKey(0)

  for func in (insert_cache, insert_cache2, insert_cache3):
    for i in range(10):
      all_times = 0
      for j in range(40):
        rng, subkey = jax.random.split(rng)
        key = jax.device_put(
            jax.random.normal(
                subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16
            ),
            device=val_sharding,
        ).block_until_ready()
        val = jax.device_put(
            jax.random.normal(
                subkey, (batch, heads, 1, dim), dtype=jnp.bfloat16
            ),
            device=val_sharding,
        ).block_until_ready()
        j = jnp.int32(j).block_until_ready()
        start = time.perf_counter()
        caches_k, caches_v = func(caches_k, caches_v, j, key, val)
        caches_k.block_until_ready()
        caches_v.block_until_ready()
        end = time.perf_counter()
        all_times += end - start
      print(func.__name__, "time is", all_times)


def test3():
  import torch
  import torch_xla2
  import torch_xla2.extra

  x = jnp.ones((10, 10, 10))
  y = jnp.ones((10, 10, 10))

  def f(x, y):
    return torch.einsum("ijm, ijn -> imn", [x, y])

  def g(x, y):
    return jnp.einsum("ijm, ijn -> imn", x, y)

  print("====== 1 ======")
  with torch_xla2.tensor.XLAFunctionMode():
    print(jax.jit(torch_xla2.extra.jax_view(f)).lower(x, y).as_text())
  print("====== 2 ======")
  print(jax.jit(g).lower(x, y).as_text())


from flax import struct


class A:

  def __init__(self, a):
    self.a = a

  def plus(self):
    self.a = self.a + 1


def flatten_A(x):
  return (x.a,), None


def unflatten_A(aux_data, flat_content):
  import pdb

  pdb.set_trace()
  return A(*flat_content)


jax.tree_util.register_pytree_node(A, flatten_A, unflatten_A)

import functools


@functools.partial(jax.jit, donate_argnums=(0,))
def f(a):
  a.plus()
  return a


def test4():
  a = A(a=jnp.zeros((2,)))
  b = f(a)
  print(b.a)


def test5():
  batch, seq, heads, dim = 96, 2048, 40, 128
  sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
  sharding = sharding.reshape((1, 8, 1, 1))
  val_sharding = sharding.reshape((1, 8, 1, 1))
  caches_k = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )
  caches_v = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )

  def insert_cache(cache, new_entry, slot, head_indexes, update_indexes):
    res = cache.at[slot, head_indexes, update_indexes.reshape(1, -1), :].set(
        new_entry
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  def insert_cache2(cache, new_entry, slot, head_indexes, update_indexes):
    res = cache.at[slot, :, update_indexes, :].set(
        jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  def insert_cache3(cache, new_entry, slot, head_indexes, update_indexes):

    index = jnp.expand_dims(jnp.full_like(update_indexes, slot), -1)
    update_indexes = jnp.expand_dims(update_indexes, -1)
    combined = jnp.concatenate([index, update_indexes], axis=-1)
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        update_window_dims=[0, 2],
        inserted_window_dims=[0, 2],
        scatter_dims_to_operand_dims=[1, 3],
    )
    res = jax.lax.scatter(
        cache,
        combined,
        new_entry.squeeze(0),
        dimension_numbers,
        unique_indices=True,
        indices_are_sorted=True,
        mode="promise_in_bounds",
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  insert_cache = jax.jit(insert_cache, donate_argnums=(0, 1))
  insert_cache2 = jax.jit(insert_cache2, donate_argnums=(0, 1))
  insert_cache3 = jax.jit(insert_cache3, donate_argnums=(0, 1))
  insert_seqlen = 1024

  subkey = jax.random.PRNGKey(234)
  to_insert = jax.device_put(
      jax.random.normal(
          subkey, (1, heads, insert_seqlen, dim), dtype=jnp.bfloat16
      ),
      device=val_sharding,
  ).block_until_ready()
  j = jnp.int32(7).block_until_ready()

  update_indexes = (jnp.arange(-insert_seqlen, 0) + 7) % 1024
  update_indexes = update_indexes
  head_indexes = jnp.arange(heads).reshape(1, -1, 1)

  rng = jax.random.PRNGKey(0)
  for func in (insert_cache3,):
    print(f"===={func.__name__}====")
    print(
        func.lower(
            caches_k, to_insert, j, head_indexes, update_indexes
        ).as_text()
    )

  for func in (insert_cache, insert_cache2, insert_cache3):
    for i in range(10):
      all_times = 0
      for j in range(40):
        rng, subkey = jax.random.split(rng)
        key = jax.device_put(
            jax.random.normal(
                subkey, (1, heads, insert_seqlen, dim), dtype=jnp.bfloat16
            ),
            device=val_sharding,
        ).block_until_ready()
        j = jnp.int32(j).block_until_ready()
        start = time.perf_counter()
        caches_k = func(caches_k, to_insert, j, head_indexes, update_indexes)
        caches_k.block_until_ready()
        end = time.perf_counter()
        all_times += end - start
      print(func.__name__, "time is", all_times)


test5()
