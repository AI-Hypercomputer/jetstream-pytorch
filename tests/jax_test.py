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

import functools
import time

import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import PositionalSharding
import torch
import torch_xla2
import torch_xla2.extra


def test1():
  """test jit cache size"""

  @functools.partial(jax.jit, static_argnums=(2,))
  # pylint: disable-next=all
  def f(x, i, issum):
    if issum:
      return x + i

    return x - i

  x = jnp.ones((10,))
  print(f(x, 0, True))
  print("cache", f._cache_size())
  print(f(x, 1, False))
  print("cache", f._cache_size())

  # pylint: disable-next=all
  class A:

    def __init__(self, a):
      self.a = a

    def incr(self):
      """increase by 1"""
      self.a += 1

  @jax.jit
  def f2(x):
    a = A(x)
    a.incr()
    return a.a

  print(f2(x))
  print(f2(x))
  print(f2(x))


# pylint: disable-next=all
def test2():
  """test insert cache"""
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

  # pylint: disable-next=all
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
  # pylint: disable-next=all
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
    for _ in range(10):
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
        # pylint: disable-next=all
        j = jnp.int32(j).block_until_ready()
        start = time.perf_counter()
        # pylint: disable-next=all
        caches_k, caches_v = func(caches_k, caches_v, j, key, val)
        caches_k.block_until_ready()
        caches_v.block_until_ready()
        end = time.perf_counter()
        all_times += end - start
      print(func.__name__, "time is", all_times)


def test3():
  """test einsum"""

  x = jnp.ones((10, 10, 10))
  y = jnp.ones((10, 10, 10))

  # pylint: disable-next=all
  def f(x, y):
    return torch.einsum("ijm, ijn -> imn", [x, y])

  def g(x, y):
    return jnp.einsum("ijm, ijn -> imn", x, y)

  print("====== 1 ======")
  # pylint: disable-next=all
  with torch_xla2.tensor.XLAFunctionMode():
    print(jax.jit(torch_xla2.extra.jax_view(f)).lower(x, y).as_text())
  print("====== 2 ======")
  print(jax.jit(g).lower(x, y).as_text())


# pylint: disable-next=all
class A:
  """Define class to do plus"""

  def __init__(self, a):
    self.a = a

  def plus(self):
    """plus"""
    self.a = self.a + 1


# pylint: disable-next=all
def flatten_A(x):
  """flatten"""
  return (x.a,), None


# pylint: disable-next=all
def unflatten_A(aux_data, flat_content):
  """unflatten"""

  # pdb.set_trace()
  return A(*flat_content)


jax.tree_util.register_pytree_node(A, flatten_A, unflatten_A)


@functools.partial(jax.jit, donate_argnums=(0,))
def f(a):
  """plus"""
  a.plus()
  return a


def test4():
  """test plus"""
  a = A(a=jnp.zeros((2,)))
  b = f(a)
  print(b.a)


# pylint: disable-next=all
def test5():
  """insert cache test"""
  batch, seq, heads, dim = 96, 2048, 40, 128
  sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
  sharding = sharding.reshape((1, 8, 1, 1))
  val_sharding = sharding.reshape((1, 8, 1, 1))
  caches_k = jnp.zeros(
      (batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16
  )
  jnp.zeros((batch, heads, seq, dim), device=sharding, dtype=jnp.bfloat16)

  def insert_cache(cache, new_entry, slot, head_indexes, update_indexes):
    res = cache.at[slot, head_indexes, update_indexes.reshape(1, -1), :].set(
        new_entry
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  # pylint: disable-next=all
  def insert_cache2(cache, new_entry, slot, head_indexes, update_indexes):
    res = cache.at[slot, :, update_indexes, :].set(
        jnp.transpose(new_entry.squeeze(0), (1, 0, 2))
    )
    res = jax.lax.with_sharding_constraint(res, sharding)
    return res

  # pylint: disable-next=all
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
  # pylint: disable-next=all
  j = jnp.int32(7).block_until_ready()

  update_indexes = (jnp.arange(-insert_seqlen, 0) + 7) % 1024
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
    for _ in range(10):
      all_times = 0
      for j in range(40):
        rng, subkey = jax.random.split(rng)
        jax.device_put(
            jax.random.normal(
                subkey, (1, heads, insert_seqlen, dim), dtype=jnp.bfloat16
            ),
            device=val_sharding,
        ).block_until_ready()
        # pylint: disable-next=all
        j = jnp.int32(j).block_until_ready()
        start = time.perf_counter()
        # pylint: disable-next=all
        caches_k = func(caches_k, to_insert, j, head_indexes, update_indexes)
        caches_k.block_until_ready()
        end = time.perf_counter()
        all_times += end - start
      print(func.__name__, "time is", all_times)


def test6():
  """move device test"""

  x = torch.randn(10, 20, 20, 20)
  x = torch_xla2.tensor.move_to_device(x)
  print(x[:, :, 0:1, :])


test6()
