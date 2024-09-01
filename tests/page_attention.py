from collections.abc import Callable
import functools
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import paged_attention
import humanize
from jax.experimental import shard_map


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
P = jax.sharding.PartitionSpec

def reconstruct_kv(page_indices: jax.Array, pages: jax.Array) -> jax.Array:
  """For each sequence, gather pages to reconstruct K/V."""
  batch_size = page_indices.shape[0]
  head_dim = pages.shape[-1]

  def per_sequence_page_gather(pages, page_indices):
    return jnp.take(pages, page_indices, 1)

  gathered = jax.vmap(per_sequence_page_gather, in_axes=(None, 0))(
      pages, page_indices
  )

  num_heads = pages.shape[0]
  return gathered.reshape(batch_size, num_heads, -1, head_dim)

def generate_qkv_decode(
    seq_lens,
    page_size,
    max_seq_len,
    num_heads,
    head_dim,
    prng_key,
    mqa=True,
    num_kv_heads=1,
    dtype=jnp.float32,
):
  assert max_seq_len % page_size == 0
  pages_per_sequence = max_seq_len // page_size
  batch_size = seq_lens.shape[0]
  total_pages = batch_size * pages_per_sequence
  k1, k2, k3, k4 = jax.random.split(prng_key, 4)
  if mqa:
    assert num_kv_heads == 1
    k_pages = jax.random.normal(
        k1,
        (total_pages, page_size, head_dim),
        dtype=dtype,
    )
    v_pages = jax.random.normal(
        k2,
        (total_pages, page_size, head_dim),
        dtype=dtype,
    )
  else:
    k_pages = jax.random.normal(
        k1, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
    )
    print_mem_usage()
    v_pages = jax.random.normal(
        k2, (num_kv_heads, total_pages, page_size, head_dim), dtype=dtype
    )
  page_indices = jnp.arange(batch_size * pages_per_sequence, dtype=jnp.int32)
  page_indices = jax.random.permutation(k3, page_indices, independent=True)
  page_indices = page_indices.reshape(batch_size, pages_per_sequence)
  q = jax.random.normal(
      k4, (batch_size, num_heads, head_dim), dtype=dtype
  )
  return q, k_pages, v_pages, page_indices

def test_reconstruct_kv():
  batch_size = 16
  seq_lens = np.array(
      [4096 * (i + 1) for i in range(batch_size)],
      dtype=np.int32,
  )
  page_size = 512
  max_seq_len = max(seq_lens)
  num_heads, head_dim = 8, 128
  q, k_pages, v_pages, page_indices = generate_qkv_decode(
      seq_lens,
      page_size,
      max_seq_len,
      num_heads,
      head_dim,
      jax.random.PRNGKey(0),
      mqa=False,
      num_kv_heads=num_heads // 2
  )
  k = reconstruct_kv(page_indices, k_pages)
  v = reconstruct_kv(page_indices, v_pages)


multi_page_grouped_query_attention_fully_pipelined = (
    paged_attention
)


@functools.partial(jax.jit, static_argnames=["mask_value"])
def grouped_query_attention_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Vanilla attention GQA implementation for reference.
  Args:
    q: A [batch_size, num_heads, head_dim] jax.Array.
    k: A [batch_size, num_kv_heads, max_seq_len, head_dim] jax.Array.
    v: A [batch_size, num_kv_heads, max_seq_len, head_dim] jax.Array.
    lengths: A i32[batch_size] jax.Array.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
  Returns:
    The output of attention([batch_size, num_heads, head_dim]), along with the
    max logit ([batch_size, num_heads]) and softmax denominator ([batch_size,
    num_heads]).
  """
  batch_size, num_heads, head_dim = q.shape
  _, num_kv_heads, max_seq_len, _ = k.shape
  assert k.shape == v.shape
  assert num_heads % num_kv_heads == 0
  q = q.reshape(batch_size, num_kv_heads, num_heads // num_kv_heads, head_dim)
  logits = jnp.einsum(
      "bhgd,bhtd->bhgt", q.astype(jnp.float32), k.astype(jnp.float32)
  )
  mask = jnp.arange(max_seq_len)[None] < lengths[:, None]
  logits = logits + jnp.where(mask, 0.0, mask_value)[:, None, None, :]
  logits_max = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - logits_max[..., None])
  denominator = unnormalized.sum(axis=-1)
  o = (
      jnp.einsum("bhgt,bhtd->bhgd", unnormalized.astype(v.dtype), v)
      / denominator[..., None]
  )
  return o.reshape(batch_size, num_heads, head_dim), (logits_max, denominator)

def test_multi_page_grouped_query_attention(
    q_kv_head_ratio: int, dtype=jnp.float32
):
  batch_size = 16
  seq_lens = [2048 for _ in range(batch_size)]
  seq_lens[-1] = 4096 * 4
  seq_lens = np.array(seq_lens, dtype=np.int32)
  max_seq_len = max(seq_lens)
  page_size = 32
  block_size = 2048
  num_kv_heads = 8
  head_dim = 128
  num_heads = num_kv_heads * q_kv_head_ratio
  q, k_pages, v_pages, page_indices = generate_qkv_decode(
      seq_lens,
      page_size,
      max_seq_len,
      num_heads,
      head_dim,
      jax.random.PRNGKey(0),
      mqa=False,
      num_kv_heads=num_kv_heads,
      dtype=dtype,
  )
  k = reconstruct_kv(page_indices, k_pages)
  v = reconstruct_kv(page_indices, v_pages)

  def run():
    o_pipelined = (
      multi_page_grouped_query_attention_fully_pipelined(
            q,
            k_pages,
            v_pages,
            seq_lens,
            page_indices,
            pages_per_compute_block=block_size // page_size,
        )
    )
    o_ref, _ = grouped_query_attention_reference(
        q, k, v, seq_lens
    )
    if q_kv_head_ratio == 1:
      # TODO(dinghua): investigate why the abs diff is large.
      np.testing.assert_allclose(o_pipelined, o_ref, atol=2e-1, rtol=3e-3)
    else:
      atol = 1e-2 if dtype == jnp.bfloat16 else 5e-3
      np.testing.assert_allclose(o_pipelined, o_ref, atol=atol, rtol=1e-3)

  run()  # warm up
  #run()

def shard_kv_heads(
    paged_attention_impl: Callable[..., Any],
    mesh: jax.sharding.Mesh,
    kv_head_mesh_axis_name: str,
):
  """Shards GQA PagedAttention along KV heads."""
  in_specs = (
      P(None, kv_head_mesh_axis_name, None),  # q
      P(kv_head_mesh_axis_name, None, None, None),  # k
      P(kv_head_mesh_axis_name, None, None, None),  # v
      P(),  # lengths
      P(),  # page_indices
  )

  out_specs = P(None, kv_head_mesh_axis_name, None)  # q

  return jax.jit(
      shard_map.shard_map(
          paged_attention_impl,
          mesh=mesh,
          in_specs=in_specs,
          out_specs=out_specs,
          check_rep=False,
      )
  )

def test_sharded_multi_page_grouped_query_attention( q_kv_head_ratio: int):
    batch_size = 16
    # seq_lens = np.array(
    #     [4096 * (i + 1) for i in range(batch_size)], dtype=np.int32
    # )
    seq_lens = [2048 for _ in range(batch_size)]
    seq_lens[-1] = 4096 * 4
    seq_lens = np.array(seq_lens, dtype=np.int32)
    max_seq_len = max(seq_lens)
    page_size = 32
    block_size = 2048
    num_heads, num_kv_heads, head_dim = 8 * q_kv_head_ratio, 8, 128
    q, k_pages, v_pages, page_indices = generate_qkv_decode(
        seq_lens,
        page_size,
        max_seq_len,
        num_heads,
        head_dim,
        jax.random.PRNGKey(0),
        mqa=False,
        num_kv_heads=num_kv_heads,
    )

    print(f"mesh shape:{mesh.shape}")
    q_pspec = jax.sharding.NamedSharding(mesh, P(None, 'kv_head', None))
    kv_pspec = jax.sharding.NamedSharding(mesh, P('kv_head', None, None, None))
    q_sharded = jax.device_put(q, q_pspec)
    k_pages_sharded = jax.device_put(k_pages, kv_pspec)
    v_pages_sharded = jax.device_put(v_pages, kv_pspec)

    paged_attention_impl = functools.partial(
        multi_page_grouped_query_attention_fully_pipelined,
        pages_per_compute_block=block_size // page_size,
    )
    sharded_paged_attention_impl = shard_kv_heads(
        paged_attention_impl,
        mesh,
        kv_head_mesh_axis_name='kv_head',
    )

    def run():
      o = paged_attention_impl(
          q,
          k_pages,
          v_pages,
          seq_lens,
          page_indices,
      )
      o_sharded = sharded_paged_attention_impl(
          q_sharded,
          k_pages_sharded,
          v_pages_sharded,
          seq_lens,
          page_indices,
      )
      if q_kv_head_ratio > 1:
        atol, rtol = 1e-2, 2e-2
      else:
        atol, rtol = 1e-1, 1e-1
      np.testing.assert_allclose(
          o[np.where(seq_lens > 0)].astype(jnp.float32),
          o_sharded[np.where(seq_lens > 0)].astype(jnp.float32),
          atol=atol,
          rtol=rtol,
      )
      return o_sharded

    with mesh:
      return run()  # warm up

mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=('kv_head',))

def next_tokens(logits):
  return jnp.argmax(logits, axis=-1)

token_sharding = jax.sharding.NamedSharding(mesh, P())
next_tokens = jax.jit(next_tokens, out_shardings=token_sharding)

def print_mem_usage():
  """Print current mem usage"""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(
        f"memory using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}"
    )  
devvices = jax.local_devices()
print_mem_usage()
#test_multi_page_grouped_query_attention(2, dtype=jnp.bfloat16)  
logits = test_sharded_multi_page_grouped_query_attention(2)
tokens = next_tokens(logits)
print(tokens)