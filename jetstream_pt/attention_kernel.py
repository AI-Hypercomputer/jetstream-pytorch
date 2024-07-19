import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map

import torch
import torch.nn.functional as F

import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def ragged_flash_attention_kernel(
    layer_ref,
    start_ref,
    end_ref,
    line_end_ref,
    pre_b_ref,
    pre_i_ref,
    q_ref,
    k_ref,
    v_ref,
    k_scaler_ref,
    v_scaler_ref,
    o_ref,  # outputs
    m_ref,  # row max
    l_ref,  # propogation coefficient
    bk: int,
    mask_value: float,
    normalize_var: bool,
    quantized: bool,
):
  """Pallas kernel for flash attention."""
  with jax.named_scope("attention_kernel"):
    b, i = pl.program_id(0), pl.program_id(1)

    @pl.when(i == 0)
    def init():
      with jax.named_scope("init"):
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        o_ref[...] = jnp.zeros_like(o_ref)

    length = line_end_ref[b]
    start = start_ref[b]
    end = end_ref[b]

    @pl.when(jnp.logical_and(i * bk < length, start != end))
    def run():
      with jax.named_scope("run_qk"):
        q = q_ref[...].astype(jnp.float32)
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...].astype(jnp.float32)
        m_prev, l_prev = m_ref[...], l_ref[...]

        qk = jax.lax.dot_general(
            q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
        )
        if normalize_var:
          qk = qk / jnp.sqrt(k.shape[-1])
        if quantized:
          qk = qk * k_scaler_ref[...]
      with jax.named_scope("run_mask"):
        start = start_ref[b]
        end = end_ref[b]
        iota = jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
        mask_start_lt_end = jnp.logical_and(
            i * bk + iota >= start, i * bk + iota < end
        ).astype(jnp.int32)
        mask_start_gt_end = jnp.logical_or(
            i * bk + iota >= start, i * bk + iota < end
        ).astype(jnp.int32)
        # mask = jax.lax.cond(start <= end, lambda: mask_start_lt_end, lambda: mask_start_gt_end)
        mask = jnp.where(start <= end, mask_start_lt_end, mask_start_gt_end)

        qk = qk + jnp.where(mask, 0.0, mask_value)

      with jax.named_scope("run_softmax"):
        m_curr = qk.max(axis=-1)

        s_curr = jnp.exp(qk - m_curr[..., None])

        l_curr = jax.lax.broadcast_in_dim(
            s_curr.sum(axis=-1), l_prev.shape, (0,)
        )
        if quantized:
          s_curr = s_curr * v_scaler_ref[...]
        o_curr_times_l_curr = jnp.dot(s_curr, v)
        m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
        m_next = jnp.maximum(m_prev, m_curr)
        alpha = jnp.exp(m_prev - m_next)
        beta = jnp.exp(m_curr - m_next)
        l_next = alpha * l_prev + beta * l_curr
        l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

        m_ref[...], l_ref[...] = m_next, l_next_safe
        o_ref[...] = (
            (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr)
            / l_next_safe
        ).astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "bk",
        "mask_value",
        "normalize_var",
        "testing",
        "quantized",
    ],
)
def ragged_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    layer,
    start: jax.Array,
    end: jax.Array,
    ragged_batch_index=None,
    ragged_block_index=None,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
    testing: bool = False,
    quantized: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  with jax.named_scope("ragged_mqa"):
    batch_size, time, head_dim = q.shape
    seq_len = k.shape[-2]

    stacked = False
    if k.ndim == 5:
      stacked = True

    def kv_index_map(
        b,
        i,
        layer_ref,
        start_ref,
        end_ref,
        line_end_ref,
        ragged_batch_index_ref,
        ragged_block_index_ref,
    ):
      index = b * (seq_len // bk) + i

      if stacked:
        return (
            layer_ref[0],
            ragged_batch_index_ref[index],
            ragged_block_index_ref[index],
            0,
        )
      return ragged_batch_index_ref[index], ragged_block_index_ref[index], 0

    def q_index_map(
        b,
        i,
        layer_ref,
        start_ref,
        end_ref,
        line_end_ref,
        ragged_batch_index_ref,
        ragged_block_index_ref,
    ):
      index = b * (seq_len // bk) + i
      if stacked:
        return layer_ref[0], ragged_batch_index_ref[index], 0, 0
      return ragged_batch_index_ref[index], 0, 0

    def scaler_index_map(b, i, layer_ref, *_):
      if stacked:
        return layer_ref[0], b, 0, i
      return b, 0, i

    line_end = jnp.where(start < end, end, seq_len - 1)

    if stacked:
      q_bp = (None, None, time, head_dim)
      kv_bp = (None, None, bk, head_dim)
      ks_bp = (None, None, 1, bk)
    else:
      q_bp = (None, time, head_dim)
      kv_bp = (None, bk, head_dim)
      ks_bp = (None, 1, bk)

    in_specs = [
        pl.BlockSpec(q_index_map, q_bp),
        pl.BlockSpec(kv_index_map, kv_bp),
        pl.BlockSpec(kv_index_map, kv_bp),
        pl.BlockSpec(scaler_index_map, ks_bp),
        pl.BlockSpec(scaler_index_map, ks_bp),
    ]
    inputs = (
        start,
        end,
        line_end,
        ragged_batch_index,
        ragged_block_index,
        q,
        k,
        v,
        k_scaler,
        v_scaler,
    )

    out, m, l = pl.pallas_call(
        functools.partial(
            ragged_flash_attention_kernel,
            bk=bk,
            mask_value=mask_value,
            normalize_var=normalize_var,
            quantized=quantized,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec(q_index_map, (None, time, head_dim)),
                pl.BlockSpec(q_index_map, (None, time, head_dim)),
                pl.BlockSpec(q_index_map, (None, time, head_dim)),
            ],
            grid=(batch_size, seq_len // bk),
        ),
        compiler_params={"dimension_semantics": ("parallel", "arbitrary")},
        interpret=testing,
        out_shape=[
            q,
            jax.ShapeDtypeStruct((batch_size, time, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, time, head_dim), jnp.float32),
        ],
    )(*inputs)
  return out, (m[..., 0], l[..., 0])


def ragged_mqa_kernel_reference(
    layer_ref,
    start_ref,
    end_ref,
    line_end_ref,
    pre_b_ref,
    pre_i_ref,
    q_ref,
    k_ref,
    v_ref,
    k_scaler_ref,
    v_scaler_ref,
    o_ref,
    m_ref,
    l_ref,
    bk: int,
    mask_value: float,
    normalize_var: bool,
    quantized: bool,
):
  """Pallas kernel for ragged attention."""
  b, i = pl.program_id(0), pl.program_id(1)
  del layer_ref

  @pl.when(i == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  # length = lengths_ref[b]
  # Always start from 0, left aligned
  length = end_ref[b]

  @pl.when(i * bk < length)
  def run():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m_prev, l_prev = m_ref[...], l_ref[...]

    qk = jax.lax.dot_general(
        q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )

    if normalize_var:
      qk = qk / math.sqrt(k.shape[-1])  # Align with meta llama
    # Quantized
    if quantized:
      qk = qk * k_scaler_ref[...]

    mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])

    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    # Quantized
    if quantized:
      s_curr = s_curr * v_scaler_ref[...]

    o_curr_times_l_curr = jnp.dot(s_curr, v)

    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

    m_ref[...], l_ref[...] = m_next, l_next_safe
    o_ref[...] = (
        (l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe
    ).astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "bk",
        "mask_value",
        "normalize_var",
        "testing",
        "quantized",
    ],
)
def ragged_mqa_reference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    layer,
    start: jax.Array,
    end: jax.Array,
    ragged_batch_index=None,
    ragged_block_index=None,
    k_scaler: jax.Array = None,
    v_scaler: jax.Array = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
    testing: bool = False,
    quantized: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi query attention."""
  batch_size, time, head_dim = q.shape
  # assert end.shape == (batch_size,)
  seq_len = k.shape[-2]

  stacked = False
  if k.ndim == 4:
    stacked = True

  def _compute_ragged_block_indices(b, i, lengths_ref):
    length = lengths_ref[b]
    not_done = i * bk < length
    am_last_batch = b == batch_size - 1
    # if length < bk, then it's -1, should be 0?
    last_good_block = jax.lax.div(length, bk) - 1

    # if not done, then still work on b, otherwise next batch
    b_next = jnp.where(not_done, b, jnp.where(am_last_batch, b, b + 1))
    # if not done, i next = i
    # if done
    # if last batch, previous good block
    # if not last batch, i next = 0
    i_next = jnp.where(
        not_done, i, jnp.where(am_last_batch, last_good_block, 0)
    )
    return b_next, i_next

  def kv_index_map(b, i, layer_ref, start_ref, end_ref, *_):
    b_next, i_next = _compute_ragged_block_indices(b, i, end_ref)
    if stacked:
      return layer_ref[0], b_next, i_next, 0
    return b_next, i_next, 0

  def kv_scale_index_map(b, i, layer_ref, start_ref, end_ref, *_):
    b_next, i_next = _compute_ragged_block_indices(b, i, end_ref)
    if stacked:
      return layer_ref[0], b_next, 0, i_next
    return b_next, 0, i_next

  if stacked:
    kv_bp = (None, None, bk, head_dim)
    ks_bp = (None, None, 1, bk)
  else:
    kv_bp = (None, bk, head_dim)
    ks_bp = (None, 1, bk)

  in_specs = [
      pl.BlockSpec(lambda b, i, *_: (b, 0, 0), (None, time, head_dim)),  # q
      pl.BlockSpec(kv_index_map, kv_bp),  # k
      pl.BlockSpec(kv_index_map, kv_bp),  # v
      pl.BlockSpec(kv_scale_index_map, ks_bp),  # k_scaler
      pl.BlockSpec(kv_scale_index_map, ks_bp),  # v_scaler
  ]

  inputs = (
      jnp.array([layer]),
      start,
      end,
      end,  # line_end, not actually used
      ragged_batch_index,
      ragged_block_index,
      q,
      k,
      v,
      k_scaler,
      v_scaler,
  )

  out, m, l = pl.pallas_call(
      functools.partial(
          ragged_mqa_kernel_reference,
          bk=bk,
          mask_value=mask_value,
          normalize_var=normalize_var,
          quantized=quantized,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=6,
          in_specs=in_specs,
          out_specs=[
              pl.BlockSpec(lambda b, *_: (b, 0, 0), (None, time, head_dim)),
              pl.BlockSpec(lambda b, *_: (b, 0, 0), (None, time, head_dim)),
              pl.BlockSpec(lambda b, *_: (b, 0, 0), (None, time, head_dim)),
          ],
          grid=(batch_size, seq_len // bk),
      ),
      interpret=testing,
      # debug=True,
      compiler_params={"dimension_semantics": ("parallel", "arbitrary")},
      out_shape=[
          q,
          jax.ShapeDtypeStruct((batch_size, time, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((batch_size, time, head_dim), jnp.float32),
      ],
  )(*inputs)
  return out, (m[..., 0], l[..., 0])


@functools.partial(
    jax.jit,
    static_argnames=[
        "bk",
        "mask_value",
        "normalize_var",
        "q_shard_axis",
        "kv_shard_axis",
        "testing",
    ],
)
def ragged_mha(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    layer,
    start: jax.Array,
    end: jax.Array,
    ragged_batch_index: jax.Array,
    ragged_block_index: jax.Array,
    k_scaler: jax.Array | None = None,
    v_scaler: jax.Array | None = None,
    bk: int = 512,
    mask_value: float = DEFAULT_MASK_VALUE,
    normalize_var: bool = True,
    q_shard_axis: int = 0,
    kv_shard_axis: int = 0,
    testing: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
  """Ragged multi head attention.
  Args:
    q: A [batch_size, compute_dim, num_heads, head_dim] jax.Array.
    k: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    v: A [batch_size, num_heads, seq_len, head_dim] jax.Array or
      PartitionQuantizedTensor.
    start: A i32[batch_size] jax.Array
    end: A i32[batch_size] jax.Array
    bk: An integer that is the sequence block size.
    logit_cap: An optional float that caps logits via tanh. By default there is
      no logit capping.
    mask_value: The value used for padding in attention. By default it is a very
      negative floating point number.
    out_dtype: An optional dtype for the output. If not provided, the output
      dtype will be q's dtype.
  Returns:
    The output of attention([batch_size, num_heads, compute_dim, head_dim]),
    along with the max logit ([batch_size, num_heads, compute_dim, 1]) and
    softmax denominator ([batch_size, num_heads, compute_dim, 1]).
  """
  mask_value = DEFAULT_MASK_VALUE
  bk = min(bk, k.shape[-2])
  bq, hq, tq, dq = q.shape
  dk = k.shape[-1]
  hkv = k.shape[-3]
  tk = k.shape[-2]

  assert k.shape[-1] == q.shape[-1]
  assert k.shape[-4] == q.shape[-4]

  rep = hq // hkv
  if rep > 1:
    q = q.reshape(bq, hkv, rep, tq, dq).reshape(bq, hkv, rep * tq, dq)
  stacked = True if k.ndim == 5 else False

  replicated_in_axes = 7
  if k_scaler is None:
    quantized = False
    if k.ndim == 5:
      kv_scale_shape = (k.shape[0], bq, 1, tk)
    else:
      kv_scale_shape = (bq, 1, tk)
    k_scale = jnp.ones(kv_scale_shape, dtype=jnp.bfloat16)
    v_scale = jnp.ones(kv_scale_shape, dtype=jnp.bfloat16)
  else:
    quantized = True
    k_scale = jnp.squeeze(k_scaler, -1)
    v_scale = jnp.squeeze(v_scaler, -1)

  if stacked:
    assert k_scale.shape == (k.shape[0], bq, 1, tk)
  else:
    assert k_scale.shape == (bq, 1, tk)

  replicated_inputs = (
      ragged_batch_index,
      ragged_block_index,
      k_scale,
      v_scale,
  )
  # New cache has t=1

  with jax.named_scope("ragged_mha_vmap"):
    out, (m, l) = jax.vmap(
        functools.partial(
            # ragged_mqa,
            ragged_mqa_reference,
            bk=bk,
            mask_value=mask_value,
            normalize_var=normalize_var,
            testing=testing,
            quantized=quantized,
            # out_dtype=out_dtype,
        ),
        in_axes=(
            q_shard_axis,
            kv_shard_axis,
            kv_shard_axis,
            *([None] * replicated_in_axes),
        ),
        out_axes=q_shard_axis,
    )(q, k, v, layer, start, end, *replicated_inputs)
  return out, (m, l)


def dense_attention(xq, keys, values, k_scaler=None, v_scaler=None, mask=None):
  """The vanilla attention kernel implementation."""

  bsz, _, _, head_dim = xq.shape
  with jax.named_scope("attn_mat1"):
    ## Attention start
    # scores = torch.einsum(jnp.einsum, "ijkl,ikml->ikjm", xq, keys) / math.sqrt(self.head_dim)
    scores = torch.einsum("ikjl,ikml->ikjm", xq, keys) / math.sqrt(head_dim)
    if k_scaler is not None:
      scores = scores * (k_scaler.reshape(bsz, 1, 1, keys.shape[2]))
    if mask is not None:
      # if mask.shape != (1,1,16,16):
      #   breakpoint()
      scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
  with jax.named_scope("attn_soft"):
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    if v_scaler is not None:
      scores = scores * v_scaler.reshape((bsz, 1, 1, keys.shape[2]))

  with jax.named_scope("attn_mat2"):
    # output = torch.einsum(
    #    "ikjm,ikml->ikjl", scores, values
    # )  # (bs, n_local_heads, seqlen, head_dim)
    output = torch.einsum("ikjm,ikml->ikjl", scores, values)
  return output


def flash_attention(
    xq,
    keys,
    values,
    layer,
    k_scaler=None,
    v_scaler=None,
    mask=None,
    normalize_var=True,
):
  mask_value: float = DEFAULT_MASK_VALUE
  if keys.ndim == 5:
    keys = keys[layer]
    values = values[layer]
    k_scaler = k_scaler[layer] if k_scaler is not None else None
    v_scaler = v_scaler[layer] if v_scaler is not None else None

  logits = torch.einsum(
      "bhqd,bhkd->bhqk", xq.type(torch.float32), keys.type(torch.float32)
  )

  if normalize_var:
    logits = logits / math.sqrt(keys.shape[-1])  # Align with meta llama
  # Quantized
  if k_scaler is not None:
    bs, hs, ls, ds = k_scaler.shape
    logits = logits * k_scaler.reshape(
        k_scaler.shape[-4], 1, 1, k_scaler.shape[-2]
    )

  # mask = jnp.arange(keys.shape[1])[None] < lengths[:, None]
  if mask is not None:
    # logits = logits + jnp.where(mask, 0.0, DEFAULT_MASK_VALUE)[:, None]
    logits = logits + mask

  logits_max, _ = torch.max(logits, axis=-1, keepdim=True)
  unnormalized = torch.exp(logits - logits_max)
  # Quantized, should not put here, otherwise sum will have this too, which cancels with denominator
  # unnormalized = unnormalized * v_scaler

  denominator = unnormalized.sum(axis=-1, keepdim=True)
  if v_scaler is not None:
    unnormalized = unnormalized * v_scaler.reshape(
        v_scaler.shape[-4], 1, 1, v_scaler.shape[-2]
    )
  o = (
      torch.einsum("bhqk,bhkd->bhqd", unnormalized.type_as(xq), values)
      / denominator
  )

  return o, (logits_max, denominator)


class RaggedAttentionKernel:
  """Ragged attention kernel."""

  def __init__(
      self, env, input_specs, output_specs, q_shard_axis, kv_shard_axis
  ):
    self.binded_ragged_mha = functools.partial(
        ragged_mha,
        bk=env.block_size,
        q_shard_axis=q_shard_axis,
        kv_shard_axis=kv_shard_axis,
        testing=env.testing,
    )
    self.binded_ragged_mha = shard_map(
        self.binded_ragged_mha,
        env.mesh,
        input_specs,
        output_specs,
        check_rep=False,
    )
    self.binded_ragged_mha = jax.jit(self.binded_ragged_mha)

  def __call__(
      self,
      xq,
      keys,
      values,
      layer,
      start,
      end,
      ragged_batch_index,
      ragged_block_index,
      k_scaler=None,
      v_scaler=None,
  ):
    return self.binded_ragged_mha(
        xq,
        keys,
        values,
        layer,
        start,
        end,
        ragged_batch_index,
        ragged_block_index,
        k_scaler,
        v_scaler,
    )
