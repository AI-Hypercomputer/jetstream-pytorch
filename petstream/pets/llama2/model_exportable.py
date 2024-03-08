# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from . import model_args 
import jax.sharding as jsharding
from jax.experimental import mesh_utils
import jax

hanq_flag = False

def make_replicated_sharding():
  P = jsharding.PartitionSpec
  num_devices = len(jax.devices())
  mesh = jsharding.Mesh(
      mesh_utils.create_device_mesh((num_devices, 1)),
      axis_names=("x", "y"),
  )
  replicated = jsharding.NamedSharding(mesh, P())
  return replicated


class RMSNorm(torch.nn.Module):
  """RMSNorm module."""

  def __init__(self, dim: int, eps: float = 1e-6, device='meta'):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim, device=device))

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    output = self._norm(x.float()).type_as(x)
    return output * self.weight


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
  ndim = x.ndim
  assert 1 < ndim
  assert freqs_cis.shape == (x.shape[-3], x.shape[-1]), x.shape
  shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
  return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  # bs, seqlen, heads, dim
  xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
  xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
  freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
  xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
  xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
  return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
  """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""

  bs, n_kv_heads, slen, head_dim = x.shape
  if n_rep == 1:
    return x
  return (
      x[:, :, None, :, :]
      .expand(bs, n_kv_heads, n_rep, slen, head_dim)
      .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
  )


class Attention(nn.Module):
  """Attention module."""

  def __init__(self, args: model_args.ModelArgs):
    super().__init__()

    self.n_kv_heads = (
        args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    )
    self.n_local_heads = args.n_heads
    self.n_local_kv_heads = self.n_kv_heads
    self.n_rep = self.n_local_heads // self.n_local_kv_heads
    self.head_dim = args.dim // args.n_heads
    self.max_seq_len = args.max_seq_len
    self.n_heads = args.n_heads

    self.wq = nn.Linear(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wk = nn.Linear(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wv = nn.Linear(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wo = nn.Linear(
        args.n_heads * self.head_dim,
        args.dim,
        bias=False,
        device=args.device,
    )

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      prefill: bool,
      input_indexes: torch.Tensor,
      cache_indexes: torch.Tensor,
      cache_k,
      cache_v,
  ):
    # bsz, seqlen, _ = x.shape
    bsz, seqlen = x.shape[0], x.shape[-2]
    # qkv fuse
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    input_indexes = input_indexes.to(torch.int64)

    xk = xk.transpose(-3, -2)
    xv = xv.transpose(-3, -2)
    ###
    if prefill:
      # Assumes prefill only
      cache_k = xk
      cache_v = xv
    else:
      xk = torch.broadcast_to(
          xk, (bsz, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
      )
      xv = torch.broadcast_to(
          xv, (bsz, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
      )
      iota = (
          torch.arange(self.max_seq_len).reshape(1, 1, self.max_seq_len, 1)
          .expand(bsz, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
      )
      if hanq_flag:
        iota = torch_xla2.call_jax(
          jax.lax.with_sharding_constraint,
          iota, 
          make_replicated_sharding()
        )

      cache_k = torch.where(iota == input_indexes.expand(self.max_seq_len).reshape(1, 1, self.max_seq_len, 1), xk, cache_k)
      cache_v = torch.where(iota == input_indexes.expand(self.max_seq_len).reshape(1, 1, self.max_seq_len, 1), xv, cache_v)

    keys = repeat_kv(
        cache_k, self.n_rep
    )  # (bs, n_local_heads, seqlen, head_dim)
    values = repeat_kv(
        cache_v, self.n_rep
    )  # (bs, n_local_heads, seqlen, head_dim)
    # (b, i, j) x (b, j, k) -> (b, i, k)
    scores = torch.einsum("ijkl,ikml->ikjm", xq, keys) / math.sqrt(
        self.head_dim
    )
    if mask is not None:
      scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.einsum(
        "ikjm,ikml->ikjl", scores, values
    )  # (bs, n_local_heads, seqlen, head_dim)
    output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output), cache_k, cache_v


class FeedForward(nn.Module):
  """Feed-forward module."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
      device = 'meta',
  ):
    super().__init__()
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    self.w1 = nn.Linear(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )
    self.w2 = nn.Linear(
        hidden_dim,
        dim,
        bias=False,
        device=device,
    )
    self.w3 = nn.Linear(
        dim,
        hidden_dim,
        bias=False,
        device=device,
    )

  def forward(self, x):
    return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
  """Transformer block."""

  def __init__(
      self,
      layer_id: int,
      args: model_args.ModelArgs,
      groups: Optional[List[Any]] = None,
  ):
    super().__init__()
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads

    self.attention = Attention(
        args,
    )
    self.feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        device=args.device
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      prefill: bool,
      input_indexes: torch.Tensor,
      cache_indexes,
      cache_k,
      cache_v,
  ):
    attn, xk, xv = self.attention.forward(
        self.attention_norm(x),
        freqs_cis,
        mask,
        prefill,
        input_indexes,
        cache_indexes,
        cache_k,
        cache_v,
    )
    h = x + attn
    out = h + self.feed_forward.forward(self.ffn_norm(h))
    return out, xk, xv


class Transformer(nn.Module):
  """Transformer module."""

  def __init__(
      self,
      params: model_args.ModelArgs,
      world_size: Optional[int] = None,
      rank: Optional[int] = None,
      groups: Optional[List[Any]] = None,
  ):
    super().__init__()
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    self.tok_embeddings = nn.Embedding(
        params.vocab_size,
        params.dim,
        device=params.device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(
          TransformerBlock(
              layer_id,
              params,
          )
      )

    self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)
    self.output = nn.Linear(
        params.dim,
        params.vocab_size,
        bias=False,
        device=params.device,
    )

    # TODO what to do with this
    freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
    )

    self.register_buffer("freqs_cis", freqs_cis)

    mask = torch.full(
        (1, 1, self.params.max_seq_len, self.params.max_seq_len), float("-inf")
    ).to(torch.bfloat16 if self.params.bf16_enable else torch.float)

    mask = torch.triu(mask, diagonal=1)
    self.register_buffer("mask", mask)

  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      input_indexes: torch.Tensor,
      cache_indexes,
      caches: List[Tuple[torch.tensor, Any]],
      prefill,
  ):
    seqlen = tokens.shape[-1]
    h = self.tok_embeddings(tokens)
    freqs_cis = self.freqs_cis.index_select(0, input_indexes)
    mask = self.mask if prefill else None

    new_caches = []
    for layer, (cache_k, cache_v) in zip(self.layers, caches):
      h, new_k, new_v = layer(
          h,
          freqs_cis,
          mask,
          prefill,
          input_indexes,
          cache_indexes,
          cache_k,
          cache_v,
      )
      new_caches.append((new_k, new_v))
    h = self.norm(h)
    output = self.output(h).float()
    return output, new_caches
