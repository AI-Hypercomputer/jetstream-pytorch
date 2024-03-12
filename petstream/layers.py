# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

import math
from typing import Any, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import jax


class Int8Embedding(torch.nn.Module):

  def __init__(self, num_embeddings, embedding_dims, device='cpu'):
    super().__init__()
    table = torch.ones(
      (num_embeddings, embedding_dims), device=device,
      dtype=torch.int8
    )
    self.register_buffer('weight', table)
    embedding_scaler = torch.ones(
      (embedding_dims,), device=device,
      dtype=torch.bfloat16
    )
    self.register_buffer('weight_scaler', embedding_scaler)
  
  def forward(self, input):
    return F.embedding(input, self.weight) * self.weight_scaler


class WeightOnlyInt8Linear(torch.nn.Module):

  def __init__(self, in_features, out_features, bias, device):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features

    weight = torch.ones((out_features, in_features), 
      dtype=torch.int8, device=device)
    self.register_buffer('weight', weight)

    weight_scaler = torch.ones((out_features, ), 
      dtype=torch.bfloat16, device=device)
    self.register_buffer('weight_scaler', weight_scaler)
    
    # if bias:
    #   self.bias = torch.nn.Parameter(
    #     torch.zeros((out_features, ), 
    #     dtype=torch.bfloat16, device=device))
    # else:
    #   self.register_parameter('bias', None)

  def forward(self, inputs):
    return F.linear(inputs, self.weight) * self.weight_scaler


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

  def __init__(self, args, env):
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

    self.env = env

    LinearLayer = WeightOnlyInt8Linear if args.quantize else nn.Linear


    self.wq = LinearLayer(
        args.dim,
        args.n_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wk = LinearLayer(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wv = LinearLayer(
        args.dim,
        self.n_kv_heads * self.head_dim,
        bias=False,
        device=args.device,
    )
    self.wo = LinearLayer(
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
      cache,
  ):
    # bsz, seqlen, _ = x.shape
    with jax.named_scope('attn_linear_before_cache'):
      bsz, seqlen = x.shape[0], x.shape[-2]

      # qkv fuse
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
      xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
      xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
      xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

    with jax.named_scope('attn_rope'):
      xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    with jax.named_scope('attn_insert_cache'):
      xk = xk.transpose(-3, -2)
      xv = xv.transpose(-3, -2)
      cache_k, cache_v = cache.update(xk, xv)

      keys = repeat_kv(
          cache_k, self.n_rep
      )  # (bs, n_local_heads, seqlen, head_dim)
      values = repeat_kv(
          cache_v, self.n_rep
      )  # (bs, n_local_heads, seqlen, head_dim)

    with jax.named_scope('attn_mat1'):
      ## Attention start
      scores = torch.einsum("ijkl,ikml->ikjm", xq, keys) / math.sqrt(
          self.head_dim
      )
      if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, max_seqlen)
    with jax.named_scope('attn_soft'):
      scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    with jax.named_scope('attn_mat2'):
      output = torch.einsum(
          "ikjm,ikml->ikjl", scores, values
      )  # (bs, n_local_heads, seqlen, head_dim)
      output = output.transpose(-3, -2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)

