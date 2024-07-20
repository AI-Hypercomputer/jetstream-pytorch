# pylint: disable-all
"""This version contains modification to make it easier to trace and support batch."""

from typing import Any, List, Optional

import jax
import torch
import torch.nn.functional as F
from jetstream_pt.layers import (
    Attention,
    Int8Embedding,
    RMSNorm,
    WeightOnlyBlockwiseQuantizedLinear,
    WeightOnlyPerChannelQuantizedLinear,
    get_quantized_enbedding_layer,
    get_quantized_linear_layer,
)
from torch import nn

from . import model_args


class FeedForward(nn.Module):
  """Feed-forward module."""

  def __init__(
      self,
      dim: int,
      hidden_dim: int,
      multiple_of: int,
      ffn_dim_multiplier: Optional[float],
      device="meta",
      env=None,
  ):
    super().__init__()
    self.env = env
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
      hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.w1 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    self.w2 = LinearLayer(
        hidden_dim,
        dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )
    self.w3 = LinearLayer(
        dim,
        hidden_dim,
        bias=False,
        device=device,
        **linear_kwargs,
    )

  def forward(self, x):
    result = self.w2(F.silu(self.w1(x)) * self.w3(x))
    return result


class TransformerBlock(nn.Module):
  """Transformer block."""

  def __init__(
      self,
      layer_id: int,
      args: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.n_heads = args.n_heads
    self.dim = args.dim
    self.head_dim = args.dim // args.n_heads

    self.attention = Attention(
        args.n_heads,
        args.n_kv_heads or args.n_heads,
        args.dim // args.n_heads,
        args.dim,
        env=env,
        device=args.device,
        layer_id=layer_id,
    )
    self.feed_forward = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
        device=args.device,
        env=env,
    )
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(
        args.dim, eps=args.norm_eps, device=args.device
    )
    self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, device=args.device)

  def forward(
      self,
      x: torch.Tensor,
      freqs_cis: torch.Tensor,
      mask: Optional[torch.Tensor],
      cache,
      start=None,
      end=None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ):
    with jax.named_scope("Attention"):
      attn = self.attention.forward(
          self.attention_norm(x),
          freqs_cis,
          mask,
          cache,
          start,
          end,
          ragged_batch_index,
          ragged_block_index,
      )
    with jax.named_scope("ffn_norm"):
      h = x + attn
      ffns = self.ffn_norm(h)

    with jax.named_scope("ffn"):
      out = h + self.feed_forward.forward(ffns)
      return out


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> torch.Tensor:
  freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
  t = torch.arange(end, device=freqs.device)  # type: ignore
  freqs = torch.outer(t, freqs).float()  # type: ignore
  freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
  return freqs_cis


class Transformer(nn.Module):
  """Transformer module."""

  def __init__(
      self,
      params: model_args.ModelArgs,
      env,
  ):
    super().__init__()
    self.env = env
    self.params = params
    self.vocab_size = params.vocab_size
    self.n_layers = params.n_layers

    Embedding = get_quantized_enbedding_layer(env.quant_config)
    self.tok_embeddings = Embedding(
        params.vocab_size,
        params.dim,
        device=params.device,
    )

    self.layers = torch.nn.ModuleList()
    for layer_id in range(params.n_layers):
      self.layers.append(TransformerBlock(layer_id, params, env))
    self.norm = RMSNorm(params.dim, eps=params.norm_eps, device=params.device)

    LinearLayer = get_quantized_linear_layer(env.quant_config)
    linear_kwargs = {}
    if LinearLayer != torch.nn.Linear:
      linear_kwargs["quant_config"] = env.quant_config

    self.output = LinearLayer(
        params.dim,
        params.vocab_size,
        bias=False,
        device=params.device,
        **linear_kwargs,
    )
    # TODO what to do with this
    freqs_cis = precompute_freqs_cis(
        self.params.dim // self.params.n_heads,
        self.params.max_seq_len * 2,
        theta=self.params.rope_theta,
    )

    self.register_buffer("freqs_cis", freqs_cis)

  @torch.no_grad()
  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[Any],
      mask,
      start=None,
      ragged_batch_index=None,
      ragged_block_index=None,
  ):
    """
    tokens: the input token for decoding
    input_pos: the decoding position relative to the start, which is the length of the decoding results
    caches: kv caches
    mask: causal mask to filter the attention results
    start: the starting position for each slot
    ragged_batch_index: precomputed batch index for ragged attention
    ragged_block_index: precomputed block index for ragged attention
    """
    with jax.named_scope("transformer_tok"):
      seqlen = tokens.shape[-1]
      h = self.tok_embeddings(tokens)

    with jax.named_scope("transformer_freq"):
      bsz, seqlen = tokens.shape
      freqs_cis = self.freqs_cis[input_pos]
      freqs_cis = freqs_cis.reshape(bsz, seqlen, -1)

    end = None if start is None else (start + input_pos) % self.env.cache_len
    # For stacked case, cannot get cache inside the loop which will cause cache copy
    for layer_id, layer in enumerate(self.layers):
      if caches[0].stacked:
        cache = caches[0]
      else:
        cache = caches[layer_id]
      # else:  # For stacked case, there is only 1 yer of kv cache

      with jax.named_scope("TransformerBlock_Layer_" + str(layer_id)):
        h = layer(
            h,
            freqs_cis,
            mask,
            cache,
            start,
            end,
            ragged_batch_index,
            ragged_block_index,
        )

    with jax.named_scope("transformer_norm"):
      h = self.norm(h)
      output = self.output(h).float()
    return output

  @staticmethod
  def get_quantized_linear_weight_to_scaler_map():
    return {
        "attention.wq.weight": "attention.wq.weight_scaler",
        "attention.wk.weight": "attention.wk.weight_scaler",
        "attention.wv.weight": "attention.wv.weight_scaler",
        "attention.wo.weight": "attention.wo.weight_scaler",
        "feed_forward.w1.weight": "feed_forward.w1.weight_scaler",
        "feed_forward.w2.weight": "feed_forward.w2.weight_scaler",
        "feed_forward.w3.weight": "feed_forward.w3.weight_scaler",
        "output.weight": "output.weight_scaler",
    }

  @staticmethod
  def get_quantized_embedding_weight_to_scaler_map():
    return {
        "tok_embeddings.weight": "tok_embeddings.weight_scaler",
    }

  @staticmethod
  def get_weight_sharding_type(model_name: str = ""):
    # ParallelEmbedding is col partitioned across the shards.
    # VocalParallelEmbedding is row partitioned across the shards.
    # ColumnParallelLinear is row partitioned across shards due to transpose.
    # RowParallelLinear is col partitioned across shards due to transpose.
    # None is no partitioning and tensor should be identical across shards
    expected_model_names = ("llama-2", "llama-3")
    assert (
        model_name in expected_model_names
    ), f"Expected model_name to one of {expected_model_names}"
    sharding_dict = {
        "rope.freqs": None,
        "attention.wq.weight": "ColumnParallelLinear",
        "attention.wk.weight": "ColumnParallelLinear",
        "attention.wv.weight": "ColumnParallelLinear",
        "attention.wo.weight": "RowParallelLinear",
        "feed_forward.w1.weight": "ColumnParallelLinear",
        "feed_forward.w2.weight": "RowParallelLinear",
        "feed_forward.w3.weight": "ColumnParallelLinear",
        "attention_norm.weight": None,
        "ffn_norm.weight": None,
        "norm.weight": None,
        "output.weight": "ColumnParallelLinear",
    }
    if model_name == "llama-2":
      sharding_dict["tok_embeddings.weight"] = "ParallelEmbedding"
    elif model_name == "llama-3":
      sharding_dict["tok_embeddings.weight"] = "VocabParallelEmbedding"
    return sharding_dict
