# pylint: disable-all
"""The original Gemma model."""

import dataclasses
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelArgs:
  """Model configuration parameters."""

  # The number of tokens in the vocabulary.
  vocab_size: int = 256000
  # The maximum sequence length that this model might ever be used with.
  max_position_embeddings: int = 8192
  # The number of blocks in the model.
  num_hidden_layers: int = 28
  # The number of attention heads used in the attention layers of the model.
  num_attention_heads: int = 16
  # The number of key-value heads for implementing attention.
  num_key_value_heads: int = 16
  # The hidden size of the model.
  hidden_size: int = 3072
  # The dimension of the MLP representations.
  intermediate_size: int = 24576
  # The number of head dimensions.
  head_dim: int = 256
  # The epsilon used by the rms normalization layers.
  rms_norm_eps: float = 1e-6
  # The dtype of the weights.
  dtype: str = 'bfloat16'
  # Whether a quantized version of the model is used.
  quant: bool = False
  # The path to the model tokenizer.
  tokenizer: Optional[str] = 'tokenizer/tokenizer.model'

def get_arg(
    param_size: str,
    seqlen,
    batch_size,
    vocab_size: int,
    bf16_enable: bool = False,
) -> ModelArgs:
  """Gets model args."""

  data = {}
  if param_size == "2b":
    data = {
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "hidden_size": 2048,
        "intermediate_size": 16384
    }
  elif param_size == "7b":
    data = {
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "hidden_size": 3072,
        "intermediate_size": 24576
    }
  return ModelArgs(
      max_seq_len=seqlen,
      max_batch_size=batch_size,
      vocab_size=vocab_size,
      bf16_enable=bf16_enable,
      **data,
  )

def get_model_args(param_size, context_length, batch_size, vocab_size, bf16_enable):
    model_args = get_arg(
        param_size=param_size,
        seqlen=context_length,
        batch_size=batch_size,
        vocab_size=vocab_size,
        bf16_enable=bf16_enable,
    )
    model_args.n_kv_heads = (
        model_args.n_heads
        if model_args.n_kv_heads is None
        else model_args.n_kv_heads
    )
    model_args.head_dim = model_args.dim // model_args.n_heads
    return model_args