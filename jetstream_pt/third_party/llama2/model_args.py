# pylint: disable-all
"""The original Llama2 model."""

import dataclasses
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass
class ModelArgs:
  """Model configuration parameters."""

  dim: int = -1
  n_layers: int = -1
  n_heads: int = -1
  n_kv_heads: Optional[int] = None
  vocab_size: int = -1  # defined later by tokenizer
  multiple_of: int = (
      256  # make SwiGLU hidden layer size multiple of large power of 2
  )
  ffn_dim_multiplier: Optional[float] = None
  norm_eps: float = 1e-5

  max_batch_size: int = -1
  max_seq_len: int = -1

  bf16_enable: bool = False
  head_dim = -1
  infer_length = 0
  device = "cpu"
  quantize = False


def get_arg(
    param_size: str,
    seqlen,
    batch_size,
    vocab_size: int,
    bf16_enable: bool = False,
) -> ModelArgs:
  """Gets model args."""

  data = {}
  if param_size == "tiny":
    data = {
        "dim": 128,
        "multiple_of": 32,
        "n_heads": 8,
        "n_layers": 3,
        "norm_eps": 1e-05,
    }
  elif param_size == "7b":
    data = {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-05,
    }
  elif param_size == "13b":
    data = {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-05,
    }
  elif param_size == "70b":
    data = {
        "dim": 8192,
        "multiple_of": 4096,
        "ffn_dim_multiplier": 1.3,
        "n_heads": 64,
        "n_kv_heads": 8,
        "n_layers": 80,
        "norm_eps": 1e-05,
    }
  return ModelArgs(
      max_seq_len=seqlen,
      max_batch_size=batch_size,
      vocab_size=vocab_size,
      bf16_enable=bf16_enable,
      **data,
  )


def get_model_args(
    param_size, context_length, batch_size, vocab_size, bf16_enable
):
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
