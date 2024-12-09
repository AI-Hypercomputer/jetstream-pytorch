# pylint: disable-all
"""The original Llama2 model."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class RopeScalingArgs:
  """Rope scaling configuration parameters."""

  factor: float = 8.0
  low_freq_factor: float = 1.0
  high_freq_factor: float = 4.0
  original_max_position_embeddings: int = 8192


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

  rope_theta: float = 10000.0
  rope_scaling_args: RopeScalingArgs = None


def get_arg(
    model_name: str,
    seqlen,
    batch_size,
    bf16_enable: bool = False,
) -> ModelArgs:
  """Gets model args."""

  data = {}
  if model_name == "llama-2-tiny":
    data = {
        "dim": 128,
        "vocab_size": 32000,
        "multiple_of": 32,
        "n_heads": 64,
        "n_kv_heads": 8,
        "n_layers": 3,
        "norm_eps": 1e-05,
    }
  elif model_name == "llama-2-7b":
    data = {
        "dim": 4096,
        "vocab_size": 32000,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-05,
    }
  elif model_name == "llama-2-13b":
    data = {
        "dim": 5120,
        "vocab_size": 32000,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-05,
    }
  elif model_name == "llama-2-70b":
    data = {
        "dim": 8192,
        "vocab_size": 32000,
        "multiple_of": 4096,
        "ffn_dim_multiplier": 1.3,
        "n_heads": 64,
        "n_kv_heads": 8,
        "n_layers": 80,
        "norm_eps": 1e-05,
    }
  elif model_name == "llama-3-8b":
    data = {
        "dim": 4096,
        "vocab_size": 128256,
        "multiple_of": 1024,
        "ffn_dim_multiplier": 1.3,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        "rope_theta": 500000.0,
    }
  elif model_name == "llama-3-70b":
    data = {
        "dim": 8192,
        "ffn_dim_multiplier": 1.3,
        "multiple_of": 4096,
        "n_heads": 64,
        "n_kv_heads": 8,
        "n_layers": 80,
        "norm_eps": 1e-05,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
    }

  return ModelArgs(
      max_seq_len=seqlen,
      max_batch_size=batch_size,
      bf16_enable=bf16_enable,
      **data,
  )


def get_model_args(model_name, context_length, batch_size, bf16_enable):
  model_args = get_arg(
      model_name=model_name,
      seqlen=context_length,
      batch_size=batch_size,
      bf16_enable=bf16_enable,
  )
  model_args.n_kv_heads = (
      model_args.n_heads
      if model_args.n_kv_heads is None
      else model_args.n_kv_heads
  )
  model_args.head_dim = model_args.dim // model_args.n_heads
  return model_args
