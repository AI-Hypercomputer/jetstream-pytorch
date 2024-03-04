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

