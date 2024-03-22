# pylint: disable-all
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""Llama2 model args. forked from: https://github.com/meta-llama/llama/blob/main/llama/model.py """

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
  device = 'cpu'
  quantize = False

