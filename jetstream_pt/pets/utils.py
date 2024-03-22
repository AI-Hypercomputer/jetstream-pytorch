# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The util file useful to all the models."""


import torch
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Any

def p2n(t):
  if isinstance(t, torch.Tensor):
    if t.dtype == torch.bfloat16:
      # Numpy doesn't have bf16 support. Convert to f32 as intermediate step.
      t = t.to(torch.float32).detach().cpu().numpy()
    else:
      t = t.detach().cpu().numpy()
  return t


def n2jtype(t: np.ndarray):
  """Converts a numpy data type to jax data type."""

  d = jnp.float32
  if t.dtype == np.float32:
    d = jnp.bfloat16
  elif t.dtype == np.int32:
    d = jnp.int32
  elif t.dtype == np.int64:
    d = jnp.int64
  elif t.dtype == np.complex64:
    d = jnp.complex64
  return d


def load_checkpoint(checkpoint_dir: str) -> Any:
  if checkpoint_dir:
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
    assert len(checkpoints) == 1, 'currently only support one file'
    # Need to merge the checkpoint to 1 file.
    checkpoint = torch.load(checkpoints[0])
    return checkpoint
  return None
