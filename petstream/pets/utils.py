"""The util file useful to all the models."""


import torch
import jax.numpy as jnp
import numpy as np


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
