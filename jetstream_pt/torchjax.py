"""This file will serve as proxy APIs for torch_xla2 API.

It serves 2 purposes:

1. torch_xla2 APIs are not 
   stable yet, and changes of it means lots of code edits throughout
   this repo. So future changes of torch_xla2 API we only need to edit
   this one file.

2. We can iterate API look and feel in this file and the influence 
   how it looks like in torch_xla2.
"""

import torch
from torch.utils import _pytree as pytree

import torch_xla2
import torch_xla2.interop

call_jax = torch_xla2.interop.call_jax
call_torch = torch_xla2.interop.call_torch


def to_torch(tensors):
  """Wrap a jax Array into XLATensor."""
  return torch_xla2.default_env().j2t_iso(tensors)


def from_torch_with_copy(tensors):
  """Convert torch tensor to Jax Array."""

  def convert_tensor(t):
    if isinstance(t, torch_xla2.tensor.XLATensor2):
      return t.jax()
    return torch_xla2.tensor.t2j(t)

  return pytree.tree_map_only(torch.Tensor, convert_tensor, tensors)


def from_torch(tensors):
  """Unwrap a XLATensor into jax Array.

  Will raise if passed in a torch.Tensor that is not XLATensor
  """
  return torch_xla2.default_env().t2j_iso(tensors)
