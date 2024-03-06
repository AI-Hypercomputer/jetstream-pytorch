"""Import llama2 model."""
import os
from typing import Any
from absl import logging
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
import jax.sharding as jsharding
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch_xla2.export import exported_program_to_jax
from .llama2 import model_args
from pathlib import Path


def flatten_state_dict(state_dict):
  i = 0
  res = []
  name_to_pos = {}
  for name, val in state_dict.items():
    res.append(val)
    name_to_pos[name] = i
    i += 1
  return res, name_to_pos


def tensor_to_jax_array(tensor):
  if isinstance(tensor, torch.Tensor):
    tensor = tensor.detach().cpu().numpy()
  return tensor


def _fill_freqs_cis(state_dict, model_args):
  state_dict["L__fn___freqs_cis"] = llama2_model.precompute_freqs_cis(
      model_args.dim // model_args.n_heads, model_args.max_seq_len * 2
  )


def load_checkpoint(checkpoint_dir: str) -> Any:
  if checkpoint_dir:
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
    assert len(checkpoints) == 1, 'currently only support one file'
    # Need to merge the checkpoint to 1 file.
    checkpoint = torch.load(checkpoints[0])
    return checkpoint
  return None


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


class ImportedModel:
  """Imported transformer model."""

  def __init__(
      self,
      param_size,
      context_length,
      ckpt_path,
      batch_size,
      bf16_enable,
      num_of_partitions,
      pt_model,
      input_prefill,
      input_decode,
      model_args,
      shard_weights_fn,
  ):
    self.model_args = model_args
    self.shard_weights = shard_weights_fn
    self.ckpt_path = ckpt_path
    self.num_of_partitions = num_of_partitions
    self.sample_input_prefill = input_prefill
    self.sample_input_decode = input_decode

    self.pt_model = pt_model

    if bf16_enable:
      self.pt_model.to(torch.bfloat16)

    # This might be the best place to run the original Pytorch model with pdb.
    # self.pt_model(*self.sample_input_prefill)
    self.convert_to_jax_fn()

    # For testing purpose, collect all the inputs & outputs for each step for validation
    self.prefill_inputs = None
    self.prefill_outputs = None
    self.decode_inputs = None
    self.decode_outputs = None

  def convert_to_jax_fn(self):
    exported_prefill = torch.export.export(
        self.pt_model, self.sample_input_prefill
    )
    exported_decode = torch.export.export(
        self.pt_model, self.sample_input_decode
    )

    self.names, self.weights, self.prefill_fn = exported_program_to_jax(
        exported_prefill, export_raw=True
    )
    _, _, self.decode_fn = exported_program_to_jax(
        exported_decode, export_raw=True
    )

  def load_weights(self) -> Any:
    checkpoint = load_checkpoint(self.ckpt_path)
    if checkpoint:
      self.pt_model.load_state_dict(checkpoint, strict=False)
      self.convert_to_jax_fn()
    self.weights = self.shard_weights(self.names, self.weights, self.num_of_partitions)
    # We don't place the weights to device here for testing
    return self.weights

  def prefill(self, weights: Any, inputs: Any) -> Any:
    self.prefill_inputs = inputs
    self.prefill_outputs = self.prefill_fn(weights, inputs)
    return self.prefill_outputs

  def decode(self, weights: Any, inputs: Any) -> Any:
    self.decode_inputs = inputs
    self.decode_outputs = self.decode_fn(weights, inputs)
    return self.decode_outputs

  def place_weights(self, weights) -> Any:
    return jax.tree_map(
        lambda x: jnp.asarray(x, dtype=n2jtype(x)),
        weights,
    )

