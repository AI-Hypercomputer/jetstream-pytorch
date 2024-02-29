"""Import llama2 model."""
import os
from typing import Any
from absl import logging
import jax
from jax.experimental import jax2tf
from jax.experimental import mesh_utils
import jax.numpy as jnp
import jax.sharding as jsharding
import numpy as np
import torch
from torch.utils import _pytree as pytree
from torch_xla2.export import exported_program_to_jax
from ..model.llama2 import model as llama2_model
from ..model.llama2 import model_exportable
from ..model import tokenizer
from tensorflow.compiler.tf2xla.python import xla as tfxla  # pylint: disable=g-direct-tensorflow-import
from pathlib import Path

CONTEXT_LENGTH = 2048
max_input_seq_length = CONTEXT_LENGTH + 256


def make_cache(args: llama2_model.ModelArgs, batch_size: int | None = None):
  """Creates a cache for each layer."""

  head_dim = args.dim // args.n_heads
  res = []
  for _ in range(args.n_layers):
    k_size = (batch_size, args.n_kv_heads, args.max_seq_len, head_dim)
    v_size = k_size
    res.append((
        torch.zeros(
            k_size, dtype=torch.bfloat16 if args.bf16_enable else torch.float
        ),
        torch.zeros(
            v_size, dtype=torch.bfloat16 if args.bf16_enable else torch.float
        ),
    ))
  return res


def get_arg(
    param_size: str,
    seqlen,
    batch_size,
    vocab_size: int,
    bf16_enable: bool = False,
) -> llama2_model.ModelArgs:
  """Gets model args."""

  data = {}
  if param_size == "tiny":
    data = {
        "dim": 128,
        "multiple_of": 32,
        "n_heads": 2,
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
  return llama2_model.ModelArgs(
      max_seq_len=seqlen,
      max_batch_size=batch_size,
      vocab_size=vocab_size,
      bf16_enable=bf16_enable,
      **data,
  )


def flatten_state_dict(state_dict):
  i = 0
  res = []
  name_to_pos = {}
  for name, val in state_dict.items():
    res.append(val)
    name_to_pos[name] = i
    i += 1
  return res, name_to_pos


def wrap_as_jax_func(func, mappings):
  """Wraps a function as a JAX function."""

  touts = [sig.dtype for sig in func.meta.output_signature]
  souts = [sig.shape for sig in func.meta.output_signature]

  def inner(weights, args):
    full_args = (weights, args)  # (296), (1)
    call_args = [full_args[type_][index] for type_, index in mappings]
    return tfxla.call_module(
        tuple(call_args),
        version=5,
        Tout=touts,  # dtype information
        Sout=souts,  # Shape information
        function_list=[],
        module=func.bytecode,
    )

  return jax2tf.call_tf(inner)


def tensor_to_jax_array(tensor):
  if isinstance(tensor, torch.Tensor):
    tensor = tensor.detach().cpu().numpy()
  return tensor


def _fill_freqs_cis(state_dict, model_args):
  state_dict["L__fn___freqs_cis"] = llama2_model.precompute_freqs_cis(
      model_args.dim // model_args.n_heads, model_args.max_seq_len * 2
  )


def make_prefill_input(caches, tokens):
  # NOTE prefill input size has to be same as context length
  input_prefill = (
      tokens,
      torch.arange(0, CONTEXT_LENGTH).to(torch.int64),  # input indexes
      torch.arange(0, CONTEXT_LENGTH).to(torch.int64),  # context indexes
      caches,  # caches
      True,  # prefil
  )
  return input_prefill


def make_decode_input(caches, tokens, position):
  # NOTE decode input size has to be 1
  # NOTE possition > CONTEXT_LENGTH
  input_prefill = (
      tokens,
      torch.arange(position, position + 1),  # input indexes
      torch.arange(position - CONTEXT_LENGTH, position),  # context indexes
      caches,  # caches
      False,  # prefil
  )
  return input_prefill


def load_checkpoint(checkpoint_dir: str) -> Any:
  if checkpoint_dir:
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pth"))
    assert len(checkpoints) == 1, 'currently only support one file'
    # Need to merge the checkpoint to 1 file.
    checkpoint = torch.load(checkpoints[0])
    return checkpoint
  return None


P = jsharding.PartitionSpec


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


class Llama2ImportedModel:
  """Imported llama2 model."""

  def __init__(
      self,
      param_size,
      context_length,
      ckpt_path,
      tokenizer_path,
      batch_size,
      bf16_enable,
      num_of_partitions,
  ):
    self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
    self.model_args = get_arg(
        param_size=param_size,
        seqlen=context_length,
        batch_size=batch_size,
        vocab_size=self.tokenizer.n_words,
        bf16_enable=bf16_enable,
    )
    self.model_args.n_kv_heads = (
        self.model_args.n_heads
        if self.model_args.n_kv_heads is None
        else self.model_args.n_kv_heads
    )
    self.model_args.head_dim = self.model_args.dim // self.model_args.n_heads

    self.ckpt_path = ckpt_path
    self.num_of_partitions = num_of_partitions

    self.pt_model = model_exportable.Transformer(self.model_args)
    if bf16_enable:
      self.pt_model.to(torch.bfloat16)

    prefill_caches = make_cache(self.model_args, 1)
    decode_caches = make_cache(self.model_args, batch_size)

    input_shape_prefill = (1, context_length)
    input_shape_decode = (batch_size, 1)

    self.sample_input_prefill = (
        torch.randint(
            0, 1000, input_shape_prefill, dtype=torch.int32
        ),  # len seq length
        torch.arange(0, context_length, dtype=torch.int32),  # input indexes
        torch.arange(0, context_length, dtype=torch.int32),  # context indexes
        prefill_caches,
        True,  # prefil
    )

    self.sample_input_decode = (
        torch.randint(
            0, 1000, input_shape_decode, dtype=torch.int32
        ),  # len = 1
        torch.tensor([0], dtype=torch.int32),
        torch.roll(torch.arange(context_length, dtype=torch.int32), 1, 0),
        decode_caches,
        False,  # prefill
    )

    # This might be the best place to run the original Pytorch model with pdb.
    # self.pt_model(*self.sample_input_prefill)
    self.convert_to_jax_fn()

    # For testing purpose
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
    self.weights = self.shard_weights(self.names, self.weights)
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

  def shard_weights(self, names: Any, weights: Any) -> Any:
    if not weights:
      return None

    mesh = jsharding.Mesh(
        mesh_utils.create_device_mesh((self.num_of_partitions, 1)),
        axis_names=("x", "y"),
    )
    y_sharding = jsharding.NamedSharding(mesh, P(None, "x"))
    x_sharding = jsharding.NamedSharding(mesh, P("x"))
    replicated = jsharding.NamedSharding(mesh, P())

    weight_sharding = []

    for name in names:
      if "tok_embeddings." in name:
        weight_sharding.append(x_sharding)
        continue
      if "attention." in name:
        if "wo" in name:
          weight_sharding.append(x_sharding)
        else:
          weight_sharding.append(y_sharding)
        continue
      if "feed_forward." in name:
        if "w2" in name:
          weight_sharding.append(x_sharding)
        else:
          weight_sharding.append(y_sharding)
        continue
      if "output" in name:
        weight_sharding.append(y_sharding)
        continue
      weight_sharding.append(replicated)

    print("Number of weights:", len(names))
    for name, w, sharding in zip(names, weights, weight_sharding):
      print(
          "Name: %s Shape: %s dtype: %s sharding %s:"
          % (name, w.shape, w.dtype, sharding)
      )
    weights = pytree.tree_map(p2n, weights)
    print(
        "After conversion, all the weights are converted to: %s",
        type(weights[0]),
        flush=True,
    )
    for name, w, sharding in zip(names, weights, weight_sharding):
      print(
          "Name: %s Shape: %s dtype: %s sharding %s:"
          % (name, w.shape, w.dtype, sharding)
      )
    return jax.lax.with_sharding_constraint(weights, weight_sharding)

  def place_weights(self, weights) -> Any:
    return jax.tree_map(
        lambda x: jnp.asarray(x, dtype=n2jtype(x)),
        weights,
    )

  def input_output_spec(self) -> Any:
    """Returns the input spec and output spec."""

    in_spec_like = (
        5,
        5,
        None,
        [tuple([0, 1]) for _ in range(0, self.model_args.n_layers)],
        False,
    )
    out_spec_like = (
        4,
        [tuple([0, 1]) for _ in range(0, self.model_args.n_layers)],
    )
    _, in_spec = pytree.tree_flatten(in_spec_like)
    _, out_spec = pytree.tree_flatten(out_spec_like)
    return in_spec, out_spec
