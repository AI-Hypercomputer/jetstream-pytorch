from . import model_args
import jax.sharding as jsharding
import jax
from jax.experimental import mesh_utils
from torch.utils import _pytree as pytree
import torch
from typing import Any
from .. import utils
import numpy as np
from jax import numpy as jnp

def get_arg(
    param_size: str,
    seqlen,
    batch_size,
    vocab_size: int,
    bf16_enable: bool = False,
) -> model_args.ModelArgs:
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
  return model_args.ModelArgs(
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


def make_cache(args: model_args.ModelArgs, batch_size: int | None = None):
  """Creates a cache for each layer."""

  head_dim = args.dim // args.n_heads
  res = []
  kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
  for _ in range(args.n_layers):
    k_size = (batch_size, kv_heads, args.max_seq_len, head_dim)
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


P = jsharding.PartitionSpec

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

# TODO: Consider take the mesh/topology instead of number of partitions.
def shard_weights(names: Any, weights: Any, num_of_partitions: int) -> Any:
    if not weights:
        return None

    mesh = jsharding.Mesh(
        mesh_utils.create_device_mesh((num_of_partitions, 1)),
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
    weights = pytree.tree_map(utils.p2n, weights)
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
    weights = jax.tree_map(
        lambda x, shard: jax.device_put(x, shard).astype(n2jtype(x)),
        weights, weight_sharding 
    )
    return jax.lax.with_sharding_constraint(weights, weight_sharding)


def make_prefill_input(context_length, caches):
  # NOTE prefill input size has to be same as context length
  input_shape_prefill = (1, context_length)
  input_prefill = (
        torch.randint(
            0, 1000, input_shape_prefill, dtype=torch.int32
        ),  # len seq length
        torch.arange(0, context_length, dtype=torch.int32),  # input indexes
        torch.arange(0, context_length, dtype=torch.int32),  # context indexes
        caches,
        True,  # prefil
    )
  return input_prefill


def make_decode_input(context_length, caches, batch_size):
  # NOTE decode input size has to be 1
  # NOTE possition > CONTEXT_LENGTH
  input_shape_decode = (batch_size, 1)
  input_decode = (
        torch.randint(
            0, 1000, input_shape_decode, dtype=torch.int32
        ),  # len = 1
        torch.tensor([0], dtype=torch.int32),
        torch.roll(torch.arange(context_length, dtype=torch.int32), 1, 0),
        caches,
        False,  # decode
    )
  return input_decode


def input_output_spec(model_args) -> Any:
    """Returns the input spec and output spec."""

    in_spec_like = (
        5,
        5,
        None,
        [tuple([0, 1]) for _ in range(0, model_args.n_layers)],
        False,
    )
    out_spec_like = (
        4,
        [tuple([0, 1]) for _ in range(0, model_args.n_layers)],
    )
    _, in_spec = pytree.tree_flatten(in_spec_like)
    _, out_spec = pytree.tree_flatten(out_spec_like)
    return in_spec, out_spec
