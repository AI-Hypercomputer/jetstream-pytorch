import torch_xla2

env = torch_xla2.default_env()


def from_jax(tensors):
  """Wrap a jax Array into XLATensor."""
  return env.j2t_iso(tensors)


def to_jax(tensors):
  """Unwrap a XLATensor into jax Array."""
  return env.t2j_iso(tensors)
