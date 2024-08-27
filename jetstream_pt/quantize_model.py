import torch
from .layers import (create_quantized_from_nn_linear, 
  create_quantized_from_nn_embedding)


def quantize_model(float_model, config):
  """Apply quantization to linear layers."""

  def quantize_nn_mod(float_model):
    for name, mod in float_model.named_modules():
      new_mod = None
      if hasattr(mod, 'get_quantized_version'):
        new_mod = mod.get_quantized_version()
      elif isinstance(mod, torch.nn.Linear):
        new_mod = create_quantized_from_nn_linear(mod, config)
      elif isinstance(mod, torch.nn.Embedding):
        new_mod = create_quantized_from_nn_embedding(mod, config)

      if new_mod:
        setattr(float_model, name, new_mod)
      

  float_model.apply(quantize_nn_mod)
  return float_model
