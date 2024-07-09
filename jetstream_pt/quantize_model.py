import torch
from .layers import create_quantized_from_nn_linear


def quantize_model(float_model, config):
  """Apply quantization to linear layers."""

  def quantize_nn_mod(float_model):
    for name, mod in float_model.named_modules():
      if isinstance(mod, torch.nn.Linear):
        new_mod = create_quantized_from_nn_linear(mod, config)
        setattr(float_model, name, new_mod)

  float_model.apply(quantize_nn_mod)
  return float_model
