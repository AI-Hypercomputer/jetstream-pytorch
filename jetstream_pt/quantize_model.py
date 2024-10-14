import torch
from .environment import QuantizationConfig
from .layers import (
    create_quantized_from_nn_linear,
    create_quantized_from_nn_embedding,
    AttentionKernel,
    Int8KVAttentionKernel,
)


def quantize_model(float_model, config: QuantizationConfig):
  """Apply quantization to linear layers."""

  def quantize_nn_mod(float_model):
    for name, mod in float_model.named_modules():
      new_mod = None
      if hasattr(mod, "get_quantized_version"):
        new_mod = mod.get_quantized_version()
      elif isinstance(mod, torch.nn.Linear):
        new_mod = create_quantized_from_nn_linear(mod, config)
      elif (
          isinstance(mod, torch.nn.Embedding)
          and config.enable_embedding_quantization
      ):
        new_mod = create_quantized_from_nn_embedding(mod, config)

      if new_mod:
        setattr(float_model, name, new_mod)

    if config.enable_kv_quantization:
      for name, mod in float_model.__dict__.items():
        if isinstance(mod, AttentionKernel):
          new_mod = Int8KVAttentionKernel(mod.env, mod.layer_id)
          setattr(float_model, name, new_mod)

  float_model.apply(quantize_nn_mod)
  return float_model
