import abc
import itertools
from typing import Dict, Any, Optional
import dataclasses
from collections import defaultdict
import torch


def _get_hf_name(module, key):
  if hasattr(module, "attr_to_property") and key in module.attr_to_property:
    return module.attr_to_property[key].huggingface_name
  return None


def _gather_names(module, myprefix, hf_prefix, result):
  for key, _ in itertools.chain(
      module.named_parameters(recurse=False),
      module.named_buffers(recurse=False),
  ):
    hf_name = _get_hf_name(module, key) or key
    result[hf_prefix + hf_name] = myprefix + key

  for name, child in module.named_children():
    hf_name = _get_hf_name(module, name) or name
    _gather_names(
        child, myprefix + name + ".", hf_prefix + hf_name + ".", result
    )


def _gather_sharding_axis(module, myprefix, result):
  if hasattr(module, "attr_to_property"):
    for key, val in module.attr_to_property.items():
      if val.sharding_axis is not None:
        result[myprefix + key] = val.sharding_axis

  for name, child in module.named_children():
    _gather_sharding_axis(child, myprefix + name + ".", result)


@dataclasses.dataclass
class AttrProperty:
  """Attributes attached to model weights."""

  huggingface_name: Optional[str] = None
  sharding_axis: Optional[int] = None


class ModuleBase(torch.nn.Module, metaclass=abc.ABCMeta):
  """nn Module that allows attaching properties.
  
  This class currently serves 2 goals:
  1. Allow model to specify alternative names for submodules / weights
     this is needed so that it can *also* load HuggingFace checkpoints
     without need to do massive rewrites.

  2. Allow model to attach information to weights, such as sharding config.

  Quantization config could be another thing to attach, but right now it's not used
  this way.
  """

  attr_to_property: Dict[str, Any]

  def __init__(self):
    super().__init__()
    self.attr_to_property = defaultdict(AttrProperty)

  def get_hf_names_to_real_name(self):
    """Return a dict of attr names to it's hf name."""
    result = {}
    _gather_names(self, "", "", result)
    return result

  def get_sharding_annotations(self):
    """Return a dict of attr names to it's sharding dim."""
    result = {}
    _gather_sharding_axis(self, "", result)
    return result

  def hf_name(self, orig_name, hf_name):
    """Set it's alternative name for a attribute or submodule."""
    self.attr_to_property[orig_name].huggingface_name = hf_name

  def annotate_sharding(self, name, axis):
    """Set sharding name for a attribute or submodule."""
    self.attr_to_property[name].sharding_axis = axis

  def convert_hf_weights(self, hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Load state_dict with hg weights."""
    weights = {}
    updated_keys = self.get_hf_names_to_real_name()
    for name, updated in updated_keys.items():
      if name in hf_weights:
        weights[updated] = hf_weights[name]

    for name in list(weights.keys()):
      if 'inv_freq' in name:
        weights.pop(name)
    if hasattr(self, 'freqs_cis'):
      weights['freqs_cis'] = self.freqs_cis
    return weights

