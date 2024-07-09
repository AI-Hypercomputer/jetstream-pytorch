import dataclasses
import glob
import os
from typing import Optional
from requests.exceptions import HTTPError
from huggingface_hub import snapshot_download
from absl import flags
import torch
from safetensors import safe_open
from jetstream_pt.environment import (
    JetEngineEnvironmentData,
    QuantizationConfig,
)
from jetstream_pt.third_party.llama import model_exportable as llama_model

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "working_dir",
    "checkpoints",
    "Directory to store downloaded/converted weights",
)
flags.DEFINE_string("hf_token", "", "huggingface token")

flags.DEFINE_integer(
    "override_max_cache_length",
    -1,
    "Size of cache, defaults to input + output length",
)


llama_model_ids = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-3-8B",
    "meta-llama/Llama-3-8B-Instruct",
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


@dataclasses.dataclass
class ModelInfo:
  """Model information."""

  model_class: torch.nn.Module
  # information needed to allocate cache
  num_layers: int
  num_heads: int
  head_dim: int


_llama2_7 = ModelInfo(llama_model.Transformer, 32, 32, 128)
_llama2_13 = ModelInfo(llama_model.Transformer, 40, 40, 128)
_llama2_70 = ModelInfo(llama_model.Transformer, 80, 8, 128)
_llama3_8 = ModelInfo(llama_model.Transformer, 32, 8, 128)


model_id_to_class = {
    "meta-llama/Llama-2-7b-chat-hf": _llama2_7,
    "meta-llama/Llama-2-7b-hf": _llama2_7,
    "meta-llama/Llama-2-13b-chat-hf": _llama2_13,
    "meta-llama/Llama-2-13b-hf": _llama2_13,
    "meta-llama/Meta-Llama-3-8B": _llama3_8,
    "meta-llama/Meta-Llama-3-8B-Instruct": _llama3_8,
    "google/gemma-2b": None,
    "google/gemma-2b-it": None,
    "google/gemma-7b": None,
    "google/gemma-7b-it": None,
    "mistralai/Mixtral-8x7B-v0.1": None,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": None,
}


def _model_dir(repo_id):
  """Model dir structure:

  working_dir/
    repo_id/
      hf_original/
      converted_bfloat/
      converted_int8/
  """
  return os.path.join(FLAGS.working_dir, repo_id)


def _hf_dir(repo_id):
  """Dir to hf repo"""
  return os.path.join(_model_dir(repo_id), "hf_original")


def _int_dir(repo_id):
  return os.path.join(_model_dir(repo_id), "converted_int8")


def construct_env_data_from_model_id(
    repo_id,
    batch_size,
    input_length,
    output_length,
    quantize,
):
  """Create Environment from model id and options"""
  tokenizer_path = os.path.join(_hf_dir(repo_id), "tokenizer.model")
  if quantize:
    checkpoint_path = _int_dir(repo_id)
    checkpoint_format = "safetensors"
  else:
    checkpoint_path = _hf_dir(repo_id)
    checkpoint_format = "safetensors"

  shard_on_batch = False

  max_cache_length = (
      FLAGS.override_max_cache_length
      if FLAGS.override_max_cache_length > 0
      else input_length + output_length
  )

  env_data = JetEngineEnvironmentData(
      tokenizer_path=tokenizer_path,
      checkpoint_path=checkpoint_path,
      checkpoint_format=checkpoint_format,
      batch_size=batch_size,
      max_decode_length=output_length,
      max_input_sequence_length=input_length,
      quant_config=QuantizationConfig(),
      cache_sequence_length=max_cache_length,
      bf16_enable=True,
      sharding_config_path="",
      shard_on_batch=shard_on_batch,
  )
  model_info = model_id_to_class.get(repo_id)
  env_data.cache_shape = (
      batch_size,
      model_info.num_heads,
      max_cache_length,
      model_info.head_dim,
  )
  env_data.num_layers = model_info.num_layers
  return env_data


def _load_weights(directory):
  safetensors_files = glob.glob(os.path.join(directory, "*.safetensors"))
  state_dict = {}
  for file_path in safetensors_files:
    with safe_open(file_path, framework="pt") as f:
      for key in f.keys():
        state_dict[key] = f.get_tensor(key).to(torch.bfloat16)
  # Load the state_dict into the model
  return state_dict


def instantiate_model_from_repo_id(
    repo_id,
    env,
):
  """Create model instance by hf model id.+"""
  model_dir = _hf_dir(repo_id)
  if not os.path.exists(model_dir) or not os.listdir(model_dir):
    # no weights has been downloaded
    _hf_download(repo_id, model_dir, FLAGS.hf_token)
  model_info = model_id_to_class.get(repo_id)
  assert model_info is not None

  env.device = "meta"
  model = model_info.model_class.from_hf_model_id(repo_id, env)
  weights = _load_weights(model_dir)
  updated_keys = model.get_hf_names_to_real_name()
  for name, updated in updated_keys.items():
    if name in weights:
      val = weights.pop(name)
      weights[updated] = val

  model.load_state_dict(weights, assign=True, strict=False)

  return model
  ## QQ do i need to set the weights onto the model?


def _hf_download(
    repo_id: str, dest_directory: str, hf_token: Optional[str] = None
) -> None:
  os.makedirs(dest_directory, exist_ok=True)
  try:
    if not hf_token:
      hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
      # NOTE: setting true allows hf to read from the config folder.
      hf_token = True
    snapshot_download(
        repo_id,
        local_dir=dest_directory,
        local_dir_use_symlinks=False,
        token=hf_token,
        allow_patterns=[
            "model-?????-of-?????.safetensors",
            "*.json",
            "*.model",
        ],
    )
  except HTTPError as e:
    if e.response.status_code == 401:
      print(
          "Please use huggingface-cli login to authenticate "
          "to download private checkpoints."
      )
      print("OR, pass `hf_token=...` explicitly.")
    raise e
