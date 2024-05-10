# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl import app
from absl import flags

from jetstream_pt.environment import JetEngineEnvironment, JetEngineEnvironmentData, process_sharding_name
from jetstream_pt.third_party.llama2 import model_exportable, model_args
from jetstream_pt.third_party.gemma import config as gemma_config, model as gemma_model

FLAGS = flags.FLAGS

_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "model type", required=False
)

_SIZE = flags.DEFINE_string("size", "tiny", "size of model")

_COLLAPSE_SAME_LAYERS = flags.DEFINE_bool("collapse_same_layers", True, "")


def create_model():
  batch_size = 3
  quant_config = QuantizationConfig(
      enable_weight_quantization=True, enable_kv_quantization=True
  )
  env_data = JetEngineEnvironmentData(
      batch_size=3,
      max_decode_length=1024,
      max_input_sequence_length=1024,
      quant_config=quant_config,
      cache_sequence_length=1024,
      bf16_enable=True,
  )
  model_name = _MODEL_NAME.value
  param_size = _SIZE.value
  if model_name.startswith("llama"):

    args = model_args.get_model_args(
        param_size,
        1024,
        batch_size,
        vocab_size=32000,
        bf16_enable=True,
    )
    args.device = "meta"
    args.quantize = False
    env = JetEngineEnvironment(env_data)
    return model_exportable.Transformer(args, env)
  elif model_name == "gemma":
    args = gemma_config.get_model_config(param_size)
    args.device = "meta"
    env_data.model_type = "gemma-" + param_size
    env_data.num_layers = args.num_hidden_layers
    env = JetEngineEnvironment(env_data)
    pt_model = gemma_model.GemmaModel(args, env)
    return pt_model


# pylint: disable-next=all
def main(argv):
  model = create_model()
  res = {}
  for k, v in model.state_dict().items():
    res[process_sharding_name(k)] = v

  print(
      f"""
# Sharding config for {_MODEL_NAME.value}
# Sharding should either be an int between 0 and rank - 1
# signifying the axis to shard or -1 / null signifying replicated

"""
  )

  for k, v in res.items():
    print(k, ":", -1, "# ", str(v.dtype), tuple(v.shape))


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
