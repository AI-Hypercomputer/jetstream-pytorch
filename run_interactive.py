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
import random
import time

from typing import List
from absl import app
from absl import flags
from colorama import Fore, Style

import jax

from jetstream.engine import token_utils
from colorama import Fore, Style
import numpy as np

import os

from jetstream_pt import engine as je

FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    "tokenizer_path",
    "tokenizer.model",
    "The tokenizer model path",
    required=False,
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "model type", required=False
)
_CKPT_PATH = flags.DEFINE_string(
    "checkpoint_path", None, "Directory for .pth checkpoints", required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    "bf16_enable", False, "Whether to enable bf16", required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    "context_length", 1024, "The context length", required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 32, "The batch size", required=False
)
_PROFILING_OUTPUT = flags.DEFINE_string(
    "profiling_output",
    "",
    "The profiling output",
    required=False,
)

_SIZE = flags.DEFINE_string("size", "tiny", "size of model")

_QUANTIZE_WEIGHTS = flags.DEFINE_bool(
    "quantize_weights", False, "weight quantization"
)
_QUANTIZE_NUM_BITS_WEIGHTS = flags.DEFINE_integer(
    "quantize_num_bits_weights", 8, "number of bits of quantized weight."
)
_QUANTIZE_IS_BLOCKWISE_WEIGHTS = flags.DEFINE_bool(
    "quantize_is_blockwise_weights", False, "blockwise quantization for weight."
)
_QUANTIZE_KV_CACHE = flags.DEFINE_bool(
    "quantize_kv_cache", False, "kv_cache_quantize"
)
_MAX_CACHE_LENGTH = flags.DEFINE_integer(
    "max_cache_length", 1024, "kv_cache_quantize"
)
_SHARDING_CONFIG = flags.DEFINE_string(
    "sharding_config", "", "config file for sharding"
)
_SHARD_ON_BATCH = flags.DEFINE_bool(
    "shard_on_batch",
    False,
    "whether to shard on batch dimension."
    "If set true, sharding_config will be ignored.",
)


def create_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  devices = jax.devices()
  start = time.perf_counter()

  quantize_weight = _QUANTIZE_WEIGHTS.value
  quanitze_is_blockwise_weight = _QUANTIZE_IS_BLOCKWISE_WEIGHTS.value
  sharding_config_path = _SHARDING_CONFIG.value
  if not sharding_config_path:
    sharding_config_name = _MODEL_NAME.value
    if quantize_weight and quanitze_is_blockwise_weight:
      sharding_config_name += "-blockwise-quant"
    sharding_config_path = os.path.join(
        "default_shardings", sharding_config_name + ".yaml"
    )

  engine = je.create_pytorch_engine(
      model_name=_MODEL_NAME.value,
      devices=devices,
      tokenizer_path=_TOKENIZER_PATH.value,
      ckpt_path=_CKPT_PATH.value,
      bf16_enable=True,
      param_size=_SIZE.value,
      context_length=_CONTEXT_LENGTH.value,
      batch_size=_BATCH_SIZE.value,
      quantize_weights=quantize_weight,
      quantize_num_bits_weights=_QUANTIZE_NUM_BITS_WEIGHTS.value,
      quanitze_is_blockwise_weight=quanitze_is_blockwise_weight,
      quantize_kv=_QUANTIZE_KV_CACHE.value,
      max_cache_length=_MAX_CACHE_LENGTH.value,
      sharding_config=sharding_config_path,
      shard_on_batch=_SHARD_ON_BATCH.value,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


# pylint: disable-next=all
def main(argv):

  engine = create_engine()

  start = time.perf_counter()
  params = engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)
  max_output_length = 1024

  if _PROFILING_OUTPUT.value:
    jax.profiler.start_trace(_PROFILING_OUTPUT.value)

  decode_state = engine.init_decode_state()
  prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
  ]
  for prompt in prompts:
    slot = random.randint(0, _BATCH_SIZE.value - 1)
    tokens, true_length = tokenizer.encode(prompt, is_bos=True)

    print(f"---- Input prompts are: {prompt}")
    print(f"---- Encoded tokens are: {tokens}")

    # pylint: disable-next=all
    prefill_result = engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    # pylint: disable-next=all
    decode_state = engine.insert(prefill_result, decode_state, slot=slot)
    sampled_tokens_list = []
    print(f"---- Streaming decode started on #slot{slot}.")
    complete = np.zeros((1,), dtype=np.bool_)
    while True:
      decode_state, result_tokens = engine.generate(params, decode_state)
      result_tokens = result_tokens.convert_to_numpy()
      output, complete = tokenizer.decode(
          slot, max_output_length, result_tokens, complete
      )
      if complete[0]:
        break
      token_id = output[0][0]
      sampled_tokens_list.append(token_id)
      # output_str = tokenizer.decode_str([token_id])
      # print(Fore.GREEN + output_str, end="", flush=True)

    # print(Style.RESET_ALL + "\n")
    # print("---- Streaming decode finished.")

    print("---- All output tokens.")
    print(sampled_tokens_list)
    print("---- All output text.")
    print(tokenizer.decode_str(sampled_tokens_list))

  if _PROFILING_OUTPUT.value:
    jax.profiler.stop_trace()


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
