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

import jax
from absl import app, flags
from colorama import Fore, Style
from jetstream.engine import token_utils
from jetstream_pt import ray_engine
from jetstream_pt.config import (
    FLAGS,
    create_engine_from_config_flags,
    define_common_flags,
    define_profiling_flags,
)

define_common_flags()
define_profiling_flags()
# _TOKENIZER_PATH = flags.DEFINE_string(
#     "tokenizer_path",
#     "tokenizer.model",
#     "The tokenizer model path",
#     required=False,
# )
# _CKPT_PATH = flags.DEFINE_string(
#     "checkpoint_path", None, "Directory for .pth checkpoints", required=False
# )
# _BF16_ENABLE = flags.DEFINE_bool(
#     "bf16_enable", False, "Whether to enable bf16", required=False
# )
# _CONTEXT_LENGTH = flags.DEFINE_integer(
#     "context_length", 1024, "The context length", required=False
# )
# _BATCH_SIZE = flags.DEFINE_integer(
#     "batch_size", 32, "The batch size", required=False
# )
# _PROFILING_OUTPUT = flags.DEFINE_string(
#     "profiling_output",
#     "",
#     "The profiling output",
#     required=False,
# )

# _SIZE = flags.DEFINE_string("size", "tiny", "size of model")

# _QUANTIZE_WEIGHTS = flags.DEFINE_bool(
#     "quantize_weights", False, "weight quantization"
# )
# _QUANTIZE_KV_CACHE = flags.DEFINE_bool(
#     "quantize_kv_cache", False, "kv_cache_quantize"
# )
# _MAX_CACHE_LENGTH = flags.DEFINE_integer(
#     "max_cache_length", 1024, "kv_cache_quantize"
# )

# _MODEL_NAME = flags.DEFINE_string(
#     "model_name", None, "model type", required=False
# )

# _SHARDING_CONFIG = flags.DEFINE_string(
#     "sharding_config", "", "config file for sharding"
# )


def create_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  start = time.perf_counter()
  engine = ray_engine.create_pytorch_ray_engine(
      model_name=FLAGS.model_name,
      tokenizer_path=FLAGS.tokenizer_path,
      ckpt_path=FLAGS.checkpoint_path,
      bf16_enable=FLAGS.bf16_enable,
      param_size=FLAGS.size,
      context_length=FLAGS.context_length,
      batch_size=FLAGS.batch_size,
      quantize_weights=FLAGS.quantize_weights,
      quantize_kv=FLAGS.quantize_kv_cache,
      max_cache_length=FLAGS.max_cache_length,
      sharding_config=FLAGS.sharding_config,
      shard_on_batch=FLAGS.shard_on_batch,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


# pylint: disable-next=all
def main(argv):

  engine = create_engine_from_config_flags()

  start = time.perf_counter()
  engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)
  stop_tokens = [vocab.eos_id, vocab.pad_id]
  max_output_length = 1024

  #   if _PROFILING_OUTPUT.value:
  #     jax.profiler.start_trace(_PROFILING_OUTPUT.value)
  profiling_output = FLAGS.profiling_output
  if profiling_output:
    jax.profiler.start_trace(profiling_output)

  engine.init_decode_state()
  prompts: List[str] = [
      "I believe the meaning of life is",
      # pylint: disable-next=all
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      # pylint: disable-next=all
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
  ]
  for prompt in prompts:
    slot = random.randint(0, FLAGS.batch_size - 1)
    tokens, true_length = token_utils.tokenize_and_pad(
        prompt, vocab, is_bos=True, jax_padding=False
    )
    print(f"---- Input prompts are: {prompt}")
    print(f"---- Encoded tokens are: {tokens}")

    # pylint: disable-next=all
    prefill_result = engine.prefill(
        params=None, padded_tokens=tokens, true_length=true_length
    )
    # pylint: disable-next=all
    decode_state = engine.insert(prefill_result, None, slot=slot)
    sampled_tokens_list = []
    while True:
      # pylint: disable-next=all
      decode_state, result_tokens = engine.generate(None, decode_state)

      slot_data = result_tokens.get_result_at_slot(slot)
      slot_tokens = slot_data.tokens
      slot_lengths = slot_data.lengths

      token_id = slot_tokens[slot, 0].item()
      if slot_lengths > max_output_length or token_id in stop_tokens:
        break

      sampled_tokens_list.append(token_id)

    print("---- All output tokens.")
    print(sampled_tokens_list)
    print("---- All output text.")
    print(vocab.tokenizer.decode(sampled_tokens_list))

  if profiling_output:
    jax.profiler.stop_trace()


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  app.run(main)
