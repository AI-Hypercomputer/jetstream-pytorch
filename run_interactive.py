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

# import torch_xla2 first!
import torch_xla2  # pylint: disable
import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags
from colorama import Fore, Style
from jetstream.engine import token_utils
from jetstream_pt import engine as je
from jetstream_pt import offline_inference
from jetstream_pt.config import FLAGS, create_engine_from_config_flags

flags.DEFINE_string("prompt_file", "", "File with prompts")

flags.DEFINE_string("prompt_format", "mlperf", "format of prompts")


def get_prompts():
  if FLAGS.prompt_file:
    if prompt_format == "mlperf":
      import pandas as pd

      data = pd.read_pickle(FLAGS.prompt_file)
      return list(data.input)


# pylint: disable-next=all
def main(argv):
  engine = create_engine_from_config_flags()
  start = time.perf_counter()
  offline_inf = offline_inference.OfflineInference(engine)

  metadata = engine.get_tokenizer()
  tokenizer = engine.build_tokenizer(metadata)

  profiling_output = FLAGS.profiling_output
  profiling_prefill = (
      FLAGS.profiling_prefill
      and profiling_output is not None
      and profiling_output != ""
  )

  prompts: List[str] = [
      "I believe the meaning of life is",
      "To add an element to an ArrayList of a specific class type in Java, you can follow the following steps:\n\n1. Create an instance of the class to be added.\n2. Get a reference to the ArrayList.\n3. Call the `add()` method on the ArrayList, passing the instance of the class as the argument.\n\nHere's an example of how to add an object of type `Person` to an ArrayList of type `ArrayList<Person>`:\n```csharp\n// Create a new instance of the Person class\nPerson person = new Person(\"John\", 25);\n\n// Get a reference to the ArrayList\nArrayList<Person> peopleList = new ArrayList<>();\n\n// Add the person object to the ArrayList\npeopleList.add(person);\n```\nIn this example, the `Person` class is assumed to have a constructor that takes two arguments: a String for the person's name, and an int for their age. You can substitute your own class and constructor as necessary.",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can. While performing the task think step-by-step and justify your steps.\n<</SYS>>\n\nQuestion 1: What is commercial real estate finance?\nQuestion 2: What are Commercial Real Estate services?\nOptions are:\n[a]. no.\n[b]. yes.\nWould the answer to these two questions be the same? [/INST]",
      "<s>[INST] <<SYS>>\nYou are an AI assistant that helps people find information. Provide a detailed answer so user don\u2019t need to search outside to understand the answer.\n<</SYS>>\n\nUse reasoning to lead to the answer of the following question:\nWhere are you likely to find water underneath?\nOptions:\n- toilet\n- sink\n- jar\n- bridge\n- house\n Reasoning process: [/INST",
      "<s>[INST] <<SYS>>\nYou are an AI assistant. You will be given a task. You must generate a detailed and long answer.\n<</SYS>>\n\nContinue the following story.\n\nKay didn't have shoes that fit her feet properly. She only wore sneakers, because the \nChoose from: [I] shoes  fitted badly. [II] sneakers  fitted badly. [/INST]",
  ]
  print("init...")
  start = time.perf_counter()
  offline_inf.init_decode_state()
  end = time.perf_counter()
  print("init done in", end - start)

  input_data = []
  for i, prompt in enumerate(prompts):
    tokens, true_length = tokenizer.encode(prompt)
    input_data.append(
        offline_inference.InputData(
            id=str(i), tokens=jnp.array(tokens), true_length=true_length
        )
    )

  results = offline_inf.batch_inference(input_data)
  import ipdb; ipdb.set_trace()
  for ids, result in results.items():
    prompt = prompts[int(ids)]
    print(f"---- Input prompts are: {prompt}")
    print(f"---- RESPONSE tokens are: ")
    print(tokenizer.decode(result))


if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  app.run(main)
