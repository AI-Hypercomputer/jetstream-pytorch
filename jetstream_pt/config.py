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

from jetstream.core.config_lib import ServerConfig
from jetstream_pt.engine import create_pytorch_engine


# pylint: disable-next=all
def create_config(
    devices,
    tokenizer_path,
    ckpt_path,
    bf16_enable,
    param_size,
    context_length,
    batch_size,
    platform,
):
  """Create a server config"""

  def func():
    return create_pytorch_engine(
        devices=devices,
        tokenizer_path=tokenizer_path,
        ckpt_path=ckpt_path,
        bf16_enable=bf16_enable,
        param_size=param_size,
        context_length=context_length,
        batch_size=batch_size,
    )

  return ServerConfig(
      interleaved_slices=(platform,),
      interleaved_engine_create_fns=(func,),
  )
