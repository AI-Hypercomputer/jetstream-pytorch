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

"""Runs a pytorch server."""
import os
from typing import Sequence

import jax
import jetstream_pt
from absl import app, flags
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig
from jetstream_pt.config import FLAGS, create_engine_from_config_flags

flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)


# pylint: disable-next=all
def main(argv: Sequence[str]):
  del argv
  os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_logs --xla_dump_hlo_as_text"
  # No devices for local cpu test. A None for prefill and a None for generate.
  devices = server_lib.get_devices()
  print(f"devices: {devices}")

  engine = create_engine_from_config_flags()

  server_config = ServerConfig(
      interleaved_slices=(f"tpu={len(jax.devices())}",),
      interleaved_engine_create_fns=(lambda a: engine,),
  )
  print(f"server_config: {server_config}")

  # We separate credential from run so that we can unit test it with local credentials.
  # We would like to add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=FLAGS.threads,
      port=FLAGS.port,
      config=server_config,
      devices=devices,
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)
