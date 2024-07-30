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

"""Runs a pytorch server with ray."""
import os
import time
from typing import Sequence
from absl import app, flags

# import torch_xla2 first!
import torch_xla2  # pylint: disable
import jax
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig
from jetstream_pt import ray_engine
from jetstream_pt.config import FLAGS

flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")
flags.DEFINE_string(
    "config",
    "InterleavedCPUTestServer",
    "available servers",
)
flags.DEFINE_integer("prometheus_port", 0, "")
flags.DEFINE_integer("tpu_chips", 16, "device tpu_chips")

flags.DEFINE_bool("enable_jax_profiler", False, "enable jax profiler")
flags.DEFINE_integer("jax_profiler_port", 9999, "port of JAX profiler server")

flags.DEFINE_bool(
    "is_disaggregated", False, "Disaggregated serving if it's True"
)

flags.DEFINE_integer("num_hosts", 0, "Number of TPU host", required=False)

flags.DEFINE_integer(
    "worker_chips", 4, "Number of TPU chips per worker", required=False
)

flags.DEFINE_string("decode_pod_slice_name", "", "Decode pod slice name")


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
      enable_jax_profiler=FLAGS.enable_jax_profiler,
      jax_profiler_port=FLAGS.jax_profiler_port,
      num_hosts=FLAGS.num_hosts,
      worker_chips=FLAGS.worker_chips,
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


def create_disaggregated_engine():
  """create a pytorch engine"""
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  start = time.perf_counter()
  prefill_engine_list, decode_engine_list = (
      ray_engine.create_pytorch_ray_engine(
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
          enable_jax_profiler=FLAGS.enable_jax_profiler,
          jax_profiler_port=FLAGS.jax_profiler_port,
          is_disaggregated=FLAGS.is_disaggregated,
          num_hosts=FLAGS.num_hosts,
          decode_pod_slice_name=FLAGS.decode_pod_slice_name,
      )
  )

  print("Initialize engine", time.perf_counter() - start)
  return (prefill_engine_list, decode_engine_list)


# pylint: disable-next=all
def main(argv: Sequence[str]):
  del argv
  os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_logs --xla_dump_hlo_as_text"
  devices = []
  for i in range(FLAGS.tpu_chips):
    devices.append(i)

  print(f"devices: {devices}")

  if FLAGS.is_disaggregated:
    prefill_engine_list, decode_engine_list = create_disaggregated_engine()
    chips = int(len(devices) / 2)
    server_config = ServerConfig(
        prefill_slices=(f"tpu={chips}",),
        prefill_engine_create_fns=(lambda a: prefill_engine_list[0],),
        generate_slices=(f"tpu={chips}",),
        generate_engine_create_fns=(lambda a: decode_engine_list[0],),
        is_ray_backend=True,
    )

  else:
    engine = create_engine()
    server_config = ServerConfig(
        interleaved_slices=(f"tpu={len(devices)}",),
        interleaved_engine_create_fns=(lambda a: engine,),
    )

  print(f"server_config: {server_config}")

  jetstream_server = server_lib.run(
      threads=FLAGS.threads,
      port=FLAGS.port,
      config=server_config,
      devices=devices,
      jax_padding=False,  # Jax_padding must be set as False
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()


if __name__ == "__main__":
  app.run(main)
