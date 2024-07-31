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

""" Runs a RayServe deployment with Jetstream interleave mode."""
import os
import time
from typing import AsyncIterator
from absl import app, flags

from ray import serve
from ray.serve.config import gRPCOptions

from jetstream.core import config_lib
from jetstream.core import orchestrator
from jetstream.core.config_lib import ServerConfig
from jetstream.core.proto import jetstream_pb2
from jetstream_pt import ray_engine
from jetstream_pt.config import FLAGS


flags.DEFINE_string("tpu_generation", "v4", "TPU generation")
flags.DEFINE_integer("tpu_chips", 16, "device tpu_chips")
flags.DEFINE_bool("enable_jax_profiler", False, "enable jax profiler")
flags.DEFINE_integer("jax_profiler_port", 9999, "port of JAX profiler server")
flags.DEFINE_integer("num_hosts", 4, "Number of TPU host", required=False)
flags.DEFINE_integer(
    "worker_chips", 4, "Number of TPU chips per worker", required=False
)


def create_head_resource_name(generation, tpu_chips):
  return f"TPU-{generation}-{tpu_chips}-head"


def create_engine(**kwargs):
  """create a pytorch engine"""
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  start = time.perf_counter()
  engine = ray_engine.create_pytorch_ray_engine(
      model_name=kwargs["model_name"],
      tokenizer_path=kwargs["tokenizer_path"],
      ckpt_path=kwargs["ckpt_path"],
      bf16_enable=kwargs["bf16_enable"],
      param_size=kwargs["param_size"],
      context_length=kwargs["context_length"],
      batch_size=kwargs["batch_size"],
      quantize_weights=kwargs["quantize_weights"],
      quantize_kv=kwargs["quantize_kv"],
      max_cache_length=kwargs["max_cache_length"],
      sharding_config=kwargs["sharding_config"],
      num_hosts=kwargs["num_hosts"],
      worker_chips=kwargs["worker_chips"],
      tpu_chips=kwargs["tpu_chips"],
      enable_jax_profiler=kwargs["enable_jax_profiler"],
      jax_profiler_port=kwargs["jax_profiler_port"],
  )

  print("Initialize engine", time.perf_counter() - start)
  return engine


@serve.deployment
class JetStreamDeployment:

  def __init__(self, **kwargs):
    os.environ["XLA_FLAGS"] = (
        "--xla_dump_to=/tmp/xla_logs --xla_dump_hlo_as_text"
    )
    devices = []
    for i in range(kwargs["tpu_chips"]):
      devices.append(i)

    print(f"devices: {devices}")

    self.engine = create_engine(**kwargs)
    server_config = ServerConfig(
        interleaved_slices=(f"tpu={len(devices)}",),
        interleaved_engine_create_fns=(lambda a: self.engine,),
    )

    engines = config_lib.get_engines(server_config, devices=devices)
    prefill_params = [pe.load_params() for pe in engines.prefill_engines]
    generate_params = [ge.load_params() for ge in engines.generate_engines]
    shared_params = [ie.load_params() for ie in engines.interleaved_engines]
    print("Loaded all weights.")

    self.driver = orchestrator.Driver(
        prefill_engines=engines.prefill_engines + engines.interleaved_engines,
        generate_engines=engines.generate_engines + engines.interleaved_engines,
        prefill_params=prefill_params + shared_params,
        generate_params=generate_params + shared_params,
        interleaved_mode=True,
        jax_padding=False,
        metrics_collector=None,
        is_ray_backend=True,
    )

    self.orchestrator = orchestrator.LLMOrchestrator(driver=self.driver)

    print("Started jetstream driver....")

  async def Decode(
      self, request: jetstream_pb2.DecodeRequest
  ) -> AsyncIterator[jetstream_pb2.DecodeResponse]:

    return self.orchestrator.Decode(request)


def main(_argv):
  resource_name = create_head_resource_name(
      FLAGS.tpu_generation, FLAGS.tpu_chips
  )
  print(f"Using head resource {resource_name}")
  deployment = JetStreamDeployment.options(
      ray_actor_options={"resources": {resource_name: 1}}
  ).bind(
      tpu_chips=FLAGS.tpu_chips,
      worker_chips=FLAGS.worker_chips,
      num_hosts=FLAGS.num_hosts,
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
  )

  grpc_port = 8888
  grpc_servicer_functions = [
      "jetstream.core.proto.jetstream_pb2_grpc.add_OrchestratorServicer_to_server",
  ]
  serve.start(
      grpc_options=gRPCOptions(
          port=grpc_port,
          grpc_servicer_functions=grpc_servicer_functions,
      ),
  )

  serve.run(deployment)


if __name__ == "__main__":
  app.run(main)
