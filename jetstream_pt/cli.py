import sys

# import torch_xla2 first!
import torch_xla2  # pylint: disable
import jax
from absl import app, flags
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig, MetricsServerConfig
import torch

from jetstream_pt import fetch_models
from jetstream_pt import environment, engine, quantize_model, torchjax
from jetstream_pt import config


FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", "", "")
flags.DEFINE_integer("override_batch_size", 32, "The batch size")
flags.DEFINE_integer("max_input_length", 1024, "The batch size")
flags.DEFINE_integer("max_output_length", 1024, "The batch size")
flags.DEFINE_string("data_type", "bfloat16", "")
flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")


def shard_weights(env, weights, weight_shardings):
  """Shard weights according to weight_shardings"""
  for k, v in weight_shardings.items():
    print("SHARDING", k, v)
  sharded = {}
  for key, val in weights.items():
    sharding = env.sharding_by_axis(weight_shardings.get(key, -1))
    with jax.default_device(jax.devices("cpu")[0]):
      arr = torch_xla2.tensor.t2j(val)
    arr = jax.device_put(arr, sharding)
    sharded[key] = torchjax.to_torch(arr)
  return sharded


def create_engine(devices):
  """Create Pytorch engine from flags"""
  torch.set_default_dtype(torch.bfloat16)
  env_data = fetch_models.construct_env_data_from_model_id(
      FLAGS.model_id,
      FLAGS.override_batch_size,
      FLAGS.max_input_length,
      FLAGS.max_output_length,
      FLAGS.data_type == "int8",
  )
  env = environment.JetEngineEnvironment(env_data)
  model = fetch_models.instantiate_model_from_repo_id(FLAGS.model_id, env)

  weight_shardings = model.get_sharding_annotations()
  sharded_weights = shard_weights(env, model.state_dict(), weight_shardings)

  quant_config = config.create_quantization_config_from_flags()
  if quant_config.enable_weight_quantization:
    model.load_state_dict(sharded_weights, assign=True, strict=False)
    quantize_model.quantize_model(model, quant_config)
    sharded_weights = model.state_dict()

  return engine.PyTorchEngine(
      pt_model=model,
      env=env,
      weights=torchjax.from_torch_with_copy(sharded_weights),
  )


def list_model():
  """Print list of models."""
  for model_id in fetch_models.model_id_to_class:
    print(model_id)


def serve():
  """Run gRPC server."""
  if FLAGS.model_id == "":
    print("Please specify model_id with --model_id")
    print("valid model ids are:")
    list_model()
    sys.exit(1)
  devices = server_lib.get_devices()
  print(f"devices: {devices}")

  server_config = ServerConfig(
      interleaved_slices=(f"tpu={len(jax.devices())}",),
      interleaved_engine_create_fns=[create_engine],
  )
  print(f"server_config: {server_config}")

  metrics_server_config: MetricsServerConfig | None = None

  # We separate credential from run so that we can unit test it with local credentials.
  # We would like to add grpc credentials for OSS.
  jetstream_server = server_lib.run(
      threads=FLAGS.threads,
      port=FLAGS.port,
      config=server_config,
      devices=devices,
      metrics_server_config=metrics_server_config,
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()


def interactive():
  """Run interactive"""
  raise RuntimeError("Not implemented")


def main(argv):
  """Entry point"""
  if len(argv) < 2:
    print("Invalid arguments. please specify 'list' or 'serve'")

  if argv[1] == "list":
    list_model()
    return

  if argv[1] == "serve":
    serve()
    return

  if argv[1] == "interative":
    list_model()
    return


if __name__ == "__main__":
  app.run(main)
