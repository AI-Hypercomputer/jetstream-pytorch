import os
from typing import List
import random
import sys
import time
# import torch_xla2 first!
import torch_xla2  # pylint: disable
import jax
from jax import numpy as jnp
from absl import app, flags
from jetstream.engine import token_utils
from jetstream.core import server_lib
from jetstream.core.config_lib import ServerConfig, MetricsServerConfig
import torch
import numpy as np
from transformers import AutoTokenizer

from jetstream_pt import fetch_models
from jetstream_pt import environment, engine, quantize_model, torchjax
from jetstream_pt import config


FLAGS = flags.FLAGS

flags.DEFINE_string("model_id", "", "")
flags.DEFINE_integer("override_batch_size", 32, "The batch size")
flags.DEFINE_integer("max_input_length", 1024, "The batch size")
flags.DEFINE_integer("max_output_length", 1024, "The batch size")
flags.DEFINE_integer("port", 9000, "port to listen on")
flags.DEFINE_integer("threads", 64, "number of worker threads in thread pool")
flags.DEFINE_string(
    "benchmark_save_offline_result_to_file",
    "",
    "if set, then save the result to the given file name",
)
flags.DEFINE_bool(
    "internal_use_local_tokenizer", 0, "Use local tokenizer if set to True"
)
flags.DEFINE_bool("enable_model_warmup", False, "enable model warmup")


def shard_weights(env, weights, weight_shardings):
  """Shard weights according to weight_shardings"""
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
  quant_config = config.create_quantization_config_from_flags()
  env_data = fetch_models.construct_env_data_from_model_id(
      FLAGS.model_id,
      FLAGS.override_batch_size,
      FLAGS.max_input_length,
      FLAGS.max_output_length,
  )
  env = environment.JetEngineEnvironment(env_data)
  if FLAGS.internal_use_local_tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(env_data.checkpoint_path)
  else:
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id)
  env.hf_tokenizer = tokenizer
  model = fetch_models.instantiate_model_from_repo_id(FLAGS.model_id, env)
  # NOTE: this is assigned later because, the model should be constructed
  # as a float model first then quantized
  env.quant_config = quant_config
  if quant_config.enable_weight_quantization:
    quantize_model.quantize_model(model, quant_config)
  weight_shardings = model.get_sharding_annotations()
  sharded_weights = shard_weights(env, model.state_dict(), weight_shardings)
  env_data.quant_config = quant_config

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
  _check_model_id()
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
      enable_model_warmup=FLAGS.enable_model_warmup,
  )
  print("Started jetstream_server....")
  jetstream_server.wait_for_termination()


def _check_model_id():
  if FLAGS.model_id == "":
    print("Please specify model_id with --model_id")
    print("valid model ids are:")
    list_model()
    sys.exit(1)


def _run_prefill_time(
    pt_engine, params, decode_state, seqlen, profiler_started
):
  """Run prefill and measure time."""
  metadata = pt_engine.get_tokenizer()
  tokenizer = pt_engine.build_tokenizer(metadata)

  text = "This is a beautiful day"
  tokens, true_length = tokenizer.encode(
      text, is_bos=True, prefill_lengths=[seqlen]
  )

  for _ in range(3):
    prefill_result, _ = pt_engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = pt_engine.insert(
        prefill_result, decode_state, slot=jnp.int32(1)
    )

  nums = 5
  start = time.perf_counter()
  for i in range(nums):
    if i == nums - 1 and FLAGS.profiling_prefill and not profiler_started:
      jax.profiler.start_trace(FLAGS.profiling_output)
      profiler_started = True

    prefill_result, _ = pt_engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = pt_engine.insert(
        prefill_result, decode_state, slot=jnp.int32(i)
    )
  jax.block_until_ready(decode_state)

  end = time.perf_counter()
  return (end - start) / nums, decode_state, profiler_started


def interactive():
  """Run interactive"""
  _check_model_id()
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  pt_engine = create_engine(devices)

  start = time.perf_counter()
  params = pt_engine.load_params()
  print("Load params ", time.perf_counter() - start)

  metadata = pt_engine.get_tokenizer()
  tokenizer = pt_engine.build_tokenizer(metadata)
  max_output_length = 1024

  profiling_output = FLAGS.profiling_output
  profiling_prefill = (
      FLAGS.profiling_prefill
      and profiling_output is not None
      and profiling_output != ""
  )

  if profiling_prefill:
    jax.profiler.start_trace(profiling_output)

  decode_state = pt_engine.init_decode_state()

  if profiling_prefill:
    jax.profiler.stop_trace()

  prompts: List[str] = [
      # pylint: disable-next=all
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
    slot = random.randint(0, FLAGS.override_batch_size - 1)
    tokens, true_length = tokenizer.encode(prompt)

    print(f"---- Input prompts are: {prompt}")
    print(f"---- Encoded tokens are: {tokens}")

    # pylint: disable-next=all
    if profiling_prefill:
      jax.profiler.start_trace(profiling_output)

    prefill_result, _ = pt_engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    # pylint: disable-next=all
    decode_state = pt_engine.insert(prefill_result, decode_state, slot=slot)

    if profiling_prefill:
      jax.profiler.stop_trace()

    sampled_tokens_list = []
    print(f"---- Streaming decode started on #slot{slot}.")
    complete = np.zeros((1,), dtype=np.bool_)
    while True:
      if profiling_output:
        jax.profiler.start_trace(profiling_output)

      decode_state, result_tokens = pt_engine.generate(params, decode_state)
      result_tokens = result_tokens.convert_to_numpy()

      if profiling_output:
        jax.profiler.stop_trace()

      output, complete = token_utils.process_result_tokens(
          tokenizer=tokenizer,
          slot=slot,
          slot_max_length=max_output_length,
          result_tokens=result_tokens,
          complete=complete,
      )
      if complete[0]:
        break
      token_ids = output[0].token_ids
      sampled_tokens_list.extend(token_ids)

    print("---- All output tokens.")
    print(sampled_tokens_list)
    print("---- All output text.")
    print(tokenizer.decode(sampled_tokens_list))


def _save_benchmark_to_file(filename, prefill_times_ms, decode_time_ms):
  lines = (
      [
          " # Offline benchmark numbers",
          " ## Model: " + FLAGS.model_id,
          f" ## Batch size: {FLAGS.override_batch_size}",
          f" ## Quantize: {FLAGS.quantize_weights}",
          " |       | time (ms) |",
          " |-------|-----------|",
      ]
      + [f"| Prefill {x} | {y} |" for x, y in prefill_times_ms.items()]
      + [f"| Decode | {decode_time_ms} |"]
  )
  with open(filename, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
    f.flush()


def benchmark_offline():
  """function to run engine offline."""
  _check_model_id()
  devices = server_lib.get_devices()
  print(f"devices: {devices}")
  pt_engine = create_engine(devices)

  start = time.perf_counter()
  params = pt_engine.load_params()
  print("Load params ", time.perf_counter() - start)

  prefill_times = {}

  decode_state = pt_engine.init_decode_state()
  profiler_started = False
  # 16 .. 1024
  for exp in range(4, 11):
    batch = 2**exp
    runtime, decode_state, profiler_started = _run_prefill_time(
        pt_engine, params, decode_state, batch, profiler_started
    )
    prefill_times[batch] = runtime

  sampled_tokens_list = []

  for i in range(3):  # warm up
    # pylint: disable-next=all
    decode_state, sampled_tokens = pt_engine.generate(
        params=params, decode_state=decode_state
    )
    sampled_tokens_list.append(sampled_tokens)

  profiling_output = FLAGS.profiling_output
  print("======= decode starting ===")

  dec_times = []
  for i in range(10):
    if profiling_output and i == 7 and not profiler_started:
      jax.profiler.start_trace(profiling_output)
      profiler_started = True
    start = time.perf_counter()
    # pylint: disable-next=all
    decode_state, sampled_tokens = pt_engine.generate(params, decode_state)
    jax.block_until_ready(decode_state)
    sampled_tokens_list.append(sampled_tokens)
    end = time.perf_counter()
    dec_times.append(end - start)
    print(i, "decode time", (end - start))

  if profiler_started:
    jax.profiler.stop_trace()

  print("prefill ", prefill_times)
  avg_decode_times = sum(dec_times[2:]) / len(dec_times[2:])
  print("decode", avg_decode_times)

  prefill_times_ms = {k: v * 1000 for k, v in prefill_times.items()}
  decode_time_ms = sum(dec_times[2:]) * 1000 / 8

  largest_prefill = max(prefill_times.items())
  print("MAX tokens:", FLAGS.override_batch_size / avg_decode_times)

  time2 = (FLAGS.override_batch_size * FLAGS.max_decode_length) / (
      FLAGS.override_batch_size * largest_prefill[1]
      + FLAGS.max_decode_length * avg_decode_times
  )
  print("MAX tokens 2:", time2)

  if FLAGS.benchmark_save_offline_result_to_file:
    _save_benchmark_to_file(
        FLAGS.benchmark_save_offline_result_to_file,
        prefill_times_ms,
        decode_time_ms,
    )


def main():
  """Main function."""

  def main_real(argv):
    """Entry point"""
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    if len(argv) < 2:
      print("Invalid arguments. please specify 'list' or 'serve'")

    if argv[1] == "list":
      list_model()
    elif argv[1] == "serve":
      serve()
    elif argv[1] == "interactive":
      interactive()
    elif argv[1] == "benchmark_offline":
      benchmark_offline()
    else:
      print(
          "Invalid arguments. please specify 'list', 'serve', or 'interactive'."
      )

  app.run(main_real)
  return 0


if __name__ == "__main__":
  main()
