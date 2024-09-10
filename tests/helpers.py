import jax
import torch
import torch_xla2
from jetstream_pt.third_party.llama import model_args
from jetstream_pt.third_party.mixtral import config as mixtral_config
from jetstream_pt import environment


# pylint: disable-next=all
def make_env_tiny(bf16_enable=True, env_data_update_fn=lambda _: None):
  torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
  torch.set_default_dtype(torch_dtype)
  jax.config.update("jax_dynamic_shapes", False)
  jax.config.update("jax_traceback_filtering", "off")
  config = model_args.get_model_args("llama-2-tiny", 128, 1, True)
  environment_data = environment.JetEngineEnvironmentData()
  environment_data.max_input_sequence_length = 128
  environment_data.max_input_sequence_length = 128
  environment_data.cache_sequence_length = 128
  environment_data.bf16_enable = bf16_enable
  environment_data.model_type = "llama-2-tiny"
  environment_data.batch_size = 1
  environment_data.num_layers = config.n_layers
  environment_data.cache_shape = (
      1,
      config.n_kv_heads,
      environment_data.cache_sequence_length,
      config.dim // config.n_heads,
  )
  environment_data.n_reps = config.n_heads // config.n_kv_heads
  environment_data.testing = True
  env_data_update_fn(environment_data)
  env = environment.JetEngineEnvironment(environment_data)
  env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
  return env, config


# pylint: disable-next=all
def make_mixtral_env(bf16_enable=True):
  torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
  torch.set_default_dtype(torch_dtype)
  jax.config.update("jax_dynamic_shapes", False)
  jax.config.update("jax_traceback_filtering", "off")
  config = mixtral_config.ModelArgs.from_name("Mixtral-tiny")
  environment_data = environment.JetEngineEnvironmentData()
  environment_data.max_input_sequence_length = 128
  environment_data.cache_sequence_length = 128
  environment_data.bf16_enable = bf16_enable
  environment_data.model_type = "mixtral"
  environment_data.batch_size = 1
  environment_data.num_layers = config.n_layer
  environment_data.cache_shape = (
      1,
      config.n_local_heads,
      environment_data.cache_sequence_length,
      config.dim // config.n_head,
  )
  environment_data.n_reps = config.n_head // config.n_local_heads
  env = environment.JetEngineEnvironment(environment_data)
  env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
  return env, config


# pylint: disable-next=all
def to_xla_tensor(tree):
  return torch_xla2.default_env().to_xla(tree)


# pylint: disable-next=all
def call_xla_model(model, weights, args):
  with jax.default_device(jax.devices("cpu")[0]):
    xla_weights, xla_inputs = to_xla_tensor((weights, args))
    with torch_xla2.default_env():
      result = torch.func.functional_call(model, xla_weights, xla_inputs)
    result_torch = torch_xla2.tensor.j2t(result.jax())
    return result_torch


# pylint: disable-next=all
def make_page_attention_env_tiny(
    bf16_enable=True, env_data_update_fn=lambda _: None
):
  torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
  torch.set_default_dtype(torch_dtype)
  jax.config.update("jax_dynamic_shapes", False)
  jax.config.update("jax_traceback_filtering", "off")
  config = model_args.get_model_args("llama-2-tiny", 128, 1, True)
  environment_data = environment.JetEngineEnvironmentData()
  environment_data.paged_attention_page_size = 32
  environment_data.paged_attention_total_num_pages = 16
  environment_data.block_size = 64
  environment_data.max_input_sequence_length = 128
  environment_data.max_input_sequence_length = 128
  environment_data.cache_sequence_length = 128
  environment_data.bf16_enable = bf16_enable
  environment_data.model_type = "llama-2-tiny"
  environment_data.batch_size = 1
  environment_data.num_layers = config.n_layers
  environment_data.cache_shape = (
      config.n_kv_heads,
      environment_data.paged_attention_total_num_pages,
      environment_data.paged_attention_page_size,
      config.dim // config.n_heads,
  )
  environment_data.testing = True
  env_data_update_fn(environment_data)
  env = environment.JetEngineEnvironment(environment_data)
  env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
  return env, config
