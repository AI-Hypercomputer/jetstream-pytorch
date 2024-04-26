from typing import Any, Optional, Union

import numpy as np
import jax
import ray
from ray.util.accelerators import tpu

from jetstream.engine import engine_api, tokenizer_pb2
from jetstream_pt.ray_engine_worker import PyTorchEngineRayWorker

Params = Any
Prefix = Any
DecodeState = Any


class PyTorchEngineRayMaster(engine_api.Engine):
  """Ray engine master to orchestrate requests and collect token response"""

  def __init__(
      self, engine_workers, tokenizer_path, context_length, batch_size
  ):
    self.engine_workers = engine_workers
    self.tokenizer_path = tokenizer_path
    self.context_length = context_length
    self.batch_size = batch_size

  # pylint: disable-next=all
  def load_params(self) -> Params:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.load_params_ray.remote()
      all_outputs.append(output)
    _ = ray.get(all_outputs)
    return None

  # pylint: disable-next=all
  def init_decode_state(
      self,
  ) -> DecodeState:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.init_decode_state_ray.remote()
      all_outputs.append(output)
    _ = ray.get(all_outputs)
    return None

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: np.ndarray,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> Prefix:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.prefill_ray.remote(
          params=params,
          existing_prefix=existing_prefix,
          padded_tokens=padded_tokens,
          true_length=true_length,
      )
      all_outputs.append(output)
    _ = ray.get(all_outputs)

  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.insert_ray.remote(
          prefix=prefix, decode_state=decode_state, slot=slot
      )
      all_outputs.append(output)
    _ = ray.get(all_outputs)

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[None, engine_api.ResultTokens]:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.generate_ray.remote(
          params=params, decode_state=decode_state
      )
      all_outputs.append(output)
    state, result_tokens = ray.get(all_outputs)[0]
    return state, result_tokens

  # pylint: disable-next=all
  def get_tokenizer(self) -> tokenizer_pb2.TokenizerParameters:
    # pylint: disable-next=all
    return tokenizer_pb2.TokenizerParameters(path=self.tokenizer_path)

  @property
  def max_concurrent_decodes(self) -> int:
    return self.batch_size

  @property
  def samples_per_slot(self) -> int:
    return 1

  @property
  def max_prefill_length(self) -> int:
    return self.context_length

  @property
  def colocated_cpus(self) -> Union[list[engine_api.CpuDevices], None]:
    return jax.devices("cpu")[0]

  def get_prefix_destination_sharding(self) -> Prefix:
    "No implementation"
    return None

  @property
  def mesh(self):
    "No implementation"
    return None


# pylint: disable-next=all
def create_pytorch_engine_ray_master(
    tokenizer_path: str,
    ckpt_path: Optional[str] = None,
    samples_per_slot: int = 1,
    bf16_enable: bool = False,
    param_size: str = "7b",
    context_length: int = 1024,
    batch_size: int = 1,
    max_decode_length: int = 4096,
    model_name="llama",
    quantize_weights=False,
    quantize_kv=False,
    max_cache_length=1024,
) -> PyTorchEngineRayMaster:

  ray.init(ignore_reinit_error=True)
  pod_name = tpu.get_current_pod_name()
  num_hosts = tpu.get_current_pod_worker_count()
  print(f"pod_name:{pod_name}, number of host: {num_hosts}")
  # pylint: disable-next=all
  engine_worker_with_tpu_resource = PyTorchEngineRayWorker.options(
      resources={"TPU": 4}
  )
  engine_workers = []
  for _ in range(num_hosts):
    engine_worker = engine_worker_with_tpu_resource.remote(
        tokenizer_path=tokenizer_path,
        ckpt_path=ckpt_path,
        samples_per_slot=samples_per_slot,
        bf16_enable=bf16_enable,
        param_size=param_size,
        context_length=context_length,
        batch_size=batch_size,
        max_decode_length=max_decode_length,
        model_name=model_name,
        quantize_weights=quantize_weights,
        quantize_kv=quantize_kv,
        max_cache_length=max_cache_length,
    )
    engine_workers.append(engine_worker)
  engine_master = PyTorchEngineRayMaster(
      engine_workers=engine_workers,
      tokenizer_path=tokenizer_path,
      context_length=context_length,
      batch_size=batch_size,
  )
  return engine_master
