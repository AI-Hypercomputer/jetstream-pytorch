from collections import defaultdict
from typing import Any, Iterable, Optional, Union, Tuple

import numpy as np
import jax
import ray
from ray.util.accelerators import tpu

from jetstream.engine import engine_api, tokenizer_pb2
from jetstream_pt.ray_worker import PyTorchRayWorker

Params = Any
Prefix = Any
DecodeState = Any
NpPrefix = Any


class PyTorchRayEngine(engine_api.Engine):
  """Ray PyTorch Engine Implementation for Multi-Host Inference Serving.
  Key Features:
  1. Manages all Ray workers.
  2. Initializes model parameters for each Ray worker.
  3. Routes incoming inference requests to Ray workers.
  4. Collects token responses from the Ray workers.
  """

  def __init__(
      self,
      engine_workers: Iterable[PyTorchRayWorker],
      tokenizer_path: str,
      context_length: int,
      batch_size: int,
      is_disaggregated: bool = False,
      pod_slice_name: str = None,
  ):
    self.engine_workers = engine_workers
    self.tokenizer_path = tokenizer_path
    self.context_length = context_length
    self.batch_size = batch_size
    self.is_disaggregated = is_disaggregated
    self.pod_slice_name = pod_slice_name

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

  def interleave_prefill(
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
    # The prefill function does not return any values;
    # the worker itself manages and maintains the prefill states.
    return None

  def disaggregated_prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: np.ndarray,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> Prefix:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.prefill_ray_disaggregation.remote(
          params=params,
          existing_prefix=existing_prefix,
          padded_tokens=padded_tokens,
          true_length=true_length,
      )
      all_outputs.append(output)
    results = ray.get(all_outputs)
    return results[0]  

  def prefill(
      self,
      *,
      params: Any,  # Weights
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: np.ndarray,  # PrefillInputs[np.ndarray],
      true_length: int,
  ) -> Prefix:
    result = None
    if self.is_disaggregated:
      result = self.disaggregated_prefill(
      params=params,
      existing_prefix=existing_prefix,
      padded_tokens=padded_tokens,
      true_length=true_length,
      ) 
    else:
      result= self.interleave_prefill(
      params=params,
      existing_prefix=existing_prefix,
      padded_tokens=padded_tokens,
      true_length=true_length,
      )

    return result
  
  
  def transfer(self, np_prefix: NpPrefix) -> Any:
    all_outputs = []
    np_prefix_ref = ray.put(np_prefix)
    for worker in self.engine_workers:
      output = worker.transfer.remote(np_prefix_ref)
      all_outputs.append(output)
    results = ray.get(all_outputs)

    return results[0]  
    
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
    # The insert function does not return any values;
    # the worker itself manages and maintains the DecodeState.
    return None

  def generate(
      self, params: Any, decode_state: DecodeState
  ) -> tuple[None, engine_api.ResultTokens]:
    all_outputs = []
    for worker in self.engine_workers:
      output = worker.generate_ray.remote(
          params=params, decode_state=decode_state
      )
      all_outputs.append(output)
    # All workers performed an all_gather operation. Since the results are
    # identical across all workers, the result from worker 0 is returned.
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
def create_pytorch_ray_engine(
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
    sharding_config=None,
    is_disaggregated: bool = False,
    num_hosts: int = 0,
    decode_pod_slice_name: str = None,
) -> Any:

  supported_models = ["llama-2", "llama-3", "gemma"]
  if model_name not in supported_models:
    raise NotImplementedError(
        f"Model name should be one of{','.join(supported_models)}"
    )
  ray.init(ignore_reinit_error=True)
  pod_name = tpu.get_current_pod_name()
  num_hosts = num_hosts if is_disaggregated else tpu.get_current_pod_worker_count()
  print(f"pod_name:{pod_name}, number of host: {num_hosts}")
  assert (
      pod_name is not None
  ), f"TPU pod name (current value:{pod_name}) can not be None"
  assert (
      num_hosts > 0
  ), f"num_hosts (current value {num_hosts}) should be a positive number"
  # pylint: disable-next=all
  engine_worker_with_tpu_resource = PyTorchRayWorker.options(
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
        sharding_config=sharding_config,
    )
    engine_workers.append(engine_worker)

  if not is_disaggregated:
    return PyTorchRayEngine(
      engine_workers=engine_workers,
      tokenizer_path=tokenizer_path,
      context_length=context_length,
      batch_size=batch_size,
      )

  workers_dict = defaultdict(list)  
  for worker in engine_workers:
    pod_slice_name = ray.get(worker.pod_slice_name.remote())
    workers_dict[pod_slice_name].append(worker)  

  prefill_engine = PyTorchRayEngine(
      engine_workers=workers_dict[pod_name],
      tokenizer_path=tokenizer_path,
      context_length=context_length,
      batch_size=batch_size,
      is_disaggregated=is_disaggregated,
      pod_slice_name=pod_name,
  )
  decode_engine = PyTorchRayEngine(
      engine_workers=workers_dict[decode_pod_slice_name],
      tokenizer_path=tokenizer_path,
      context_length=context_length,
      batch_size=batch_size,
      is_disaggregated=is_disaggregated,
      pod_slice_name=decode_pod_slice_name,
  )
  return (prefill_engine, decode_engine)
