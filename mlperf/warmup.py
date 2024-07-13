import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
import json
import random
import time
from typing import Any, AsyncGenerator, Optional
import os


import grpc
from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc
from jetstream.engine.token_utils import load_vocab
from jetstream.third_party.llama3 import llama3_tokenizer
import numpy as np
from tqdm.asyncio import tqdm  # pytype: disable=pyi-error
import pandas


@dataclass
class InputRequest:
  prompt: str = ""
  prompt_len: int = 0
  output: str = ""
  output_len: int = 0
  sample_idx: int = -1


@dataclass
class RequestFuncOutput:
  input_request: Optional[InputRequest] = None
  generated_token_list: list[str] = field(default_factory=list)
  generated_text: str = ""
  success: bool = False
  latency: float = 0
  ttft: float = 0
  prompt_len: int = 0

  # Flatten the structure and return only the necessary results
  def to_dict(self):
    return {
        "prompt": self.input_request.prompt,
        "original_output": self.input_request.output,
        "generated_text": self.generated_text,
        "success": self.success,
        "latency": self.latency,
        "prompt_len": self.prompt_len,
        "sample_idx": self.input_request.sample_idx,
    }


async def grpc_async_request(
    api_url: str, request: Any
) -> tuple[list[str], float, float]:
  """Send grpc synchronous request since the current grpc server is sync."""
  options = [("grpc.keepalive_timeout_ms", 10000)]
  async with grpc.aio.insecure_channel(api_url, options=options) as channel:
    stub = jetstream_pb2_grpc.OrchestratorStub(channel)
    print("Making request")
    ttft = 0
    token_list = []
    request_start_time = time.perf_counter()
    response = stub.Decode(request)
    async for resp in response:
      if ttft == 0:
        ttft = time.perf_counter() - request_start_time
      token_list.extend(resp.stream_content.samples[0].token_ids)
    latency = time.perf_counter() - request_start_time
    print("Done request: ", latency)
    return token_list, ttft, latency


async def send_request(
    api_url: str,
    tokenizer: Any,
    input_request: InputRequest,
    pbar: tqdm,
    session_cache: str,
    priority: int,
) -> RequestFuncOutput:
  """Send the request to JetStream server."""
  # Tokenization on client side following MLPerf standard.
  token_ids = np.random.randint(0, 1000, input_request.request_len)
  request = jetstream_pb2.DecodeRequest(
      session_cache=session_cache,
      token_content=jetstream_pb2.DecodeRequest.TokenContent(
          token_ids=token_ids
      ),
      priority=priority,
      max_tokens=input_request.output_len,
  )
  output = RequestFuncOutput()
  output.input_request = input_request
  output.prompt_len = input_request.prompt_len
  generated_token_list, ttft, latency = await grpc_async_request(
      api_url, request
  )
  output.ttft = ttft
  output.latency = latency
  output.generated_token_list = generated_token_list
  # generated_token_list is a list of token ids, decode it to generated_text.
  output.generated_text = ""
  output.success = True
  if pbar:
    pbar.update(1)
  return output


async def benchmark(
    api_url: str,
    max_length: int,
    tokenizer: Any = None,
    request_rate: float = 0,
    disable_tqdm: bool = False,
    session_cache: str = "",
    priority: int = 100,
):
  """Benchmark the online serving performance."""

  print(f"Traffic request rate: {request_rate}")

  benchmark_start_time = time.perf_counter()
  tasks = []
  interesting_buckets = [
      4,
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
  ]

  for length in interesting_buckets:
    if length > max_length:
      break
    request = InputRequest()
    request.request_len = length
    print("send request of length", request.request_len)
    tasks.append(
        asyncio.create_task(
            send_request(
                api_url=api_url,
                tokenizer=None,
                input_request=request,
                pbar=None,
                session_cache=session_cache,
                priority=priority,
            )
        )
    )
  outputs = await asyncio.gather(*tasks)

  benchmark_duration = time.perf_counter() - benchmark_start_time
  return benchmark_duration, outputs


def main(args: argparse.Namespace):
  print(args)
  random.seed(args.seed)
  np.random.seed(args.seed)
  api_url = f"{args.server}:{args.port}"

  benchmark_result, request_outputs = asyncio.run(
      benchmark(api_url=api_url, max_length=args.max_length)
  )
  print("DURATION:", benchmark_result)


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
      description="Benchmark the online serving throughput."
  )
  parser.add_argument(
      "--server",
      type=str,
      default="0.0.0.0",
      help="Server address.",
  )
  parser.add_argument("--seed", type=int, default=0)

  parser.add_argument("--port", type=str, default=9000)
  parser.add_argument("--max-length", type=int, default=512)

  parsed_args = parser.parse_args()
  main(parsed_args)
