import requests
import os
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

prompt = "who is going to win Euro 2024?"
prompt = "how do I cook fried rice?"

channel = grpc.insecure_channel("localhost:8888")
stub = jetstream_pb2_grpc.OrchestratorStub(channel)

request = jetstream_pb2.DecodeRequest(
    text_content=jetstream_pb2.DecodeRequest.TextContent(
        text=prompt
    ),
    priority=0,
    max_tokens=4,
)

response = stub.Decode(request)
output = []
for resp in response:
  #print(resp)
  output.extend(resp.stream_content.samples[0].text)

text_output = "".join(output)
print(f"Prompt: {prompt}")
print(f"Response: {text_output}")
