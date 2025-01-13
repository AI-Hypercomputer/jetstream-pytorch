# Jetstream-PyTorch
JetStream Engine implementation in PyTorch

# Latest Release:

The latest release version is tagged with `jetstream-v0.2.3`. If you are running the release version
Please follow the README of the that version here:
https://github.com/google/jetstream-pytorch/blob/jetstream-v0.2.3/README.md

Commandline Flags might have changed between the release version to HEAD.

# Outline

1. Ssh to Cloud TPU VM (using v5e-8 TPU VM)
   a. Create a Cloud TPU VM if you haven’t
2. Download jetstream-pytorch github repo
3. Run the server
4. Run benchmarks
5. Typical Errors

# Ssh to Cloud TPU VM (using v5e-8 TPU VM)

```bash
gcloud compute config-ssh
gcloud compute tpus tpu-vm ssh "your-tpu-vm" --project "your-project" --zone "your-project-zone"
```
## Create a Cloud TPU VM in a GCP project  if you haven’t
Follow the steps in
* https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm

# Clone repo and install dependencies 

## Get the jetstream-pytorch code
```bash
git clone https://github.com/google/jetstream-pytorch.git
git checkout jetstream-v0.2.4
```

(optional) Create a virtual env using `venv` or `conda` and activate it.

## 2. Run installation script:

```bash
cd jetstream-pytorch
source install_everything.sh
```


# Run jetstream pytorch

## List out supported models

```
jpt list
```

This will print out list of support models and variants:

```
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-chat-hf
meta-llama/Llama-2-13b-hf
meta-llama/Llama-2-70b-hf
meta-llama/Llama-2-70b-chat-hf
meta-llama/Meta-Llama-3-8B
meta-llama/Meta-Llama-3-8B-Instruct
meta-llama/Meta-Llama-3-70B
meta-llama/Meta-Llama-3-70B-Instruct
meta-llama/Llama-3.1-8B
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.2-1B
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.3-70B
meta-llama/Llama-3.3-70B-Instruct
google/gemma-2b
google/gemma-2b-it
google/gemma-7b
google/gemma-7b-it
mistralai/Mixtral-8x7B-v0.1
mistralai/Mixtral-8x7B-Instruct-v0.1
```

To run jetstream-pytorch server with one model:

```
jpt serve --model_id meta-llama/Meta-Llama-3-8B-Instruct
```

If it's the first time you run this model, it will download weights from 
HuggingFace. 

HuggingFace's Llama3 weights are gated, so you need to either run 
`huggingface-cli login` to set your token, OR, pass your hf_token explicitly.

To pass hf token explicitly, add `--hf_token` flag
```
jpt serve --model_id meta-llama/Meta-Llama-3-8B-Instruct --hf_token=...
```

To login using huggingface hub, run:

```
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```
Then follow its prompt.

After the weights are downloaded,
Next time when you run this `--hf_token` will no longer be required.

To run this model in `int8` quantization, add `--quantize_weights=1`.
Quantization will be done on the flight as the weight loads.

Weights downloaded from HuggingFace will be stored by default in `checkpoints` folder.
in the place where `jpt` is executed.

You can change where the weights are stored with `--working_dir` flag.

If you wish to use your own checkpoint, then, place them inside 
of the `checkpoints/<org>/<model>/hf_original` dir (or the corresponding subdir in `--working_dir`). For example,
Llama3 checkpoints will be at `checkpoints/meta-llama/Llama-2-7b-hf/hf_original/*.safetensors`. You can replace these files with modified
weights in HuggingFace format. 

## Send one request

Jetstream-pytorch uses gRPC for handling requests, the script below demonstrates how to
send gRPC in Python. You can also use other gPRC clients.

```python
import requests
import os
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

prompt = "What are the top 5 languages?"

channel = grpc.insecure_channel("localhost:8888")
stub = jetstream_pb2_grpc.OrchestratorStub(channel)

request = jetstream_pb2.DecodeRequest(
    text_content=jetstream_pb2.DecodeRequest.TextContent(
        text=prompt
    ),
    priority=0,
    max_tokens=2000,
)

response = stub.Decode(request)
output = []
for resp in response:
  output.extend(resp.stream_content.samples[0].text)

text_output = "".join(output)
print(f"Prompt: {prompt}")
print(f"Response: {text_output}")
```


# Run the server with ray
Below are steps run server with ray:
1. Ssh to Cloud Multiple Host TPU VM (v5e-16 TPU VM)
2. Step 2 to step 5 in Outline 
3. Setup ray cluster 
4. Run server with ray

## Setup Ray Cluster 
Login host 0 VM, start ray head with below command: 

```bash

ray start --head

```

Login other host VMs, start ray head with below command:

```bash

ray start --address='$ip:$port'

```

Note: Get address ip and port information from ray head.

## Run server with ray

Here is an example to run the server with ray for llama2 7B model:

```bash
export DISABLE_XLA2_PJRT_TEST="true"
python run_server_with_ray.py --tpu_chips=16 --num_hosts=4 --worker_chips=4 -model_name=$model_name          --size=7b --batch_size=96 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"
```

# Run benchmark
Start the server and then go to the deps/JetStream folder (downloaded during `install_everything.sh`)

```bash
cd deps/JetStream
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 2000  --dataset-path  $dataset_path --dataset sharegpt --save-request-outputs --warmup-mode=sampled --model=$model_name
```
Please look at `deps/JetStream/benchmarks/README.md` for more information.



## Run server with Ray Serve

### Prerequisites

If running on GKE:

1. Follow instructions on [this link](https://github.com/GoogleCloudPlatform/ai-on-gke/tree/main/ray-on-gke/guides/tpu) to setup a GKE cluster and the TPU webhook.
2. Follow instructions
   [here](https://cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/cloud-storage-fuse-csi-driver)
   to enable GCSFuse for your cluster. This will be needed to store the
   converted weights.
3. Deploy one of the sample Kuberay cluster configurations:
```bash
kubectl apply -f kuberay/manifests/ray-cluster.tpu-v4-singlehost.yaml
```
or
```bash
kubectl apply -f kuberay/manifests/ray-cluster.tpu-v4-multihost.yaml
```


### Start a Ray Serve deployment

Single-host (Llama2 7B):

```bash
export RAY_ADDRESS=http://localhost:8265

kubectl port-forward svc/example-cluster-kuberay-head-svc 8265:8265 &

ray job submit --runtime-env-json='{"working_dir": "."}' -- python run_ray_serve_interleave.py  --tpu_chips=4 --num_hosts=1 --size=7b --model_name=llama-2 --batch_size=32 --max_cache_length=2048 --tokenizer_path=/llama/tokenizer.model --checkpoint_path=/llama/ckpt --quantize_weights=True --quantize_type="int8_per_channel" --quantize_kv_cache=True --sharding_config="default_shardings/llama.yaml"
```

Multi-host (Llama2 70B):

```bash
export RAY_ADDRESS=http://localhost:8265

kubectl port-forward svc/example-cluster-kuberay-head-svc 8265:8265 &

ray job submit --runtime-env-json='{"working_dir": "."}' -- python run_ray_serve_interleave.py  --tpu_chips=8 --num_hosts=2 --size=70b --model_name=llama-2 --batch_size=8 --max_cache_length=2048 --tokenizer_path=/llama/tokenizer.model --checkpoint_path=/llama/ckpt --quantize_weights=True --quantize_type="int8_per_channel" --quantize_kv_cache=True --sharding_config="default_shardings/llama.yaml"
```

### Sending an inference request

Port-forward to port 8888 for gRPC:
```
kubectl port-forward svc/example-cluster-kuberay-head-svc 8888:8888 &
```

Sample python script:

```python
import requests
import os
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

prompt = "What are the top 5 languages?"

channel = grpc.insecure_channel("localhost:8888")
stub = jetstream_pb2_grpc.OrchestratorStub(channel)

request = jetstream_pb2.DecodeRequest(
    text_content=jetstream_pb2.DecodeRequest.TextContent(
        text=prompt
    ),
    priority=0,
    max_tokens=2000,
)

response = stub.Decode(request)
output = []
for resp in response:
  output.extend(resp.stream_content.samples[0].text)

text_output = "".join(output)
print(f"Prompt: {prompt}")
print(f"Response: {text_output}")
```



# Typical Errors

## Unexpected keyword argument 'device'

Fix:
* Uninstall jax and jaxlib dependencies 
* Reinstall using `source install_everything.sh

## Out of memory

Fix:
* Use smaller batch size
* Use quantization

# Links

## JetStream
* https://github.com/google/JetStream

## MaxText
* https://github.com/google/maxtext


