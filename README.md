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
3. Clone repo and install dependencies 
4. Download and convert weights
5. Run checkpoint converter (quantizer)
6. Local run
7. Run the server
8. Run benchmarks
9. Typical Errors

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
git checkout jetstream-v0.2.3
```

(optional) Create a virtual env using `venv` or `conda` and activate it.

## 2. Run installation script:

```bash
cd jetstream-pytorch
source install_everything.sh
```

# Download and convert weights

## LLaMA
### Get official llama weights from meta-llama

Following instructions here: 
* Llama-2: https://github.com/meta-llama/llama#download
* Llama-3: https://github.com/meta-llama/llama3/#download

After you have downloaded the weights, it will also download a `tokenizer.model` file that is 
the tokenizer that we will use.

## Gemma
### Get Gemma Checkpoint from HuggingFace

Please sign agreement on Huggingface website to access Gemma checkpoints. Download Gemma PyTorch checkpoint using huggingface-cli. Gemma Tokenizer is included in the checkpoint.

```bash
# Install huggingface-cli and login if it's not set up.
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download google/gemma-7b-pytorch --local-dir $input_ckpt_dir
```

## Mixtral
### Get Mixtral Checkpoint from HuggingFace

Please sign agreement on Huggingface website to access Mixtral checkpoints. Download Mixtral PyTorch checkpoint using huggingface-cli. Mixtral Tokenizer is included in the checkpoint.

```bash
huggingface-cli download mistralai/Mixtral-8x7B-v0.1 --local-dir $input_ckpt_dir
```

## Run weight safetensor convert

There are limited support (only Llama models as of now) for accessing checkpoints on GCS. Accessing GCS takes a long time and therefore storing checkpoints to local is recommended.

```bash
export input_ckpt_dir=Original llama weights directory
export output_ckpt_dir=The output directory
export model_name="llama-3" # or "llama-2", "gemma", "mixtral"
export quantize_weights=True # Whether to quantize weights
export quantize_type="int8_per_channel" # "quantize_weights" needs to be turned on. Availabe quantize type: {"int8", "int4"} x {"per_channel", "blockwise"}, "int8_per_channel" is the default option if not specified.
python -m convert_checkpoints --model_name=$model_name --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize_type=$quantize_type --quantize_weights=$quantize_weights
```


# Local run

Set tokenizer path
```bash
export tokenizer_path=tokenizer model file path
```

## Llama-2 7b
```bash
python run_interactive.py --size=7b --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-2 13b
```bash
python run_interactive.py --size=13b --model_name=$model_name --batch_size=64 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-3 8b
```bash
python run_interactive.py --size=8b --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-3 70b
```bash
python run_interactive.py --size=70b --model_name=$model_name --batch_size=8 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Gemma 7b
```bash
python run_interactive.py --model_name=$model_name --size=7b --batch_size=64 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/$model_name.yaml
```

## Mixtral 8x7b
```bash
python run_interactive.py --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/$model_name.yaml
```


# Run the server
Here is an example to run the server with llama2 7B config.

```bash
python run_server.py --model_name=$model_name --size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize_weights --quantize_type=$quantize_type --quantize_kv_cache=$quantize_weights --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"
```

Now you can fire gRPC to it.

Optional flags: 
* `--shard_on_batch=1` This makes the model to shard on 
  the batch dimension. I.e. this runs in data parallel mode instead of model
  parallel. This will ignore the sharding config. This is recommended for Gemma 2B
  model, because Gemma 2B is small enough to fit on a single TPU chip.

* `--sharding_config=<path>` This makes use of alternative sharding config instead of
  the ones in default_shardings directory.


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
python run_server_with_ray.py --tpu_chips=16 --num_hosts=4 --worker_chips= 4 -model_name=$model_name          --size=7b --batch_size=96 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"
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


