# Jetstream-PyTorch
JetStream Engine implementation in PyTorch

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
git checkout jetstream-v0.2.2
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

```bash
export input_ckpt_dir=Original llama weights directory
export output_ckpt_dir=The output directory
export model_name="llama-3" # or "llama-2", "gemma", "mixtral"
export quantize_weights=True # Whether to quantize weights
export quantize_type="int8_per_channel" # "quantize_weights" needs to be turned on. Availabe quantize type: {"int8", "int4"} x {"per_channel", "blockwise"}, "int8_per_channel" is the default option if not specified.
python -m convert_checkpoints --model_name=$model_name --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize_type=$quantize_type
```


# Local run

Set tokenizer path
```bash
export tokenizer_path=tokenizer model file path
```

## Llama-2 7b
```bash
python run_interactive.py --size=7b --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-2 13b
```bash
python run_interactive.py --size=13b --model_name=$model_name --batch_size=64 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-3 8b
```bash
python run_interactive.py --size=8b --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Llama-3 70b
```bash
python run_interactive.py --size=70b --model_name=$model_name --batch_size=8 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/llama.yaml
```

## Gemma 7b
```bash
python run_interactive.py --model_name=$model_name --size=7b --batch_size=64 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/$model_name.yaml
```

## Mixtral 8x7b
```bash
python run_interactive.py --model_name=$model_name --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/$model_name.yaml
```


# Run the server
Here is an example to run the server with llama2 7B config.

```bash
python run_server.py --model_name=$model_name --size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"
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
python run_server_with_ray.py --tpu_chips=16 -model_name=$model_name --size=7b --batch_size=96 --max_cache_length=2048 --quantize_weights=$quantize --quantize_type=$quantize_type --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --sharding_config="default_shardings/llama.yaml"
```

# Run benchmark
Start the server and then go to the deps/JetStream folder (downloaded during `install_everything.sh`)

```bash
cd deps/JetStream
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 2000  --dataset-path  $dataset_path --dataset sharegpt --save-request-outputs --warmup-first=True
```
Please look at `deps/JetStream/benchmarks/README.md` for more information.


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


