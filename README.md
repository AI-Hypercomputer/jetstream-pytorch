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
Follow step 1-9 in the following guide
* https://cloud.google.com/tpu/docs/v5e-inference#prepare-a-project

# Clone repo and install dependencies 

## Get the jetstream-pytorch code
```bash
git clone https://github.com/google/jetstream-pytorch.git
```

(optional) Create a virtual env using `venv` or `conda` and activate it.

## 2. Run installation script:

```bash
cd jetstream-pytorch
source install_everything.sh
```
NOTE: the above script will export PYTHONPATH, so sourcing will make it to take effect in the current shell

# Download and convert weights

## Get official llama weights from meta-llama

Following instructions here: https://github.com/meta-llama/llama#download

After you have downloaded the weights, it will also download a `tokenizer.model` file that is 
the tokenizer that we will use.

## Run weight safetensor convert

```bash
export input_ckpt_dir=Original llama weights directory
export output_ckpt_dir=The output directory
export quantize=True #whether to quantize
python -m convert_checkpoints --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize=$quantize
```


# Local run

Set tokenizer path
```bash
export tokenizer_path=tokenizer model file path from meta-llama
```

## Llama 7b
```bash
python run_interactive.py --size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path
```

## Llama 13b
```bash
python run_interactive.py --size=13b --batch_size=64 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir --tokenizer_path=$tokenizer_path
```


# Run the server
NOTE: the `--platform=tpu=8` need to specify number of tpu devices (which is 4 for v4-8 and 8 for v5light-8`)

```bash
python run_server.py --param_size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --platform=tpu=8
```
Now you can fire gRPC to it

# Run benchmark
go to the deps/JetStream folder (downloaded during `install_everything.sh`)

```bash
cd deps/JetStream
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 2000  --dataset-path  $dataset_path --dataset sharegpt --save-request-outputs
```
Please look at `deps/JetStream/benchmarks/README.md` for more information.


# Typical Errors

##  Unexpected keyword argument 'device'

Fix:
* Uninstall jax and jaxlib dependencies 
* Reinstall using `source install_everything.sh

##  Out of memory

Fix:
* Use smaller batch size
* Use quantization


