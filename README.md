# Jetstream-PyTorch
JetStream Engine implementation in PyTorch


# Install

### 1. Get the jetstream-pytorch code
```bash
git clone https://github.com/pytorch-tpu/jetstream-pytorch.git
```

1.1 (optional) Create a virtual env using `venv` or `conda` and activate it.

### 2. Run installation script:

```bash
sh install_everything.sh
```


# Get weights

### First get official llama weights from meta-llama

Following instructions here: https://github.com/meta-llama/llama#download

After you have downloaded the weights, it will also download a `tokenizer.model` file that is 
the tokenizer that we will use.

### Run weight merger to convert (and )
```bash
export input_ckpt_dir=Original llama weights directory
export output_ckpt_dir=The output directory
export quantize=True #whether to quantize
python -m convert_checkpoints --input_checkpoint_dir=$input_ckpt_dir --output_checkpoint_dir=$output_ckpt_dir --quantize=$quantize
```


# Local run

## Llama 7b
```
python benchmarks/run_offline.py --size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir/model.safetensors --tokenizer_path=tokenizer.model
```

## Llama 13b
```
python benchmarks/run_offline.py --size=13b --batch_size=96 --max_cache_length=1280 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir/model.safetensors --tokenizer_path=tokenizer.model
```
NOTE: for 13b model we recommend to use `--max_cache_length=1280`, i.e. this implements sliding window attention.


# Run the server
NOTE: the `--platform=tpu=8` need to specify number of tpu devices (which is 4 for v4-8 and 8 for v5light-8`)

```bash
python run_server.py --param_size=7b --batch_size=128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir/model.safetensors   --tokenizer_path=tokenizer.model --platform=tpu=8
```
Now you can fire gRPC to it

# Run benchmark
go to the deps/JetStream folder (downloaded during `install_everything.sh`)
```bash
cd deps/JetStream
python benchmark_serving.py --tokenizer /home/hanq/jetstream-pytorch/tokenizer.model --num-prompts 2000  --dataset ~/data/ShareGPT_V3_unfiltered_cleaned_split.json --warmup-first=1 --save-request-outputs
```
The ShareGPT dataset can be downloaded at 

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
Please look at `deps/JetStream/benchmarks/README.md` for more information.
