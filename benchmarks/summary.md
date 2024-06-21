# Benchmark results of various models


## Llama 3 - 8B

Date | Device  | dtype | batch size | cache length |max input length |max output length| throughput (token/s) 
----| ------- | ------ |---------- | -------------|-----------------|------------------|----------------------
2024-04-24 | TPU v5e-8 | bfloat16 | 128 | 2048 | 1024 | 1024 | 8249 
2024-04-24 | TPU v5e-8 | int8 | 256 | 2048 | 1024 | 1024 | 10873


## Gemma - 7B

Date | Device  | dtype | batch size | cache length |max input length |max output length| throughput (token/s) 
----| ------- | ------ |---------- | -------------|-----------------|------------------|----------------------
2024-05-10 | TPU v5e-8 | bfloat16 | 96 | 2048 | 1024 | 1024 | 3236
2024-05-10 | TPU v5e-8 | int8 | 128 | 2048 | 1024 | 1024 | 4695

## Gemma - 2B

Date | Device  | dtype | batch size | cache length |max input length |max output length| throughput (token/s) 
----| ------- | ------ |---------- | -------------|-----------------|------------------|----------------------
2024-05-14 | TPU v5e-8 | bfloat16 | 512 | 2048 | 1024 | 1024 | 8700
2024-05-14 | TPU v5e-8 | int8 | 1024 | 2048 | 1024 | 1024 | 8746
2024-06-13 | TPU v5e-1 | bfloat16 | 1024 | 2048 | 1024 | 1024 | 4249


** NOTE: ** Gemma 2B uses `--shard_on_batch` flag so it's data parallel instead
of model parallel.


## Llama 2 - 7B

Date | Device  | dtype | batch size | cache length |max input length |max output length| throughput (token/s) 
----| ------- | ------ |---------- | -------------|-----------------|------------------|----------------------
2024-03-28 | TPU v5e-8 | bfloat16 | 96 | 2048 | 1024 | 1024 | 3663
2024-03-28 | TPU v5e-8 | int8 | 96 | 2048 | 1024 | 1024 | 4783 

## Llama 2 - 13B

Date | Device  | dtype | batch size | cache length |max input length |max output length| throughput (token/s) 
----| ------- | ------ |---------- | -------------|-----------------|------------------|----------------------
2024-03-28 | TPU v5e-8 | bfloat16 | 48 | 2048 | 1024 | 1024 | 2056
2024-03-28 | TPU v5e-8 | int8 | 96 | 2048 | 1024 | 1024 | 3458 
2024-03-28 | TPU v5e-8 | bfloat16 | 80 | 1280 | 1024 | 1024 | 2911
2024-03-28 | TPU v5e-8 | int8 | 96 | 1280 | 1024 | 1024 | 3938

**NOTE:** When cache length is less than the sum of max input length + max output length
  we employ *Rolling window attention*. 


# Instructions to reproduce:

Please refer [README.md](README.md) for instructions in how to get the model weights.

**NOTE** Different weights can produce different benchmark results (due to generating)
different sentence length. For llama, we used the `-chat` versions of the weight.
For Gemma we used the `-it` (instruction finetuned) version of the weights.

## Run the server
NOTE: the `--platform=tpu=8` need to specify number of tpu devices (which is 4 for v4-8 and 8 for v5light-8`)

```bash
python run_server.py --param_size=7b --batch_size= 128 --max_cache_length=2048 --quantize_weights=$quantize --quantize_kv_cache=$quantize --checkpoint_path=$output_ckpt_dir   --tokenizer_path=$tokenizer_path --platform=tpu=8 --model=$model_name
```
Now you can fire gRPC to it

# Run benchmark
go to the deps/JetStream folder (downloaded during `install_everything.sh`)

```bash
cd deps/JetStream
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
export dataset_path=ShareGPT_V3_unfiltered_cleaned_split.json
python benchmarks/benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 2000  --dataset-path  $dataset_path --dataset sharegpt --save-request-outputs --warmup-first=True
```
Please look at `deps/JetStream/benchmarks/README.md` for more information.
