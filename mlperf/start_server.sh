#!/usr/bin/env bash

CACHE_LENGTH=2048
INPUT_SIZE=1024
OUTPUT_SIZE=1024
CHECKPOINT_PATH=~/data/mlperf/llama2-70b
pushd ..
python run_server.py \
  --model_name=llama-2 \
  --size=70b \
  --ring_buffer=0 \
  --batch_size=96 \
  --max_cache_length=$CACHE_LENGTH \
  --max_decode_length=$OUTPUT_SIZE \
  --context_length=$INPUT_SIZE \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1
popd
