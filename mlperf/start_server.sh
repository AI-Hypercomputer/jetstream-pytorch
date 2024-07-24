#!/usr/bin/env bash

CACHE_LENGTH=3072
INPUT_SIZE=512
OUTPUT_SIZE=512
CHECKPOINT_PATH=mlperf/data/mixtral-instruct-quantized/

pushd ..
python run_server.py \
  --model_name=mixtral \
  --lazy_cache_update=1 \
  --ring_buffer=0 \
  --batch_size=128 \
  --max_cache_length=$CACHE_LENGTH \
  --max_decode_length=$OUTPUT_SIZE \
  --context_length=$INPUT_SIZE \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1
popd
