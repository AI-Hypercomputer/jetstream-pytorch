#!/usr/bin/env bash
me=$(basename "$0")

CACHE_LENGTH=2048
INPUT_SIZE=1024
OUTPUT_SIZE=1024
CHECKPOINT_PATH=mlperf/data/mixtral-instruct-quantized/

pushd ..
python run_interactive.py \
  --model_name=mixtral \
  --batch_size=8 \
  --max_cache_length=$CACHE_LENGTH \
  --max_decode_length=$OUTPUT_SIZE \
  --context_length=$INPUT_SIZE \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1
popd