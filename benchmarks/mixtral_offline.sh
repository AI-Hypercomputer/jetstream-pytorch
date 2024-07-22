CACHE_LENGTH=3072
INPUT_SIZE=2048
OUTPUT_SIZE=1024
BATCH_SIZE=256
CHECKPOINT_PATH=mlperf/data/mixtral-instruct-quantized/

pushd ..
python -m benchmarks.run_offline \
  --lazy_cache_update=1 \
  --ring_buffer=0 \
  --use_gmm_threshold=2048 \
  --model_name=mixtral \
  --batch_size=$BATCH_SIZE \
  --max_cache_length=$CACHE_LENGTH \
  --max_decode_length=$OUTPUT_SIZE \
  --context_length=$INPUT_SIZE \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1 \
  --profiling_output=/mnt/disks/hanq/mixtral-profiles
popd
