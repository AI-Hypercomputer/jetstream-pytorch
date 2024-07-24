CACHE_LENGTH=$1
BATCH_SIZE=$2
INPUT_SIZE=1024
OUTPUT_SIZE=1024
CHECKPOINT_PATH=mlperf/data/llama2-70b/
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export XLA_FLAGS="--xla_disable_hlo_passes=rematerialization"

pushd ..
python -m benchmarks.run_offline \
  --lazy_cache_update=1 \
  --ring_buffer=0 \
  --quantize_activation=1 \
  --model_name=llama-2 \
  --size=70b \
  --batch_size=$BATCH_SIZE \
  --max_cache_length=$CACHE_LENGTH \
  --max_decode_length=$OUTPUT_SIZE \
  --context_length=$INPUT_SIZE \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1 \
  --profiling_output=/mnt/disks/hanq/llama-profiles
popd
echo "batch was $2 cache was $1"
