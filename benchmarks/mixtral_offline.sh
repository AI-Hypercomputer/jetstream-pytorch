CACHE_LENGTH=$1
INPUT_SIZE=512
OUTPUT_SIZE=1024
BATCH_SIZE=$2
CHECKPOINT_PATH=mlperf/data/mixtral-instruct-quantized/

pushd ..
python -m benchmarks.run_offline \
  --internal_mixtral_act_quant_w1=0 \
  --internal_mixtral_act_quant_w2=0 \
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
  --profiling_output=/tmp/mixtral-profiles
popd
