#!/usr/bin/env bash
me=$(basename "$0")

BASEDIR=mlperf
USER_CONFIG=$BASEDIR/user.conf
DATA_DISK_DIR=$BASEDIR/data
TOTAL_SAMPLE_COUNT=900

# HF model id
TOKENIZER_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
LOADGEN_RUN_TYPE=offline-performance
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}

mkdir -p ${OUTPUT_LOG_DIR} && cp ../${USER_CONFIG} ${OUTPUT_LOG_DIR}

OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/mlperf_log_accuracy.json

CHECKPOINT_PATH=mlperf/data/mixtral-instruct-quantized/

LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS


DATASET_PATH=$BASEDIR/data/mixtral_1k_data.pkl

pushd ..
python -m mlperf.offline_mode \
  --mlperf_test_mode=$1 \
  --model_name=mixtral \
  --checkpoint_path=$CHECKPOINT_PATH/model.safetensors \
  --tokenizer_path=$CHECKPOINT_PATH/tokenizer.model \
  --quantize_weights=1 \
  --quantize_type=int8_per_channel \
  --quantize_kv_cache=1 \
	--input_mode tokenized \
  --output_mode tokenized \
	--mlperf_conf $BASEDIR/mlperf.conf \
	--user_conf ${USER_CONFIG} \
	--audit_conf no_audit \
	--total_sample_count ${TOTAL_SAMPLE_COUNT} \
	--dataset_path ${DATASET_PATH} \
	--output_log_dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/server_accuracy_log.log

if [ "$1" = "accuracy" ]; then
python -m mlperf.evaluate_accuracy \
		--checkpoint-path ${TOKENIZER_PATH} \
		--mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
		--dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
fi
popd
