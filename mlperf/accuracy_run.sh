#!/usr/bin/env bash
me=$(basename "$0")

API_URL=0.0.0.0:9000
USER_CONFIG=mlperf/user.conf
TOTAL_SAMPLE_COUNT=24576
DATASET_PATH=$DATA_DISK_DIR/processed-data.pkl

# HF model id
TOKENIZER_PATH="meta-llama/Llama-2-70b-hf"
LOADGEN_RUN_TYPE=offline-accuracy
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}

mkdir -p ${OUTPUT_LOG_DIR} && cp ../${USER_CONFIG} ${OUTPUT_LOG_DIR}

OUTPUT_ACCURACY_JSON_PATH=${OUTPUT_LOG_DIR}/llama/mlperf_log_accuracy.json

CACHE_LENGTH=2048
INPUT_SIZE=1024
OUTPUT_SIZE=1024
CHECKPOINT_PATH=$DATA_DISK_DIR/llama2-70b/

LIBTPU_INIT_ARGS="--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true"
# makes subsequent runs faster
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache2"
export LIBTPU_INIT_ARGS

pushd ..

python -m mlperf.evaluate_accuracy \
		--checkpoint-path ${TOKENIZER_PATH} \
		--mlperf-accuracy-file ${OUTPUT_ACCURACY_JSON_PATH} \
		--dataset-file ${DATASET_PATH} 2>&1 | tee ${OUTPUT_LOG_DIR}/evaluate_offline_accuracy_log.log
popd
