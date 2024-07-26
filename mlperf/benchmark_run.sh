BASEDIR=.
API_URL=0.0.0.0:9000
USER_CONFIG=$BASEDIR/user.conf
DATA_DISK_DIR=$BASEDIR/data
TOTAL_SAMPLE_COUNT=24576
DATASET_PATH=/mnt/disks/hanq/jetstream-pytorch/mlperf/data/open_orca_gpt4_tokenized_llama.sampled_24576.pkl

# HF model id
#TOKENIZER_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
TOKENIZER_PATH="meta-llama/Llama-2-70b-hf"

LOADGEN_RUN_TYPE=server-performance
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}

mkdir -p ${OUTPUT_LOG_DIR} && cp ../${USER_CONFIG} ${OUTPUT_LOG_DIR}

python -m main \
	--accuracy \
	--api-url ${API_URL} \
	--scenario Server \
	--is-stream \
	--input-mode tokenized \
  --output-mode tokenized \
	--log-pred-outputs \
	--mlperf-conf $BASEDIR/mlperf.conf \
	--user-conf ${USER_CONFIG} \
	--audit-conf no-audit \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--dataset-path ${DATASET_PATH} \
	--tokenizer-path ${TOKENIZER_PATH} \
	--log-interval 1000 \
	--output-log-dir ${OUTPUT_LOG_DIR} 2>&1 | tee ${OUTPUT_LOG_DIR}/server_accuracy_log.log
