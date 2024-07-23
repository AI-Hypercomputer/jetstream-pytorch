BASEDIR=mlperf
API_URL=0.0.0.0:9000
USER_CONFIG=$BASEDIR/user.conf
DATA_DISK_DIR=$BASEDIR/data
TOTAL_SAMPLE_COUNT=15000
DATASET_PATH=$BASEDIR/data/mixtral_15k_data.pkl

# HF model id
TOKENIZER_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"

LOADGEN_RUN_TYPE=offline-performance
OUTPUT_LOG_DIR=${DATA_DISK_DIR}/logs/${OUTPUT_LOG_ID}
OUTPUT_LOG_ID=${MODEL_NAME}-${DATASET_TYPE}-${LOADGEN_RUN_TYPE}-${LOADGEN_RUN_TIMESTAMP}

mkdir -p ${OUTPUT_LOG_DIR} && cp ../${USER_CONFIG} ${OUTPUT_LOG_DIR}

pushd ..
python -m mlperf.main \
	--api-url ${API_URL} \
	--scenario Offline \
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
popd
