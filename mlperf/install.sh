#!/usr/bin/env bash

DATA_DISK_DIR=data

mkdir -p $DATA_DISK_DIR

pip install -U "huggingface_hub[cli]"
pip install \
  transformers \
  nltk==3.8.1 \
  evaluate==0.4.0 \
  absl-py==1.4.0 \
  rouge-score==0.1.2 \
  sentencepiece==0.1.99 \
  accelerate==0.21.0

# install loadgen
pip install mlperf-loadgen


pushd $DATA_DISK_DIR

# model weights
gcloud storage cp gs://sixiang_gcp/mixtral-instruct-quantized ./ --recursive
# NOTE: uncomment one so you dont download too much weights to your box
# gcloud storage cp gs://sixiang_gcp/llama2-70b/llama2-70b/ ./ --recursive

# Get mixtral data
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_v4.pkl
mv mixtral_8x7b%2F2024.06.06_mixtral_15k_v4.pkl mixtral_15k_data.pkl
wget https://inference.mlcommons-storage.org/mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl
mv mixtral_8x7b%2F2024.06.06_mixtral_15k_calibration_v4.pkl mixtral_15k_calibration_data.pkl

# Get llama70b data
gcloud storage cp \
   gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.calibration_1000.pkl \
   processed-calibration-data.pkl
gcloud storage cp \
     gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
     processed-data.pkl
popd
