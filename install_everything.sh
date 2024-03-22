# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
TORCHXLA_TAG=jetstream-pytorch

pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage safetensors

git clone https://github.com/google/JetStream.git
git clone https://github.com/pytorch/xla.git
cd xla && git checkout $TORCHXLA_TAG

export PYTHONPATH=$PYTHONPATH:$(pwd)/xla/experimental/torch_xla2:$(pwd)/JetStream

