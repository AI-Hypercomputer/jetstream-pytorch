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

TORCHXLA_TAG=6dccf0a02d7828516bdb589f2ae0dc79b64488fa  # updated May 10, 2024
JETSTREAM_TAG=v0.2.1

# Uninstall existing jax
pip show jax && pip uninstall -y jax
pip show jaxlib && pip uninstall -y jaxlib
pip show libtpu-nightly && pip uninstall -y libtpu-nightly

pip install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage 
pip install safetensors colorama coverage ray[default] humanize

mkdir -p deps
pushd deps
git clone https://github.com/google/JetStream.git
git clone https://github.com/pytorch/xla.git
pushd xla/experimental/torch_xla2
git checkout $TORCHXLA_TAG
pip install .
popd  # now at the folder deps
pushd JetStream
git checkout $JETSTREAM_TAG
pip install .
popd # now at the folder deps
popd # now at the folder current file
pip install -e .