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
JETSTREAM_TAG=v0.2.0

# Uninstall existing jax
pip3 show jax && pip3 uninstall -y jax
pip3 show jaxlib && pip3 uninstall -y jaxlib
pip3 show libtpu-nightly && pip3 uninstall -y libtpu-nightly

pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage 
pip3 install safetensors colorama coverage ray[default] humanize

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