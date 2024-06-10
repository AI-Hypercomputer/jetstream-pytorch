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

# Uninstall existing jax
pip show jax && pip uninstall -y jax
pip show jaxlib && pip uninstall -y jaxlib
pip show libtpu-nightly && pip uninstall -y libtpu-nightly
pip show tensorflow && pip uninstall -y tensorflow
pip show ray && pip uninstall -y ray
pip show flax && pip uninstall -y flax
pip show keras && pip uninstall -y keras
pip show tensorboard && pip uninstall -y tensorboard
pip show tensorflow-text && pip uninstall -y tensorflow-text
pip show torch_xla2 && pip uninstall -y torch_xla2

pip install flax==0.8.3
pip install jax[tpu]==0.4.28 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install tensorflow-text
pip install tensorflow

pip install ray[default]==2.22.0
# torch cpu
pip install torch==2.2.1+cpu --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow flatbuffers absl-py sentencepiece seqio google-cloud-storage 
pip install safetensors colorama coverage humanize

git submodule update --init --recursive
pip show google-jetstream && pip uninstall -y google-jetstream
pip show torch_xla2 && pip uninstall -y torch_xla2
pip install -e .
