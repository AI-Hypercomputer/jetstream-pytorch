TORCHXLA_TAG=jetstream-pytorch

pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage safetensors colorama

mkdir -p deps
pushd deps
git clone https://github.com/google/JetStream.git
git clone https://github.com/pytorch/xla.git
pushd xla/experimental/torch_xla2
git checkout $TORCHXLA_TAG
pip install .
popd  # now at the folder deps
pushd JetStream
pip install .
popd # now at the folder deps
popd # now at the folder current file
pip install -e .