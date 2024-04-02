TORCHXLA_TAG=jetstream-pytorch

pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage safetensors

mkdir -p deps
pushd deps
git clone https://github.com/google/JetStream.git
git clone https://github.com/pytorch/xla.git
pushd xla
git checkout $TORCHXLA_TAG
popd
popd  # now at the folder of jetstream-pytorch

export PYTHONPATH=$PYTHONPATH:$(pwd)/deps/xla/experimental/torch_xla2:$(pwd)/deps/JetStream:$(pwd)

