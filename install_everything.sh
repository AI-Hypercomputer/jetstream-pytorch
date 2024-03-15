TORCHXLA_TAG=jetstream-pytorch

pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece seqio google-cloud-storage safetensors

git clone https://github.com/google/JetStream.git
git clone https://github.com/pytorch/xla.git
cd xla && git checkout $TORCHXLA_TAG

export PYTHONPATH=$PYTHONPATH:$(pwd)/xla/experimental/torch_xla2:$(pwd)/JetStream

