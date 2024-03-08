git clone git@github.com:google/JetStream.git
git clone git@github.com:pytorch/xla.git


pip3 install pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# torch cpu
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install tensorflow flatbuffers absl-py flax sentencepiece

export PYTHONPATH=$PYTHONPATH:$(pwd)/xla/experimental/torch_xla2:$(pwd)/JetStream

