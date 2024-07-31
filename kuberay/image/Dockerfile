FROM rayproject/ray:2.32.0-py310

RUN pip install flax==0.8.3
RUN pip install jax[tpu]==0.4.30 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install tensorflow-text
RUN pip install tensorflow

RUN pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install tensorflow flatbuffers absl-py sentencepiece seqio google-cloud-storage
RUN pip install safetensors colorama coverage humanize

RUN git clone https://github.com/google/jetstream-pytorch
WORKDIR jetstream-pytorch

RUN git submodule update --init --recursive
RUN pip install -e .
