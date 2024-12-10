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

# Ubuntu:22.04
# Use Ubuntu 22.04 from Docker Hub.
# https://hub.docker.com/_/ubuntu/tags?page=1&name=22.04
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_JETSTREAM_VERSION=main

RUN apt -y update && apt install -y --no-install-recommends \
    ca-certificates \
    git \
    python3.10 \
    python3-pip

RUN python3 -m pip install --upgrade pip

RUN update-alternatives --install \
    /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN git clone https://github.com/AI-Hypercomputer/jetstream-pytorch.git && \
cd /jetstream-pytorch && \
git checkout ${PYTORCH_JETSTREAM_VERSION} && \
bash install_everything.sh

COPY jetstream_pytorch_server_entrypoint.sh /usr/bin/

RUN chmod +x /usr/bin/jetstream_pytorch_server_entrypoint.sh

ENTRYPOINT ["/usr/bin/jetstream_pytorch_server_entrypoint.sh"]