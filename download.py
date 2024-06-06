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
import os
from typing import Optional

from huggingface_hub import snapshot_download
from requests.exceptions import HTTPError

from absl import app, flags

_HF_REPO_ID = flags.DEFINE_string("hf_repo_id", "", "The huggingface repo id.")
_HF_TOKEN =  flags.DEFINE_string("hf_token", "", "The personal token to huggingface")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", ".", "The target dir to download the repo.") 

def hf_download() -> None:
    from huggingface_hub import snapshot_download
    output_dir = f"{_OUTPUT_DIR.value}/checkpoints/{_HF_REPO_ID.value}"
    os.makedirs(output_dir, exist_ok=True)
    try:
        snapshot_download(_HF_REPO_ID.value, local_dir=output_dir, local_dir_use_symlinks=False, token=_HF_TOKEN.value, ignore_patterns="*.safetensors")
    except HTTPError as e:
        if e.response.status_code == 401:
            print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            return
        else:
            raise e
            return
    print(f"Repo downloaded to {output_dir}")

def main(argv) -> None:
    hf_download()

if __name__ == "__main__":
  app.run(main)
