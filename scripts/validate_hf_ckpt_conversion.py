import torch
from safetensors import safe_open

"""
Script to compare converted checkpoint for debugging purpose.
"""

converted_from_orig = (
    "/mnt/disks/lsiyuan/llama_weight/7B-FT-chat-converted/model.safetensors"
)

converted_from_hf = "/mnt/disks/lsiyuan/llama_weight/hf_llama_2_7b_converted_bf16/model.safetensors"

orig_state_dict = {}
with safe_open(converted_from_orig, framework="pt", device="cpu") as f:
  for key in f.keys():
    orig_state_dict[key] = f.get_tensor(key)

hf_state_dict = {}
with safe_open(converted_from_hf, framework="pt", device="cpu") as f:
  for key in f.keys():
    hf_state_dict[key] = f.get_tensor(key)

for key in orig_state_dict.keys():
  if key != "rope.freqs":
    assert key in hf_state_dict, f"{key} in orig but not in hf"
  else:
    print("rope.freqs skipped.")

for key in hf_state_dict.keys():
  assert key in orig_state_dict, f"{key} in hf but not in orig"


def _calc_cosine_dist(x, y):
  x = x.flatten().to(torch.float32)
  y = y.flatten().to(torch.float32)
  return (torch.dot(x, y) / (x.norm() * y.norm())).item()


for key in hf_state_dict.keys():
  orig_w = orig_state_dict[key]
  hf_w = hf_state_dict[key]
  print(f"weight diff {key} : {_calc_cosine_dist(orig_w, hf_w)}")
