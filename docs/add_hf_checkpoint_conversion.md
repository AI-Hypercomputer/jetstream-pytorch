# Guide on adding HuggingFace checkpoint conversion support

## Prerequisites:
The model implementation has been added in JetStream-pt
The checkpoint conversion from a certain format is already supported. (Or no conversion is needed for the checkpoint)

Please check this [guide](https://github.com/google/jetstream-pytorch/blob/main/docs/add_a_new_model.md) for adding a new model.

## Use case:
The user has the checkpoint for the same model architecture in another format (e.g. HF format for LLaMA model). And want to have JetStream-pt support this checkpoint format.

## Guide

Converting a public checkpoint to JetStream-pt format is mostly about finding the weight key mapping between the public checkpoint and JetStream model implementation. Besides the name mapping, the layout of the weights might be different among different checkpoint formats (e.g. Weight interleaved differently due to difference in Rotary Embedding implementation). These differences are model and checkpoint format specific.

**Note** The model code and checkpoint format can be different from model to model, the following guide demonstrate a general guide, specific models may require additional effort for the checkpoint conversion support.

The checkpoint conversion logic in the checkpoint conversion script.

### Step 1 Find the HuggingFace checkpoint you want to convert
In this example, let’s use meta-llama/llama-2 7B as an example

You can download the checkpoints to a local folder using 
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir Llama-2-7b-hf


**Note** You may need to go to Huggingface website to sign an agreement to get the permission to download the model

### Step 2 Inspect the weight names in the checkpoint:

Usually there is a model.safetensors.index.json file in the checkpoint. [example](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/model.safetensors.index.json)

Alternatively, you can load the weights locally and inspect the model key names(Usually it’s in safetensor format, and it’s sharded)

Example script:
```Python
import glob
import os
import torch
from safetensors import safe_open

checkpoint_folder = "/mnt/disks/lsiyuan/llama_weight/Meta-Llama-3-8B-Instruct"

safetensor_files = glob.glob(os.path.join(checkpoint_folder, "*.safetensors"))

for st_f in safetensor_files:
  with safe_open(st_f, framework="pt", device="cpu") as f:
    for key in f.keys():
      weight_tensor = f.get_tensor(key)
      print(f"Weight name {key}, Shape: {weight_tensor.shape}, dtype: {weight_tensor.dtype}")
```

Got the following output:

```
lm_head.weight torch.Size([32000, 4096]) x torch.float16
model.norm.weight torch.Size([4096]) x torch.float16
model.embed_tokens.weight torch.Size([32000, 4096]) x torch.float16
model.layers.0.input_layernorm.weight torch.Size([4096]) x torch.float16
model.layers.0.mlp.down_proj.weight torch.Size([4096, 11008]) x torch.float16
model.layers.0.mlp.gate_proj.weight torch.Size([11008, 4096]) x torch.float16
model.layers.0.mlp.up_proj.weight torch.Size([11008, 4096]) x torch.float16
model.layers.0.post_attention_layernorm.weight torch.Size([4096]) x torch.float16
model.layers.0.self_attn.k_proj.weight torch.Size([4096, 4096]) x torch.float16
model.layers.0.self_attn.o_proj.weight torch.Size([4096, 4096]) x torch.float16
model.layers.0.self_attn.q_proj.weight torch.Size([4096, 4096]) x torch.float16
model.layers.0.self_attn.rotary_emb.inv_freq torch.Size([64]) x torch.float32
model.layers.0.self_attn.v_proj.weight torch.Size([4096, 4096]) x torch.float16
… # Duplicated name for model.layers.x
```

If it’s hard to tell which layer the weight is for, the HF model class can be checked in the checkpoint config file [example](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json#L4). Then we can find the model code in the transformer repo by searching the model class name [model code](https://github.com/huggingface/transformers/blob/bdf36dcd48106a4a0278ed7f3cc26cd65ab7b066/src/transformers/models/llama/modeling_llama.py#L1084)


### Step 3 Inspect the weight names in JetStream-pt model implementation:

Run the model in JetStream using benchmarks/run_offline.py. The weight names, shape and dtype will be printed in the log (Omitting Layer N which are duplicated names)

Example:

```
Name: freqs_cis, shape: (2048, 64) x complex64
Name: tok_embeddings.weight, shape: (32000, 4096) x bfloat16
Name: layers.0.attention.wo.weight, shape: (4096, 4096) x bfloat16
Name: layers.0.attention.wq.weight, shape: (4096, 4096) x bfloat16
Name: layers.0.attention.wk.weight, shape: (4096, 4096) x bfloat16
Name: layers.0.attention.wv.weight, shape: (4096, 4096) x bfloat16
Name: layers.0.feed_forward.w1.weight, shape: (11008, 4096) x bfloat16
Name: layers.0.feed_forward.w2.weight, shape: (4096, 11008) x bfloat16
Name: layers.0.feed_forward.w3.weight, shape: (11008, 4096) x bfloat16
Name: layers.0.attention_norm.weight, shape: (4096,) x bfloat16
Name: layers.0.ffn_norm.weight, shape: (4096,) x bfloat16
Name: norm.weight, shape: (4096,) x bfloat16
Name: output.weight, shape: (32000, 4096) x bfloat16
```

If it’s hard to tell which layer the weight is for, you can find out the meaning of the weight, please check the model implementation under jetstream_pt/third_party.

### Step 4 By comparing the weight names, or diving into the model code, we can find out the mapping:
	
	In this example:

HF lm_head.weight -> JetStream-pt output.weight
HF model.norm.weight -> JetStream-pt norm.weight
HF model.embed_tokens.weight -> JetStream-pt tok_embeddings.weight
HF model.layers.X.input_layernorm.weight -> layers.X.attention_norm.weight
HF model.layers.0.post_attention_layernorm.weight -> layers.0.ffn_norm.weight
HF model.layers.X.self_attn.{q/k/v/o}_proj.weight -> layers.X.attention.w{q/k/v/o}.weight
HF model.layers.X.mlp.gate_proj.weight -> layers.X.feed_forward.w1.weight
HF model.layers.X.mlp.down_proj.weight -> layers.X.feed_forward.w2.weight
HF model.layers.X.mlp.up_proj.weight -> layers.X.feed_forward.w3.weight
freqs_cis is a special case, in JetStream PyTorch, the weight is pre-computed during weight loading, so no need to map the Huggingface freq weight over.

### Step 5 Validate the converted checkpoint:

If there is a checkpoint in already supported format, convert the checkpoint in supported format first, as the golden data to compare with the converted checkpoint from the new format.

Write a small script, or reuse the [script](https://github.com/google/jetstream-pytorch/blob/main/scripts/validate_hf_ckpt_conversion.py) to compare the 2 converted checkpoints.

Fix the difference between 2 converted checkpoints if there is any. (This will be model and checkpoint format specific)

### Step 6 End-to-end validation: From checkpoint conversion to serving

Example

```
export input_ckpt_dir=/mnt/disks/lsiyuan/llama_weight/7B-FT-chat
export output_ckpt_dir=/mnt/disks/lsiyuan/llama_weight/hf_llama_2_7b_converted_bf16_2
export model_name="llama"
export from_hf=True
python -m convert_checkpoints --model_name=$model_name \
    --input_checkpoint_dir=$input_ckpt_dir \
    --output_checkpoint_dir=$output_ckpt_dir \
    --quantize_weights=$quantize_weights \
    --quantize_type=$quantize_type \
    --from_hf=True
```