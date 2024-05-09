Add a new Model
===============

This doc is a detailed guide on how to add a new model to jetstream-pytorch.
The complexity of adding a new model depends highly on the model architecture itself,
and right now is a manual process.

NOTE: Only LLMs that employ autoregressive decoding that utilices a KV cache are suitable
for serving with Jetstream. Other models such as Stable Diffusion are NOT suitable
with the optimization techniques used in Jetstream.

The core part of adding a model is to let Jetstream serving engine manage
the KV cache. This management is abstracted by the [`class CacheInterface`](jetstream_pt/cache_manager.py). This interface has a single `update` method that will abstract
the act of inserting and then reading the cache.

We will walk through this process using [Gemma model](https://github.com/google/gemma_pytorch) as an example.

# Step 0: Get the model code

Jetstream pytorch stores its models in the jetstream_pt/third_party directory.

The usual convention is:

1. Make a verbatim copy of the model code and supporting files 
   (such as args class, tokenizers etc) in a separate directory. In our case
   it would be [jetstream_pt/third_party/gemma](jetstream_pt/third_party/gemma)

2. Make a copy of the `model.py` to `model_original.py`; because we will be modifying
   it to follow the conventions of Jetstream; and keeping the original can help with
   debugging accuracies (and unit tests).

*Optional:* Clean up model implementation: The easiest model to port are those of 
    "reference implementations". Models already with optimizations and/or custom
    cuda kernels would need to have those changes removed.

In our case, we choose to use the reference Gemma model from google's github instead
of the HuggingFace version, because HuggingFace version have also training code that 
would need to be removed.

# Step 1: Modify the model to fit the calling conventions expected by Jetstream.

The model that Jetstream expects and calls follows this calling convention:

```python
class Model(torch.nn.Module):

  def forward(
      self,
      tokens: torch.Tensor,
      input_pos: torch.Tensor,
      caches: List[CacheInterface],
      mask: torch.Tensor,
  ) -> torch.Tensor:

```

The arguments are:

* `tokens`: A int tensor with shape (batch_size, sequence_length). This is the token ids 
     before embedding

* `input_pos`: The position of the tokens in the overall sentence. This is an int 
     tensor of shape (batch_size, sequence_length). Note: due to continues batching,
     not all batch have the same sequence length.

* `caches`: A list of objects implementing the `CacheInterface`. CacheInterface has a 
     single `update` method.

* `mask`: Mask used in causal attention.

The return value should be a tensor of shape (batch_size, sequence_length, vocab_size)
of **logits** (not probability) for the next token.

### Gemma example:

Now looking back to our Gemma model reference. There are 2 classes in the original
model that is suitable to be our model [GemmaModel](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L353) and [GemmaForCausalLM](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L386). Looking at their forward method signature:

```python

class GemmaModel(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:

class GemmaForCausalLM(nn.Module):
    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
```

We can see that `GemmaModel` is probably closest to port. So we choose that one.
However there are few issues:

1. GemmaModel takes `hidden_states` instead of tokens
2. GemmaModel returns `hidden_states` after the layers and not logits.

Let's fix those first.

Looking at where `GemmaModel` is called in `model.py`, we found that:

```
        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
```

So the input_tokens are embedded with `self.embedder` and processed before calling
`GemmaModel`. So let's move these bit to inside of GemmaModel.

Now, look where the output of `GemmaModel` is consumed, we see it is feed to `self.sampler`.

`self.sampler` is of class `Sampler` and it's forward has:

```python
    hidden_states = hidden_states.index_select(
        1, output_positions).squeeze(dim=1)
    logits = torch.matmul(hidden_states, embedding.t())
    if embedding_bias is not None:
        logits += embedding_bias

    if temperatures is None:
        return torch.argmax(logits, dim=-1).squeeze(dim=-1)
    ...
```

We see it performed some math with hidden states to produce logits, which is what 
GemmaModel should return. Now, let move these bits into `GemmaModel` as well.

Lastly, GemmaModel takes a list of tuple of torch.Tensor as input for caches, 
we need to replace it with cache object.

This cache is plumbed through all the way to `GemmaAttention`, and the [following lines](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L264C1-L268C53):

```python
        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)
```

is precisely the cache update.
So we need to replace those lines with

```
xk = xk.transpose(1, 2)
xv = xv.transpose(1, 2)
k_cache, v_cache = cache.update(xk, xv)
```

The transpose is needed because the cache interface's `update` method expects
shape of (batch, num_heads, sequence length, head dim) instead of 
         (batch, sequence length, num_heads, head dim) that GemmaAttention produces.


In our case, because the Attention math is the standard one, we can just call out
to `AttentionKernel` defined in [layers.py](jetstream_pt/layers.py). `AttentionKernel`
also handles reading and writing of `cache` automatically.

At this point, the model should be runnable. However to run it on a realistic TPU, 
we need to add model parallelism.

# Step 2: Add model parallelism

Model parallelism is often neccesary to run on TPUs. The typical setup for running
inference work loads is by using TPU `v5light-8` which has 8 TPU chips with 16GB of
high bandwidth memory (HBM) each. The typical `7B` model won't fit on single chip.

So we need to add model parallelism so the model weights are sharded among the 8 devices.
This is necesary for larger models, such as 70Bs even on high memory chips (v5p).
So it's a good practice to do it right away.

Jetstream uses GSMPD to for tensor parallelism, the only information we need to 
give it is, for every tensor weights, what axis we will shard. We do so by writing
a sharding config file.

## Generate an sharding config:

The keys of the sharding file is the name of the weights, (with numeric layers replaced with *),
and value the axis to shard.
for Gemma, we can generate such file by printing out the keys in it's `state_dict`. 
See [create_empty_sharding_map.py](scripts/create_empty_sharding_map.py) for example.

Below:

```yaml
freqs_cis : -1 #  torch.complex64 (16384, 128)
layers.*.self_attn.qkv_proj.weight: 0
layers.*.self_attn.o_proj.weight: 1
layers.*.self_attn.wo.weight : 1 # 1, -1] #  torch.float32 (2048, 2048)
layers.*.self_attn.wq.weight : 0 # -1, 1] #  torch.float32 (2048, 2048)
layers.*.self_attn.wk.weight : 0 # -1, 1] #  torch.float32 (256, 2048)
layers.*.self_attn.wv.weight : 0 # -1, 1] #  torch.float32 (256, 2048)
layers.*.mlp.gate_proj.weight : 0 # -1, 1] #  torch.float32 (16384, 2048)
layers.*.mlp.gate_proj.bias : 0 # -1] #  torch.float32 (16384,)
layers.*.mlp.up_proj.weight : 0 # -1, 1] #  torch.float32 (16384, 2048)
layers.*.mlp.up_proj.bias : 0 # -1] #  torch.float32 (16384,)
layers.*.mlp.down_proj.weight : 1 # 1, -1] #  torch.float32 (2048, 16384)
layers.*.mlp.down_proj.bias : -1 #  torch.float32 (2048,)
layers.*.input_layernorm.weight : -1 #  torch.float32 (2048,)
layers.*.post_attention_layernorm.weight : -1 #  torch.float32 (2048,)
norm.weight : -1 #  torch.float32 (2048,)
embedder.weight : 1 # # 1, -1] #  torch.float32 (256000, 2048)
```

the weights `layers.*.self_attn.qkv_proj.weight` where * goes for 1..28, are sharded
on the second dimension (0 based indexing) etc. and -1 means "replicated".

Theoretically, any valid sharding would work. To find a sharding that performs well one
can usually get some hints from the original model implementation.

For example, in case of Gemma, the authors also provided an TPU version: https://github.com/google/gemma_pytorch/blob/main/gemma/model_xla.py

in that file, those with `ColumnParallelLinear` should be sharded on the dimension 0, 
and with `RowParallelLinear` should be shard on dimension 1; the others should be
replicated.

# Step 3: Activation Sharding and quantization

Sometimes we would like to specify shardings for the activation because GSPMD cannot
fully infer all the shardings.

The typical example of such case happens after a reshape. For example: if I have a matrix
of shape [A, B * C]; and the second dim is sharded; reshaping it to shape [A, B, C],
the compiler would know that one of the dim B or C is sharded, but cannot know which one.
In this case, it is helpful to specify with a sharding constraint.

This is done by calling `env.apply_sharding(tensor, axis=1)` on the tensor.

The `env` object is an instance of `Environment` class; that will be passed in the 
model constructor. It also contains some common configurations (such as whether user wants quantization), that is useful for the models.

For such, we the store that variable in `self.env` and use it when needed.

For quantization, it suffices to swap `nn.Linear` layers with `Int8QuantizedLinear` defined in
`layers.py`

# Step 4: Wiring everything up.

The last step is to modify [engine.py](https://github.com/google/jetstream-pytorch/blob/main/jetstream_pt/engine.py#L738)
and add an if branch in this function.

This function should receive information about model name and size; and 
here it should instantate the model object itself. It also need to tell the environment
information about the cache to allocate: notably how many layers and the shape of
cache. The shape is expected to be (batch size, num_kv_heads, sequence_length, head_dim).

## Test it out

After these steps you should be able to run your model using

```bash
python run_interactive.py --size=7b --batch_size=128 --max_cache_length=2048 --tokenizer_path=$tokenizer_path --sharding_config=default_shardings/$model_name.yaml --model=gemma
```

If you run it without checkpoint_path it will use random weights, so you can 
verify that the code actually run.

# Step 5: Weight convertion

Because we modified the model, and the names of variables on the model might have
changed. If so, we need to also modify `convert_weights.py` script to map
the original weights to modified names. 

For example: I split qkv projection to 3 separate projection, this helps with 
performance in a sharded environment. So I need to make `convert_weights` script
able to split the weights as well.

