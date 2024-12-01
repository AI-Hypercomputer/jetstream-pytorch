
import time

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec, Mesh


from jetstream.engine import sampling_utils
from jetstream_pt.engine import PyTorchEngine

def sample_topk_logits(logits, topk, temperature, rng):
      sorted_indices = jnp.argsort(logits, descending=True)  # Sort in descending order
      topk_mask = jnp.arange(sorted_indices.shape[-1]) < topk
      topk_idxs = jnp.where(topk_mask, sorted_indices, -1)
      topk_logits = jnp.where(topk_idxs == -1, -jnp.inf, logits)

      # self.rng, sub_rng = jax.random.split(self.rng)
      sub_rng = rng
      sampled_idx = jnp.expand_dims(
          jax.random.categorical(sub_rng, topk_logits / temperature).astype(
              jnp.int32
          ),
          axis=-1,
      )
      print(f"topk_idxs {topk_idxs.shape} sampled_idx {sampled_idx.shape}")
      sampled_tokens = jnp.squeeze(
          jnp.take_along_axis(topk_idxs, sampled_idx, axis=-1), axis=-1
      ).astype(jnp.int32)

      return sampled_tokens

def sample_weighted_logits(logits, topk, temperature, rng):
  return jax.random.categorical(rng, logits / temperature)
  
def replicate_array_with_sharding(array):
    # Create a sharding with None for all dimensions (meaning replicated)
    mesh = jax.sharding.Mesh(jax.devices(), ('data',))  # or your existing mesh
    sharding = NamedSharding(
        mesh,
        PartitionSpec(None,) * len(array.shape)  # None for each dimension
    )
    return jax.device_put(array, sharding)

def test_custom_sampling():
  """test custom sampling performance"""
  batch_size = 96
  hidden_size = 8192
  rng = jax.random.key(0)
  logits = jax.random.normal(rng, (batch_size,1, hidden_size), dtype=float)
  logits = replicate_array_with_sharding(logits)


  topk = jax.random.randint(rng, (batch_size,), dtype=int, minval=1, maxval=3)
  temperature = jax.random.uniform(rng, (batch_size,), dtype=float, minval=0.0, maxval=1.0)
  
  samplers = []
  # sampler = sample_topk_logits
  sampler = sample_weighted_logits
  sampler = jax.jit(sampler)
  print(f"logits {logits[:, -1].shape}, topk {topk[0]}, temperature {temperature[0]}")
  sampler(logits[:, -1], topk[0], temperature[0], rng)

  for i in range(batch_size):
    rng, sub_rng = jax.random.split(rng)
    partial = jax.tree_util.Partial(sampler, topk=topk[i], temperature=temperature[i], rng=sub_rng)
    samplers.append(partial)

  custom_sampling = PyTorchEngine._custom_sampling
#   custom_sampling = jax.jit(custom_sampling)
  custom_sampling(logits, samplers)

  start = time.perf_counter()
  loops = 10
  # for i in range(loops):
  result = custom_sampling(logits, samplers)
  result.block_until_ready()
  end = time.perf_counter()
  duration = end - start
  print(f"Custom sampling: total time {duration} for {loops} loops")
  
  start = time.perf_counter()
  loops = 10
  # for i in range(loops):
  result = sampler(logits, topk[0], temperature[0], rng)
  result.block_until_ready()
  end = time.perf_counter()
  duration = end - start
  print(f"Uniform sampling: total time {duration} for {loops} loops")

test_custom_sampling()
