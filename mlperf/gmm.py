import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from functools import partial
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

devices = mesh_utils.create_device_mesh((8, 1))
mesh = Mesh(devices, axis_names=('x', 'y'))
jax.config.update('jax_default_matmul_precision', "float32")
import torch
interp = False
pdtype = jnp.dtype('float32')

def _reference_gmm(lhs: torch.Tensor, rhs: torch.Tensor,
                   group_sizes: torch.Tensor, tiling=None, preferred_element_type=None, interpret=False) -> torch.Tensor:
    start = 0
    out = []
    for i, size in enumerate(group_sizes):
      result = lhs[start:start + size, :] @ rhs[i, :, :]
      out.append(result)
      start += group_sizes[i]
    return jnp.concatenate(out)

#gmm = _reference_gmm




@partial(
   shard_map, 
   mesh=mesh, 
   in_specs=(
       P(),
       P(None, 'x', None),
       P(None, None, 'x'),
       P(None, 'x', None),
       P(None, None)), 
   out_specs=(P()), check_rep=False)
def temp(x, w1, w2, w3, expert_indices):
    print('x inside', x.shape)
    print('w1', w1.shape)
    print('w2', w2.shape)
    print('w3', w3.shape)
    print('index', expert_indices.shape)
    def _histogram(input,  min: int, max: int):
        assert min <= max, "min must be less than or equal to max."

        def searchsorted(sorted_sequence, values_to_search):
            return (jax.numpy.expand_dims(sorted_sequence, 1) == values_to_search).sum(axis=1)

        bin_edges = jax.numpy.linspace(
            min, max, max - min + 1, dtype=input.dtype)
        return searchsorted(bin_edges, input)

    num_tokens, k = expert_indices.shape
    _, n = x.shape
    top_flat = expert_indices.flatten()
    hidden_states_order = top_flat.argsort()
    hidden_states_reverse_order = hidden_states_order.argsort()
    # Always replicated, so okay to skip manual sharding.
    hidden_states_indices = jnp.arange(num_tokens).repeat(k)[hidden_states_order]
    hidden_states_sorted = x[hidden_states_indices] # (num_tokens, hidden_dim/8)
    group_sizes = _histogram(top_flat, 0, 7)
    # w1 (num_experts, hiddent_dim/8, intermeditate)
    gmm1 = gmm(hidden_states_sorted.astype('bfloat16'), 
      jnp.transpose(w1, (0, 2, 1)).astype('bfloat16'), 
      group_sizes, tiling=(16,128,128), 
      preferred_element_type=pdtype, interpret=interp)
    gmm3 =  gmm(
      hidden_states_sorted.astype('float32'), 
      jnp.transpose(w3, (0, 2, 1)).astype('float32'), 
      group_sizes, tiling=(16,128,128), preferred_element_type=pdtype, interpret=interp)

    gmm1_s = gmm1
    gmm3_s = gmm3
    #gmm1_s = jax.lax.psum(gmm1, 'x')
    #gmm3_s = jax.lax.psum(gmm3, 'x')
    silu = jax.nn.silu(gmm1_s)
    sgmm = silu * gmm3_s # (num_tokens, intermediate_size)
    gmm2 = gmm(
      sgmm, 
      jnp.transpose(w2, (0, 2, 1)).astype('float32'), 
      group_sizes, 
      tiling=(8,512,512), 
      preferred_element_type=pdtype, interpret=interp) #(num_tokens, hidden_dim/8)
    print(gmm2.shape)
    gmm2 = jax.lax.psum(gmm2, 'x')
    current_hidden_states = gmm2[hidden_states_reverse_order].reshape(-1, k, n)
    return current_hidden_states


# Create a PRNG key
key = jax.random.PRNGKey(123)  # Using a different seed for variety

seqlen = 16

expert_indices = jax.random.randint(key, shape=(seqlen, 2), minval=0, maxval=8)
hidden_states = jax.random.normal(key, (seqlen, 4096), dtype=jnp.bfloat16)

w1 = jnp.broadcast_to(jnp.arange(8).reshape((8, 1, 1)).astype('float32'), (8, 14336, 4096))
w2 = jnp.broadcast_to(jnp.arange(8).reshape((8, 1, 1)).astype('float32'), (8, 4096, 14336))
w3 = jnp.broadcast_to(jnp.arange(8).reshape((8, 1, 1)).astype('float32'), (8, 14336, 4096))


hidden_states = jax.device_put(hidden_states, NamedSharding(mesh, P(None, "x")))
w1 = jax.device_put(w1, NamedSharding(mesh, P(None, "x")))
w2 = jax.device_put(w2, NamedSharding(mesh, P(None, None, "x")))
w3 = jax.device_put(w3, NamedSharding(mesh, P(None, "x")))

def exp_einsum(x, w1, expert_indices):
  w1_weights = w1[expert_indices] 
  x1 = jnp.einsum("ti,taoi -> tao", x, w1_weights)
  return x1

def _repeat_index(num_tokens, k):
  start = jnp.arange(num_tokens).repeat(k)
  start = start.reshape((num_tokens, k))
  return start.T.flatten()

def exp_gmm(x, w1, expert_indices):
  num_tokens, k = expert_indices.shape
  _, n = x.shape
  e, o, i = w1.shape 
  top_flat = expert_indices.flatten()
  hidden_states_order = top_flat.argsort()
  hidden_states_reverse_order = hidden_states_order.argsort()
  # Always replicated, so okay to skip manual sharding.
  hidden_states_indices = jnp.arange(num_tokens).repeat(k)[hidden_states_order]
  hidden_states_sorted = x[hidden_states_indices] # (num_tokens, hidden_dim/8)
  def _histogram(input,  min: int, max: int):
      assert min <= max, "min must be less than or equal to max."

      def searchsorted(sorted_sequence, values_to_search):
          return (jax.numpy.expand_dims(sorted_sequence, 1) == values_to_search).sum(axis=1)

      bin_edges = jax.numpy.linspace(
          min, max, max - min + 1, dtype=input.dtype)
      return searchsorted(bin_edges, input)

  group_sizes = _histogram(top_flat, 0, 7)
  gmm1 = gmm(hidden_states_sorted.astype('float32'), 
             jnp.transpose(w1, (0, 2, 1)).astype('float32'), 
             group_sizes, 
             tiling=(16,128,128), 
             preferred_element_type=pdtype, interpret=interp)
  return gmm1[hidden_states_reverse_order].reshape(-1, k, o)


def forward_for_long_seq_len(x, w1, w2, w3, expert_indices):
    seqlen = x.shape[0]
    num_experts = w1.shape[0]

    # e = total num of exp = 8
    # t = seqlen
    # o = config.imtermediate size
    # i = config.dim
    with jax.named_scope("conditional_ff"):
      x1 = jax.nn.silu(jnp.einsum("ti,eoi -> teo", x, w1))
      x3 = jnp.einsum("ti, eoi-> teo", x, w3)
      expert_outs = (
          jnp.einsum("teo, eio -> tei", (x1 * x3), w2)
      )
      # e = 8; need to reduce to 2
      seq_indexes = jnp.expand_dims(jnp.arange(seqlen), 1)
      return expert_outs[seq_indexes, expert_indices]

def main():

  # x = jnp.arange(8).reshape((2, 4)).astype('float64') / 100
  # w1 = jnp.arange(3 * 4 * 5).reshape((3,5,4)).astype('float64')
  # w2 = jnp.arange(3 * 4 * 5).reshape((3,4,5)).astype('float64')
  # w3 = jnp.arange(3 * 4 * 5).reshape((3,5,4)).astype('float64')
  # expert_indices = jnp.array([[0, 2], [1, 0]])
  x = hidden_states
  out1 = forward_for_long_seq_len(x, w1, w2, w3, expert_indices)
  out = temp(x, w1, w2, w3, expert_indices)
  print(jnp.max(jnp.abs(out1 - out)))

  # out1 = exp_einsum(x, w1, expert_indices)
  # out_gmm = exp_gmm(x, w1, expert_indices)
  # print(out1 - out_gmm)


  #group_sizes = jnp.array([4] * 8)
  #gmm1 = gmm(hidden_states.astype('float32'), 
  #            w1.astype('float32'), 
  #            group_sizes, 
  #            tiling=(16,128,128), interpret=interp)
  #gmm1_ref = _reference_gmm(hidden_states.astype('float32'), 
  #            w1.astype('float32'), 
  #            group_sizes, 
  #            tiling=(16,128,128), interpret=interp)
  #print(gmm1 - gmm1_ref)


if __name__ == '__main__':
  main()


