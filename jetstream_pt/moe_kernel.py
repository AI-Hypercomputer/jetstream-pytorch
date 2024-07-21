import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox.gmm import gmm

preferred_dtype = jnp.dtype('float32')


def gmm_auto_tile(lhs, rhs, group_sizes):
  m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[2]
  tm = min(256, m)
  tk = 256 if k == 1792 else (min(4096, k) )
  tn = 256 if n == 1792 else (min(4096, n) )
  gmm_result = gmm(
    lhs, #.astype(preferred_dtype), 
    rhs, #.astype(preferred_dtype), 
    group_sizes, tiling=(tm, tk, tn), 
    preferred_element_type=preferred_dtype)
  return gmm_result.astype(lhs.dtype)


def eval_gmm(x, w1, w2, w3, expert_indices):
    x = x.astype(preferred_dtype)
    w1 = w1.astype(preferred_dtype)
    w2 = w2.astype(preferred_dtype)
    w3 = w3.astype(preferred_dtype)
    def _histogram(input,  min_: int, max_: int):
        assert min_ <= max_, "min_ must be less than or equal to max_."
        def searchsorted(sorted_sequence, values_to_search):
            return (jnp.expand_dims(sorted_sequence, 1) == values_to_search).sum(axis=1)
        bin_edges = jnp.linspace(
            min_, max_, max_ - min_ + 1, dtype=input.dtype)
        return searchsorted(bin_edges, input)

    num_tokens, k = expert_indices.shape
    _, n = x.shape
    top_flat = expert_indices.flatten()
    hidden_states_order = top_flat.argsort()
    hidden_states_reverse_order = hidden_states_order.argsort()
    # Always replicated, so okay to skip manual sharding.
    hidden_states_indices = jnp.arange(num_tokens).repeat(k)[hidden_states_order]
    hidden_states_sorted = x[hidden_states_indices] # (num_tokens, hidden_dim/8)
    num_expert = w1.shape[0]
    group_sizes = _histogram(top_flat, 0, num_expert - 1)
    # w1 (num_experts, hiddent_dim/8, intermeditate)
    gmm1 = gmm_auto_tile(
      hidden_states_sorted, jnp.transpose(w1, (0, 2, 1)), group_sizes)
    gmm3 = gmm_auto_tile(
      hidden_states_sorted, 
      jnp.transpose(w3, (0, 2, 1)),
      group_sizes)
    silu = jax.nn.silu(gmm1)
    sgmm = silu * gmm3 # (num_tokens, intermediate_size)
    gmm2 = gmm_auto_tile(
      sgmm, 
      jnp.transpose(w2, (0, 2, 1)),
      group_sizes)
    #gmm2 = gmm2.astype(w1.dtype)
    gmm2 = jax.lax.psum(gmm2, 'x')
    current_hidden_states = gmm2[hidden_states_reverse_order].reshape(-1, k, n)
    return current_hidden_states