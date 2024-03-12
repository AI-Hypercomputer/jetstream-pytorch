import torch_xla2
import jax
import jax.numpy as jnp
import torch


class CacheInterface:
    # cache for ONE layer

    def update(self, key, value):
        """Update the cache for this key and value.
        
        The key, and val will have shape (Batch, Heads, Seqlen, Head dim)
        The cache is free to store them in a different format.
        Return the full cache after update.
        This cache instance need to know which position / layer is 
        the update for.
        """

class KVCachePrefill:

    def update(self, key, value):
        """This cache just remembers the stuff."""
        self.cache_k = key
        self.cache_v = value
        return key, value

    def state(self):
        return self.cache_k, self.cache_v



# Refactor out cache management
# Easier to test for quantized kv cache
class KVCacheGenerate:

    def __init__(self, 
        cache_k: torch.Tensor,  # previous cache
        cache_v: torch.Tensor,  # previous cache
        position: int,  # position to store the cache
        sharding,
    ):
        super().__init__()
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.pos = position
        self.sharding = sharding

    def update(self, key, value):
        keyj, valuej = torch_xla2.tensor.unwrap((key, value))
        keyj = jax.lax.with_sharding_constraint(keyj, self.sharding)
        valuej = jax.lax.with_sharding_constraint(valuej, self.sharding)
        self.cache_k._elem = self.cache_k._elem.at[:, :, self.pos].set(keyj)
        self.cache_v._elem = self.cache_v._elem.at[:, :, self.pos].set(valuej)
        return self.cache_k, self.cache_v 

    def state(self):
        return self.cache_k, self.cache_v

    @classmethod
    def empty(cls, shape, device):
        k = jnp.zeros(shape, device=device)
        v = jnp.zeros(shape, device=device)
        k, v = torch_xla2.tensor.wrap((k, v))
        pos = jnp.array([0])  # replicated
        return cls(k, v, 0, device)
        

class Int8KVCache:

    def __init__(self, 
        cache_k, 
        cache_v, 
        # input_pos,  # used to write cache
        # context_pos, #  used to read cache
        prefill = True,
        input_indexes = None,
        sharding = None,
    ):
        super().__init__()
        # These things are jax arrays!
        self.cache_k = torch_xla2.tensor.wrap(cache_k)  #<bsz, heads, seq, dim> x int8
        self.cache_v = torch_xla2.tensor.wrap(cache_v)
        bsz, heads, _, dim = cache_k.shape
        self.k_scaler = torch_xla2.tensor.wrap(jnp.ones((bsz, heads, 1, dim), dtype=jnp.bfloat16, device=sharding))
        self.v_scaler = torch_xla2.tensor.wrap(jnp.ones((bsz, heads, 1, dim), dtype=jnp.bfloat16, device=sharding))

        # self.input_pos = input_pos
        # self.context_pos = context_pos
        self.max_seq_len = cache_k.shape[2]
        self.prefill = prefill
        self.input_indexes = input_indexes

    @classmethod
    def empty(cls, shape, device):
        pass



    def update(self, xk, xv):
        # k_val: (bsz, numheads, seq len, head dim)
        # cache_shape = (Batch heads seqlen headdim)
        if self.prefill:
            bsz, heads, seqlen, head_dim = xk.shape
            return xk.reshape(seqlen, bsz, heads, head_dim), xv.reshape(seqlen, bsz, heads, head_dim)
            # Assumes prefill only
            self.cache_k = xk
            self.cache_v = xv
            return self.cache_k, self.cache_v
        else:
            self.cache_k, self.cache_v = torch_xla2.extra.call_jax(
                update_caches_jax, self.cache_k, self.cache_v, xk, xv, self.input_indexes)
            return self.cache_k * self.k_scaler, self.cache_v * self.v_scaler

import functools
def update_caches_jax(cache_k, cache_v, xk, xv, position):
    # import pdb; pdb.set_trace()
    cache_k = cache_k.at[position].set(xk.astype(jnp.int8).squeeze(2))
    cache_v = cache_v.at[position].set(xv.astype(jnp.int8).squeeze(2))
    return cache_k, cache_v
