import unittest

import jax
import numpy as np
import jax.numpy as jnp
import torch

from jetstream_pt.third_party.llama import model_args
from jetstream_pt import environment
from jetstream_pt.page_attention_manager import PageAttentionManager
from jetstream_pt.cache_manager import PageKVCacheGenerate, KVCachePrefill
from absl.testing import parameterized

P = jax.sharding.PartitionSpec


class PageAttentionTest(parameterized.TestCase):

  def _make_env(self, bf16_enable=True):
    torch_dtype = torch.bfloat16 if bf16_enable else torch.float32
    torch.set_default_dtype(torch_dtype)
    jax.config.update("jax_dynamic_shapes", False)
    jax.config.update("jax_traceback_filtering", "off")
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", False)
    mesh = jax.sharding.Mesh(np.array(jax.devices()), axis_names=("x",))
    replicated = jax.sharding.NamedSharding(mesh, P())
    config = model_args.get_model_args("tiny", 128, 1, True)
    environment_data = environment.JetEngineEnvironmentData()
    environment_data.max_input_sequence_length = 128
    environment_data.max_input_sequence_length = 128
    environment_data.cache_sequence_length = 128
    environment_data.bf16_enable = bf16_enable
    environment_data.model_type = "llama-2-tiny"
    environment_data.batch_size = 3
    environment_data.num_layers = config.n_layers
    environment_data.cache_shape = (
        1,
        config.n_kv_heads,
        environment_data.cache_sequence_length,
        config.dim // config.n_heads,
    )
    env = environment.JetEngineEnvironment(environment_data)
    env.apply_sharding = lambda *args, **kwargs: None  # don't shard on cpu
    env.sharding = replicated
    return env, config

  def test_prefill_insert(self):

    env, _ = self._make_env()

    pam = PageAttentionManager(
        batch_size=3,
        paged_attention_total_num_pages=20,
        paged_attention_page_size=4,
        max_pages_per_sequence=4,
    )
    shape = (1, 6, 4, 2)
    decode_caches = []
    decode_caches.append(
        PageKVCacheGenerate.empty(shape=shape, device=None, env=env)
    )
    decode_caches = [c.state() for c in decode_caches]

    prefill_chache = KVCachePrefill()
    k, v = jnp.arange(6), jnp.arange(6)
    k, v = jnp.reshape(k, (1, 1, 3, 2)), jnp.reshape(k, (1, 1, 3, 2))
    prefill_chache.update(k, v, 0)
    prefill_caches = [prefill_chache]
    prefill_caches = [c.state() for c in prefill_caches]

    num_pages, update_indexes = pam.reserve_pages_insert(0, 3)
    _, kv_heads, _, dim = prefill_caches[0][0].shape
    tep_kv = jnp.zeros((kv_heads, num_pages * 4, dim), dtype=jnp.bfloat16)

    caches = pam.insert_prefill_cache(
        prefill_caches=prefill_caches,
        decode_caches=decode_caches,
        update_indexes=update_indexes,
        tep_kv=tep_kv,
        sharding=env.sharding,
    )
    expected_kv = jnp.arange(6).reshape(3, 2)
    padding = jnp.asarray([[0, 0]])
    expected_kv = jnp.concatenate((expected_kv, padding))

    self.assertTrue(
        jnp.array_equal(
            caches[0][0][0, 0, 0:4, 0:2], expected_kv.astype(jnp.bfloat16)
        )
    )
    self.assertTrue(
        jnp.array_equal(
            caches[0][1][0, 0, 0:4, 0:2], expected_kv.astype(jnp.bfloat16)
        )
    )

  def test_prefill_insert_multiple_pages(self):

    jax.config.update("jax_platform_name", "cpu")
    print(f"---------> {jax.devices()}")

    env, _ = self._make_env()

    pam = PageAttentionManager(
        batch_size=3,
        paged_attention_total_num_pages=20,
        paged_attention_page_size=4,
        max_pages_per_sequence=4,
    )
    shape = (1, 20, 4, 2)
    decode_caches = []
    decode_caches.append(
        PageKVCacheGenerate.empty(shape=shape, device=None, env=env)
    )
    decode_caches = [c.state() for c in decode_caches]

    self.cache_sharding = env.cache_sharding

    prefill_chache = KVCachePrefill()
    k, v = jnp.arange(12), jnp.arange(12)
    k, v = jnp.reshape(k, (1, 1, 6, 2)), jnp.reshape(k, (1, 1, 6, 2))
    prefill_chache.update(k, v, 0)
    prefill_caches = [prefill_chache]
    prefill_caches = [c.state() for c in prefill_caches]

    num_pages, update_indexes = pam.reserve_pages_insert(0, 6)
    _, kv_heads, _, dim = prefill_caches[0][0].shape
    tep_kv = jnp.zeros((kv_heads, num_pages * 4, dim), dtype=jnp.bfloat16)

    decode_caches = pam.insert_prefill_cache(
        prefill_caches=prefill_caches,
        decode_caches=decode_caches,
        update_indexes=update_indexes,
        tep_kv=tep_kv,
        sharding=env.sharding,
    )

    self.assertEqual(len(decode_caches), 1)
    expected = jnp.arange(16).at[12:16].set([0, 0, 0, 0]).reshape(1, 2, 4, 2)

    updated_k = jax.lax.slice_in_dim(decode_caches[0][0], 0, 2, axis=1)
    self.assertTrue(jnp.array_equal(updated_k, expected))
    noupdated_k = jax.lax.slice_in_dim(decode_caches[0][0], 2, 20, axis=1)
    self.assertTrue(jnp.array_equal(noupdated_k, jnp.zeros_like(noupdated_k)))

  def test_reserve_pages_decode(self):

    env, _ = self._make_env()

    pam = PageAttentionManager(
        batch_size=3,
        paged_attention_total_num_pages=20,
        paged_attention_page_size=4,
        max_pages_per_sequence=4,
    )
    slot = 1
    seq_len = 8
    pam.reserve_pages_insert(slot, seq_len)
    expected_slot_page_indices = jnp.asarray([0, 1])
    slot_page_indices = pam.page_indices[slot][0:2]
    self.assertTrue(
        jnp.array_equal(slot_page_indices, expected_slot_page_indices)
    )

    lens = jnp.asarray([0, seq_len, 0])
    pam.fill_new_pages(lens)
    expected_slot_page_indices = jnp.asarray([0, 1, 2, 19])
    slot_page_indices = pam.page_indices[slot]
    self.assertTrue(
        jnp.array_equal(slot_page_indices, expected_slot_page_indices)
    )

    expected_0_page_indices = jnp.asarray([19, 19, 19, 19])
    zer0_page_indices = pam.page_indices[0][0:4]
    self.assertTrue(jnp.array_equal(zer0_page_indices, expected_0_page_indices))

  def test_get_page_token_indices(self):
    env, _ = self._make_env()

    pam = PageAttentionManager(
        batch_size=5,
        paged_attention_total_num_pages=20,
        paged_attention_page_size=4,
        max_pages_per_sequence=4,
    )
    pam.reserve_pages_insert(1, 8)
    pam.reserve_pages_insert(3, 13)
    pam.reserve_pages_insert(0, 3)

    lens = jnp.asarray([3, 8, 0, 13, 0])
    pam.fill_new_pages(lens)

    page_token_indices = pam.get_page_token_indices(lens)

    expected_page_indices = jnp.asarray([6, 7, 5])
    expected_token_indices = jnp.asarray([3, 4, 9])
    self.assertTrue(
        jnp.array_equal(page_token_indices[0], expected_page_indices)
    )
    self.assertTrue(
        jnp.array_equal(page_token_indices[1], expected_token_indices)
    )


if __name__ == "__main__":
  unittest.main()
