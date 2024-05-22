import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest
import torch
import torch_xla2
import jax
import jax.numpy as jnp


class JaxTorchTest(unittest.TestCase):
    """Unit test compare Jax and Torch gap with float precision"""

    def test_matmul_bfloat16_xla2(self):
        """test jax vs torch matmul diff with bfloat16 on cpu"""
        torch.set_default_dtype(torch.bfloat16)
        r = c = 1000
        q = torch.randn((r, c))
        k = torch.randn((r, c))
        print(f"torch matlmul: {q.shape} * {k.shape}")
        result = torch.matmul(q, k)

        jax_q = torch_xla2.tensor.t2j(q)
        jax_k = torch_xla2.tensor.t2j(k)
        print(f"torch matlmul: {jax_q.shape} * {jax_k.shape}")
        jax_result = jnp.matmul(jax_q, jax_k)
        target_result = torch_xla2.tensor.j2t(jax_result)
        print(
            f"----------------------- matmul: Diff norm {(target_result - result).norm()}"
        )
        self.assertTrue(torch.allclose(target_result, result, atol=1))

    def test_matmul_bfloat32(self):
        """test jax vs torch matmul diff with bfloat32 on cpu"""
        torch.set_default_dtype(torch.float32)
        r = c = 1000
        q = torch.randn((r, c))
        k = torch.randn((r, c))
        print(f"torch matlmul: {q.shape} * {k.shape}")
        result = torch.matmul(q, k)

        jax_q = torch_xla2.tensor.t2j(q)
        jax_k = torch_xla2.tensor.t2j(k)
        print(f"torch matlmul: {jax_q.shape} * {jax_k.shape}")
        jax_result = jnp.matmul(jax_q, jax_k)
        target_result = torch_xla2.tensor.j2t(jax_result)
        print(
            f"----------------------- matmul: Diff norm {(target_result - result).norm()}"
        )
        self.assertTrue(torch.allclose(target_result, result, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
