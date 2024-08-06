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

import unittest
from unittest.mock import patch, MagicMock
from absl import app
from absl.testing import flagsaver
from parameterized import parameterized, param
from run_server import flags


class MockServer(MagicMock):
  """Mock server."""

  def run(self, **kwargs):
    """Run."""
    return self

  def wait_for_termination(self):
    """Wait for termination."""
    raise SystemExit("Successfully exited test.")


def mock_engine(**kwargs):
  """Mock engine."""
  return kwargs


class ServerRunTest(unittest.TestCase):
  """Server run test."""

  def reset_flags(self):
    """Reset flag."""
    flagsaver.restore_flag_values(self.original)

  def setup(self):
    """Setup."""

    f = flags.FLAGS
    # pylint: disable-next=all
    self.original = flagsaver.save_flag_values()
    return f

  @parameterized.expand(
      [
          param(["test1", "--model_name", "llama-3"], "llama-3"),
          param(["test2", "--model_name", "llama-2"], "llama-2"),
          param(["test3", "--model_name", "mixtral"], "mixtral"),
          param(["test4", "--model_name", "gemma"], "gemma"),
      ]
  )
  @patch("jetstream_pt.engine.create_pytorch_engine", mock_engine)
  @patch("jetstream.core.server_lib.run", MockServer().run)
  def test_no_change_from_defaults(self, args, expected):
    """test defaults remain unchanged when launching a server for different models.

    Args:
        args (List): List to simulate sys.argv with dummy first entry at index 0.
        expected (str): model_name flag value to inspect
    """
    # pylint: disable-next=all
    from run_server import main

    f = self.setup()
    with self.assertRaisesRegex(SystemExit, "Successfully exited test."):
      app.run(main, args)

    # run_server
    self.assertEqual(f.port, 9000)
    self.assertEqual(f.threads, 64)
    self.assertEqual(f.config, "InterleavedCPUTestServer")
    self.assertEqual(f.prometheus_port, 0)
    self.assertEqual(f.enable_jax_profiler, False)
    self.assertEqual(f.jax_profiler_port, 9999)

    # quantization configs
    self.assertEqual(f.quantize_weights, False)
    self.assertEqual(f.quantize_activation, False)
    self.assertEqual(f.quantize_type, "int8_per_channel")
    self.assertEqual(f.quantize_kv_cache, False)

    # engine configs
    self.assertEqual(f.tokenizer_path, None)
    self.assertEqual(f.checkpoint_path, None)
    self.assertEqual(f.bf16_enable, True)
    self.assertEqual(f.context_length, 1024)
    self.assertEqual(f.batch_size, 32)
    self.assertEqual(f.size, "tiny")
    self.assertEqual(f.max_cache_length, 1024)
    self.assertEqual(f.shard_on_batch, False)
    self.assertEqual(f.sharding_config, "")
    self.assertEqual(f.ragged_mha, False)
    self.assertEqual(f.starting_position, 512)
    self.assertEqual(f.temperature, 1.0)
    self.assertEqual(f.sampling_algorithm, "greedy")
    self.assertEqual(f.nucleus_topp, 0.0)
    self.assertEqual(f.topk, 0)
    self.assertEqual(f.ring_buffer, True)

    # profiling configs
    self.assertEqual(f.profiling_prefill, False)
    self.assertEqual(f.profiling_output, "")

    # model_name flag updates
    self.assertEqual(f.model_name, expected)

    # reset back to original flags
    self.reset_flags()

  @parameterized.expand([param(["test1", "--model_name", "llama3"])])
  @patch("jetstream_pt.engine.create_pytorch_engine", mock_engine)
  def test_call_server_object(self, args):
    """tests whether running the main script from absl.app.run launches a server
    and waits for termination

    Args:
        args (List): List to simulate sys.argv with dummy first entry at index 0.
    """
    with patch(
        "jetstream.core.server_lib.run", autospec=MockServer().run
    ) as mock_server:
      # pylint: disable-next=all
      from run_server import main

      _ = self.setup()
      with self.assertRaises(SystemExit):
        app.run(main, args)
      self.assertEqual(mock_server.call_count, 1)
      self.assertEqual(
          mock_server.return_value.wait_for_termination.call_count, 1
      )

      # reset back to original flags
      self.reset_flags()


if __name__ == "__main__":
  unittest.main()
