# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import contextlib
import copy
import gc
import time
import math
import logging
import os
import sys
import array
import collections
import threading

import torch_xla2
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import mlperf_loadgen as lg
from jetstream_pt.config import create_engine_from_config_flags
from jetstream_pt import offline_inference

_MLPERF_ID = "mixtral-8x7b"

logging.basicConfig(level=logging.DEBUG)

sys.path.insert(0, os.getcwd())
log = logging.getLogger("main2.py")


from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "mlperf_test_mode",
    "performance",
    "performance, accuracy, submission",
)
flags.DEFINE_string(
    "api_url", None, "SAX published model path.", required=False
)
flags.DEFINE_string("dataset_path", None, "", required=False)
flags.DEFINE_bool("is_stream", False, "", required=False)
flags.DEFINE_string(
    "input_mode",
    "tokenized",
    "Input mode",
)
flags.DEFINE_string(
    "output_mode",
    "tokenized",
    "Output mode",
)

flags.DEFINE_string(
    "audit_conf",
    "audit.conf",
    "audit config for LoadGen settings during compliance runs",
    required=False,
)
flags.DEFINE_string(
    "mlperf_conf",
    "mlperf.conf",
    "mlperf rules config",
    required=False,
)
flags.DEFINE_string(
    "user_conf",
    "user.conf",
    "user config for user LoadGen settings such as target QPS",
    required=False,
)
flags.DEFINE_integer(
    "total_sample_count",
    15000,
    "Number of samples to use in benchmark.",
    required=False,
)
flags.DEFINE_integer(
    "perf_count_override",
    None,
    "Overwrite number of samples to use in benchmark.",
    required=False,
)
flags.DEFINE_string(
    "output_log_dir",
    "output-logs",
    "Where logs are saved.",
    required=False,
)
flags.DEFINE_bool(
    "enable_log_trace",
    False,
    "Enable log tracing. This file can become quite large",
    required=False,
)
flags.DEFINE_bool(
  "skip_warmup", False, "Skips warmup"
)
flags.DEFINE_bool(
  "internal_dummy_model", False, "Skips actual model compute, used for testing"
)
flags.DEFINE_integer(
  "log_every_n_complete", 100, "print after n completions"
)


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}


def pad_tokens(tokens):
  true_length = len(tokens)
  target_length = max(int(2 ** math.ceil(math.log2(true_length))), 32)
  padded = tokens + [0] * (target_length - true_length)
  return padded, true_length


@contextlib.contextmanager
def timed(msg):
  log.info(msg + " start")
  start = time.perf_counter()
  yield
  end = time.perf_counter()
  log.info(msg + " done: " + str(end - start))


def _classify_query(dataset_rows, index):
  # return groupped indexes
  if FLAGS.model_name == 'mixtral':
    sample = dataset_rows[index][1]
    input_len = sample.tok_input_len
    total_len = sample.tok_input_len + sample.tok_ref_output_len
    if total_len <= 512:
      return 0
    elif total_len <= 1280 and input_len <= 1024:
      return 1
    else:
      return 2
  else:
    sample = dataset_rows[index][1]
    total_len = sample.tok_input_length + (sample.tok_output_length * 3)
    if total_len <= 512:
      return 0
    elif total_len <= 1024:
      return 1
    else:
      return 2




def _pick_batch_size(num_samples, max_batch, dataset_size, sample_size):
  """max_batch to not run OOM."""
  if num_samples <= max_batch:
    return num_samples
  mult = math.ceil(num_samples / max_batch)
  return math.ceil(num_samples / mult * (sample_size / dataset_size))


def _log_complete(sample_id, response_token_ids):
  assert (response_token_ids[0] <= 32000)
  n_tokens = len(response_token_ids)
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  response_data = response_info[0]
  response_size = response_info[1] * response_array.itemsize
  query_sample_response = lg.QuerySampleResponse(
      sample_id, response_data, response_size, n_tokens
  )
  lg.QuerySamplesComplete([query_sample_response])
  _log_complete.count += 1
  # import ipdb; ipdb.set_trace()
  if (_log_complete.count) % FLAGS.log_every_n_complete == 0:
    log.info(f'{_log_complete.count} queries have completed')

_log_complete.count = 0

def _log_first(sample_id, response_token_ids):
  assert (response_token_ids[0] <= 32000)
  assert len(response_token_ids) == 1
  response_token_ids = np.array(response_token_ids, dtype=np.int64)
  response_array = array.array("B", response_token_ids.tobytes())
  response_info = response_array.buffer_info()
  first_token_response = lg.QuerySampleResponse(
      sample_id, response_info[0], response_info[1]
  )
  lg.FirstTokenComplete([first_token_response])


class SUT:

  def __init__(self, data, offline_inf):
    # dict of int (cache length) -> offline_inf
    self.offline_inf = offline_inf

    # pandas dataframe, it has tok
    self._dataset = data
    self.pandas_rows = list(self._dataset.iterrows())

    # List of things with .id and .index
    self._queries = None

    # index to loaded data
    self._processed_data = None

    # self.replicated = self.offline_inf.engine.env.sharding_by_axis(-1)
    self._sample_id_to_input = None
    self._sample_id_to_bucket = None
    self._groupped_queries = [[], [], []]
    self._id_to_index = {}
    self._index_to_group = {}
    self._eos = offline_inf[0].tokenizer.eos_id

  def _get_eos_seq(self, id_):
    idx = self._id_to_index[id_]
    pandas_row = self.pandas_rows[idx][1]
    if hasattr(pandas_row, 'tok_stop_sequence'):
      return pandas_row.tok_stop_sequence
    else:
      return [self._eos]

  def issue_queries(self, queries):
    log.info('issue queries called')
    self._groupped_queries = [[], [], []]
    assert self._sample_id_to_input is not None
    self._processed_data = []
    self._queries = queries
    for q in queries:
      self._id_to_index[q.id] = q.index
      group = self._index_to_group[q.index]
      input_data = copy.copy(self._sample_id_to_input[q.index])
      input_data.id = q.id
      self._groupped_queries[group].append(input_data)
    if len(self._queries) != sum(len(q) for q in self._groupped_queries):
      import ipdb; ipdb.set_trace()

    # At this point _processed_data is ready

  @timed("flush_queries")
  def flush_queries(self):
    start = time.perf_counter()
    completed = set()
    resp = collections.defaultdict(list)
    lock = threading.RLock()
    def emit_token(id_, token):
      nonlocal resp
      nonlocal completed
      with lock:
        resp[id_].append(token)
        end_seq = self._get_eos_seq(id_)
        is_end = (token == self._eos) or (end_seq == resp[id_][-len(end_seq):])
        if is_end:
          _log_complete(id_, resp[id_])
          completed.add(id_)
          if id_ in resp:
            del resp[id_]
        return is_end

    def emit_first_token(id_, token):
      nonlocal resp
      nonlocal completed
      # emit first token
      with lock:
        _log_first(id_, [token])
        end_seq = self._get_eos_seq(id_)
        is_end = (token == self._eos) or (len(end_seq) == 1 and end_seq[0] == token)
        if is_end:
          # We have four OpenOrca samples that return empty (eos_token) output.
          # It was decided to allow two eos_tokens to not break throughput computation
          # PR - https://github.com/mlcommons/inference/pull/1778
          _log_complete(id_, [token, self._eos])
          completed.add(id_)
          if id_ in resp:
            import pdb; pdb.set_trace()
            del resp[id_]
        return is_end

    for group_idx in [2,1,0]:
      group = self._groupped_queries[group_idx]
      self.offline_inf[group_idx].init_decode_state()
      result = self.offline_inf[group_idx].batch_inference_with_callback(
        group, emit_first_token, emit_token)

      # some never reached eos but reached max sequence
      with lock:
        for key, value in resp.items():
          if key in completed:
            continue
          _log_complete(key, value)
          completed.add(key)

      if group_idx != 0:
        # no need to drop state for the last one
        self.offline_inf[group_idx].decode_state = None
        gc.collect()

    end = time.perf_counter()

  def LoadSamplesToRam(self, sample_list):
    """Pads the data, move them to jax array on device"""
    log.info("LoadSamplesToRam start")
    start = time.perf_counter()
    input_data = {}

    for sample_id in sample_list:
      p = self.pandas_rows[sample_id][1]
      padded, length = pad_tokens(p.tok_input)
      input_data[sample_id] = offline_inference.InputData(
          "", jnp.array(padded), length  # to be filled later
      )
      self._index_to_group[sample_id] = _classify_query(self.pandas_rows, sample_id)

    for data in input_data.values():
      # make sure tokens are transfered to device
      jax.block_until_ready(data.tokens)

    self._sample_id_to_input = input_data

    end = time.perf_counter()
    log.info(f"LoadSamplesToRam finished: {end - start}s")

  def UnloadSamplesFromRam(self, sample_list):
    print("UnloadSamplesFromRam called")
    pass


def _count_by_bucket(dataset):
  if FLAGS.model_name == 'mixtral':
    total_len = dataset.tok_input_len + dataset.tok_ref_output_len

    group1 = total_len <= 512
    group2 = (total_len <= 1280) & (dataset.tok_input_len <= 1024)

    # with 5 percent extra
    mult = FLAGS.total_sample_count / len(dataset) * 1.05

    counts = [
        # power of 2
        math.ceil(len(dataset[group1]) * mult),
        math.ceil(len(dataset[~group1 & group2]) * mult),
        math.ceil(len(dataset[~group1 & ~group2]) * mult),
    ]
    return counts
  else:
    total_len = dataset.tok_input_length + dataset.tok_output_length
    group1 = total_len <= 512
    group2 = total_len <= 1024
    # with 5 percent extra
    mult = FLAGS.total_sample_count / len(dataset) * 1.05
    counts = [
        math.ceil(len(dataset[group1]) * mult),
        math.ceil(len(dataset[~group1 & group2]) * mult),
        math.ceil(len(dataset[~group1 & ~group2]) * mult),
    ]
    return counts


def main(argv):
  del argv
  args = FLAGS
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  if len(jax.devices()) < 4:
    print("Looks like TPU not available?", jax.devices())
    return -1
  # jax.config.update("jax_explain_cache_misses", True)

  settings = lg.TestSettings()
  settings.scenario = lg.TestScenario.Offline
  user_conf = FLAGS.user_conf

  settings.FromConfig(FLAGS.mlperf_conf, _MLPERF_ID, "Offline")
  settings.FromConfig(user_conf, _MLPERF_ID, "Offline")
  log.info("Mlperf config: %s", FLAGS.mlperf_conf)
  log.info("User config: %s", user_conf)

  dataset = pd.read_pickle(FLAGS.dataset_path)
  rows = list(dataset.iterrows())
  counts_by_bucket = _count_by_bucket(dataset)
  log.info(f"Counts by bucket {counts_by_bucket}")

  if FLAGS.model_name == "mixtral":
    length_and_batch = (
        (512, 2048),
        (1280, 512),
        (3072, 256),
    )
  else:
    length_and_batch = (
        (512, 512),
        (1024, 256),
        (2048, 96),
    )
  engines = []
  params = None
  for i, (length, max_batch) in enumerate(length_and_batch):
    batch = min(counts_by_bucket[i], max_batch)
    log.info(f"Using batch size of {batch} for {length}")
    engine = create_engine_from_config_flags(batch=batch, cache_len=length)
    offline_inf = offline_inference.OfflineInference(engine, params)
    offline_inf.dummy = FLAGS.internal_dummy_model
    params = offline_inf.params
    engines.append(offline_inf)

  if not FLAGS.skip_warmup:
    with timed("warmup"):
      for (length, _), engine in zip(length_and_batch, engines):
        log.info(f"warm up for {length}")
        engine.init_decode_state()
        engine.warmup(length)
        if length != 3072:
          # dont need to drop state for the last one
          engine.decode_state = None  # drop state
          gc.collect()

  sut = SUT(dataset, engines)

  if FLAGS.mlperf_test_mode == "accuracy":
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning(
        "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
    )
  elif FLAGS.mlperf_test_mode == "submission":
    settings.mode = lg.TestMode.SubmissionRun
    settings.print_timestamps = True
  else:
    settings.mode = lg.TestMode.PerformanceOnly
    settings.print_timestamps = True

  settings.use_token_latencies = True

  os.makedirs(FLAGS.output_log_dir, exist_ok=True)
  log_output_settings = lg.LogOutputSettings()
  log_output_settings.outdir = FLAGS.output_log_dir
  log_output_settings.copy_summary_to_stdout = True
  log_settings = lg.LogSettings()
  log_settings.log_output = log_output_settings
  log_settings.enable_trace = FLAGS.enable_log_trace

  lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
  qsl = lg.ConstructQSL(
      FLAGS.total_sample_count,
      FLAGS.total_sample_count,
      sut.LoadSamplesToRam,
      sut.UnloadSamplesFromRam,
  )
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(
      lgSUT, qsl, settings, log_settings, FLAGS.audit_conf
  )
  log.info(f"query counts {[len(q) for q in sut._groupped_queries]}")
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)
