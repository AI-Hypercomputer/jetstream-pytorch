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

import gc
import time
import math
import logging
import os
import sys
import array

import torch_xla2
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from . import backend, dataset

import mlperf_loadgen as lg
from jetstream_pt.config import create_engine_from_config_flags
from jetstream_pt import offline_inference

_MLPERF_ID = "mixtral-8x7b"

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main.py")

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "scenario",
    "Offline",
    "Scenario",
)
flags.DEFINE_string(
    "api_url", None, "SAX published model path.", required=False
)
flags.DEFINE_string("dataset_path", None, "", required=False)
flags.DEFINE_bool("accuracy", False, "Run accuracy mode", required=False)
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


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
}

def pad_tokens(tokens):
    true_length = len(tokens)
    target_length = int(2 ** math.ceil(math.log2(true_length)))
    padded = tokens + [0] * (target_length - true_length)
    return padded, true_length


class SUT:

    def __init__(self, data, offline_inf):
        self.offline_inf = offline_inf

        # pandas dataframe, it has tok
        self._dataset = data

        # List of things with .id and .index
        self._queries = None

        # index to loaded data
        self._processed_data = None

        #self.replicated = self.offline_inf.engine.env.sharding_by_axis(-1)
        self._sample_id_to_input = None


    def issue_queries(self, queries):
        assert self._sample_id_to_input is not None
        self._processed_data = []
        self._queries = queries
        for q in queries:
            input_data = self._sample_id_to_input[q.index]
            input_data.id = q.id
            self._processed_data.append(input_data)
        
        assert len(self._queries) == len(q.id for q in self._queries)
        assert len(self._queries) == len(q.id for q in self._processed_data)
        # At this point _processed_data is ready

    def flush_queries(self):
        log.info('flush_queries called')
        assert self._processed_data
        result = self.offline_inf.batch_inference(self._processed_data)
        assert set(q.id for q in result.keys()) == set(q.id for q in self._processed_data)
        for key, val in result.items():
            key = int(key)
            lg.FirstTokenComplete([make_response(key, [val[0]])])
            resp = make_response(key, val)
            lg.QuerySamplesComplete([resp])
            log.info('recording query: ' + str(key))
        import ipdb; ipdb.set_trace()
        log.info('flush_queries done')
    
    def LoadSamplesToRam(self, sample_list):
        """Pads the data, move them to jax array on device"""
        log.info('LoadSamplesToRam start')
        start = time.perf_counter()
        input_data = {}
        pandas_rows = list(self._dataset.iterrows())

        for sample_id in sample_list:
            p = pandas_rows[sample_id][1]
            padded, length = pad_tokens(p.tok_input)
            input_data[sample_id] = offline_inference.InputData(
                '',  # to be filled later
                jnp.array(padded),
                length
            )

        for data in input_data.values():
            # make sure tokens are transfered to device
            jax.block_until_ready(data.tokens)  

        self._sample_id_to_input = input_data

        end = time.perf_counter()
        log.info(f'LoadSamplesToRam finished: {end - start}s')

    def UnloadSamplesFromRam(self, sample_list):
        print('UnloadSamplesFromRam called')
        pass


def make_response(id_, response_token_ids):
    n_tokens = len(response_token_ids)
    response_token_ids = np.array(response_token_ids, dtype=np.int64)
    response_array = array.array("B", response_token_ids.tobytes())
    response_info = response_array.buffer_info()
    response_data = response_info[0]
    response_size = response_info[1] * response_array.itemsize
    query_sample_response = lg.QuerySampleResponse(
        id_, response_data, response_size, n_tokens
    )
    return query_sample_response

def main(argv):
  del argv
  args = FLAGS
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")

  settings = lg.TestSettings()
  settings.scenario = scenario_map[args.scenario.lower()]
  user_conf = FLAGS.user_conf

  settings.FromConfig(FLAGS.mlperf_conf, _MLPERF_ID, FLAGS.scenario)
  settings.FromConfig(user_conf, _MLPERF_ID, FLAGS.scenario)
  log.info("Mlperf config: %s", FLAGS.mlperf_conf)
  log.info("User config: %s", user_conf)

  engine = create_engine_from_config_flags()
  offline_inf = offline_inference.OfflineInference(engine)

  dataset = pd.read_pickle(FLAGS.dataset_path)
  log.info('Warmup...')
  offline_inf.warmup(2048)
  log.info('Warmup done')


  sut = SUT(dataset, offline_inf)

  if FLAGS.accuracy:
    settings.mode = lg.TestMode.AccuracyOnly
    log.warning(
        "Accuracy run will generate the accuracy logs, but the evaluation of the log is not completed yet"
    )
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
  qsl = lg.ConstructQSL(100, 100, sut.LoadSamplesToRam, sut.UnloadSamplesFromRam)
  log.info("Starting Benchmark run")
  lg.StartTestWithLogSettings(
      lgSUT, qsl, settings, log_settings, FLAGS.audit_conf
  )
  log.info("Run Completed!")
  log.info("Destroying SUT...")
  lg.DestroySUT(lgSUT)

  log.info("Destroying QSL...")
  lg.DestroyQSL(qsl)


if __name__ == "__main__":
  # Disable garbage collection to avoid stalls when running tests.
  gc.disable()
  app.run(main)