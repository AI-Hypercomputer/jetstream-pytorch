import math
import pandas as pd
import dataclasses
from collections import defaultdict
from absl import flags, app

from typing import Dict

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '', '')

@dataclasses.dataclass
class Stat:
  cache_size: int
  batch_size: int
  prefill_times: Dict[int, float]
  decode_time: float

scenario1 = [
  Stat(
    cache_size = 512,
    batch_size = 2048,
    prefill_times = {
      16: 0.016024088603444397, 
      32: 0.021154335999926843, 64: 0.02999803279999469, 128: 0.043986773600045125, 256: 0.07524209819985117, 512: 0.1388279377999
4246},
decode_time = 0.28033976474989686
  ),
  Stat(
    cache_size = 1280,
    batch_size = 512,
    prefill_times = {
      16: 0.016024088603444397, 
      32: 0.020686019999993734, 64: 0.02952769919993443, 128: 0.04383329960000992, 256: 0.07538782240008005, 512: 0.13893127239989553, 1024: 0.2693996697998955},
decode_time=0.11505070800001249,
  ),
  Stat(
    cache_size = 3072,
    batch_size = 256,
    prefill_times = {32: 0.021193669800049976, 64: 0.030565194799964956, 128: 0.04334795760005363, 256: 0.07586566419995507, 512: 0.13899565000010625, 1024: 0.26945373279995694, 2048: 0.35605709000010394},
    decode_time = 0.06467210225014242,
  )
]

def eval_scenario(dataset, scenario):
  total_len = dataset.tok_input_len + dataset.tok_ref_output_len
  bucket = 0 + (total_len > 512) + ((total_len > 1280) | (dataset.tok_input_len > 1024)) 
  dataset.insert(2, 'bucket', bucket)

  total_input_tokens = 0
  total_output_tokens = 0
  total_prefill_times = defaultdict(float)
  total_decode_times = defaultdict(float)
  output_tokens_by_bucket = defaultdict(int)
  for _, data in dataset.iterrows():
    stat = scenario[data.bucket]
    total_input_tokens += data.tok_input_len
    total_output_tokens += data.tok_ref_output_len
    input_len_bucket = 2**math.ceil(math.log2(data.tok_input_len))
    if input_len_bucket == 2048 and data.bucket == 1:
      import pdb; pdb.set_trace()
    total_prefill_times[input_len_bucket] += stat.prefill_times[input_len_bucket]
    output_tokens_by_bucket[data.bucket] += data.tok_ref_output_len
  
  for k in output_tokens_by_bucket.keys():
    stat = scenario[k]
    total_decode_times[k] = output_tokens_by_bucket[k] / stat.batch_size * scenario[k].decode_time

  print('Total input tokens', total_input_tokens)
  print('Total output tokens', total_output_tokens)
  print('Input / output', total_input_tokens / total_output_tokens)
  print('Prefill times', total_prefill_times)
  print('pref throughput', total_input_tokens / sum(total_prefill_times.values()))
  print('decode times', total_decode_times)
  print('decode throughput', total_output_tokens / sum(total_decode_times.values()) )
  print('overall throughput', 
   total_output_tokens / 
   (sum(total_decode_times.values()) + sum(total_prefill_times.values())))

    

def main(argv):
  dataset = pd.read_pickle(FLAGS.dataset_path)
  eval_scenario(dataset, scenario1)

if __name__ == '__main__':
  app.run(main)


