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
      16: 0.01611422459827736, 
      32: 0.021419103996595367, 
      64: 0.028536087402608247, 
      128: 0.04435291379923001, 
      256: 0.07951394880074077,
      512: 0.1326028381998185,
      },
    decode_time = 0.424,
  ),
  Stat(
    cache_size = 1280,
    batch_size = 512,
    prefill_times = {
      16: 0.016024088603444397, 
      32: 0.020953572198050097, 
      64: 0.02750178739661351, 
      128: 0.041067036200547594, 
      256: 0.0747021451999899, 
      512: 0.13786499319830908, 
      1024: 0.23777181179611945},
    decode_time = 0.193,
  ),
  Stat(
    cache_size = 3072,
    batch_size = 256,
    prefill_times = {
      16: 0.016528940404532476,
      32: 0.021071263996418566,
      64: 0.028353967401199043,
      128: 0.04160083620226942,
      256: 0.07336672520032153,
      512: 0.1326028381998185,
      1024: 0.21599584679934197,
      2048: 0.34984797299839554},
    decode_time = 0.1693,
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


