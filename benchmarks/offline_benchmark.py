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
    prefill_times = {16: 0.02084908019969589, 32: 0.024125573800120037, 64: 0.02697298339990084, 128: 0.03641403259971412, 256: 0.05809259879970341, 512: 0.10703752639965387},
    decode_time = 0.359
    #ecode_time = 0.28
  ),
  Stat(
    cache_size = 1280,
    batch_size = 512,
    prefill_times={16: 0.02070321020000847, 32: 0.02408570580009837, 64: 0.02650543759955326, 128: 0.036217428799864136, 256: 0.057748028799687746, 512: 0.10604073840004276, 1024: 0.20993155719988862},
    decode_time=0.094,
  ),
  Stat(
    cache_size = 3072,
    batch_size = 256,
    prefill_times={16: 0.020371186199918158, 32: 0.024281639599939807, 64: 0.02710893359981128, 128: 0.03605372060046648, 256: 0.0574128626001766, 512: 0.10610043820051943, 1024: 0.2097496903996216, 2048: 0.4301163775999157},
    decode_time = 0.0552,
  ),
]

scenario2 = [
  scenario1[2],
  scenario1[2],
  scenario1[2]
]
llama_run0 = Stat(
  cache_size = 512,
  batch_size = 512,
  prefill_times = {16: 0.02977288120018784, 32: 0.03336286680132616, 64: 0.03993437020108104, 128: 0.050098399000125936, 256: 0.07680363660037984, 512: 0.13509546700224745},
  decode_time=0.1300,
)
llama_run1 = Stat(
  cache_size = 1024,
  batch_size = 256,
  prefill_times = {16: 0.029987109202193097, 32: 0.03547879060206469, 64: 0.03783936820109375, 128: 0.05036673900031019, 256: 0.07769698440097272, 512: 0.135469542798819, 1024: 0.26336726440058555},
  decode_time=0.0757,
)
llama_run2 = Stat(
    cache_size = 2048,
    batch_size = 96,
    prefill_times = {16: 0.03186069100047462, 32: 0.03398795099928975, 64: 0.04061121419945266, 128: 0.05849674239871092, 256: 0.09263006160035729, 512: 0.175282979599433, 1024: 0.34900624539877756, 2048: 0.7503339296003105},
    decode_time = 0.0431,
)
scenario3 = [llama_run0, llama_run1, llama_run2]
def eval_scenario(dataset, scenario):

  total_input_tokens = 0
  total_output_tokens = 0
  total_prefill_times = defaultdict(float)
  total_decode_times = defaultdict(float)
  output_tokens_by_bucket = defaultdict(int)
  for _, data in dataset.iterrows():
    if hasattr(data, 'tok_input_len'):
      stat = scenario[data.bucket]
      total_input_tokens += data.tok_input_len
      total_output_tokens += data.tok_ref_output_len
      input_len_bucket = 2**math.ceil(math.log2(data.tok_input_len))
      total_prefill_times[input_len_bucket] += stat.prefill_times[input_len_bucket]
      output_tokens_by_bucket[data.bucket] += data.tok_ref_output_len
    else:
      stat = scenario[data.bucket]
      total_input_tokens += data.tok_input_length
      total_output_tokens += data.tok_output_length
      input_len_bucket = 2**math.ceil(math.log2(data.tok_input_length))
      total_prefill_times[input_len_bucket] += stat.prefill_times[input_len_bucket]
      output_tokens_by_bucket[data.bucket] += data.tok_output_length
  
  for k in output_tokens_by_bucket.keys():
    stat = scenario[k]
    total_decode_times[k] = output_tokens_by_bucket[k] / stat.batch_size * scenario[k].decode_time

  prefill_total = sum(total_prefill_times.values())
  decode_total = sum(total_decode_times.values())
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
  print('prefill total time', prefill_total)
  print('decode total time', decode_total)

    

def main(argv):
  dataset = pd.read_pickle(FLAGS.dataset_path)
  if hasattr(dataset, 'tok_input_len'):
    total_len = dataset.tok_input_len + dataset.tok_ref_output_len
    bucket = 0 + (total_len > 512) + ((total_len > 1280) | (dataset.tok_input_len > 1024)) 
    dataset.insert(2, 'bucket', bucket)
  else:
    total_len = dataset.tok_input_length + dataset.tok_output_length
    bucket = 0 + (total_len > 512) + (total_len > 1024) 
    dataset.insert(2, 'bucket', bucket)
  #eval_scenario(dataset, scenario1)
  print('======== scenario 2 ========')
  eval_scenario(dataset, scenario3)

if __name__ == '__main__':
  app.run(main)


