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

import json

CUTOFF_INPUT = 1024
CUTOFF_OUTPUT = 1024

# batch size 60, ful cache, bfloat
prefill_bucket_size_to_s = {64: 0.007696230011060834, 128: 0.011508351005613805, 256: 0.01721684739459306, 512: 0.03257157760672271, 1024: 0.08185497261583805}

# batch size 96, ful cache, quantized
prefill_bucket_size_to_s = {64: 0.006911616190336645, 128: 0.011646182998083532, 256: 0.01875854718964547, 512: 0.0334438294172287, 1024: 0.0643601292045787}

# batch size 96, rolling, bfloat
prefill_bucket_size_to_s = {64: 0.007730783987790346, 128: 0.011515899002552033, 256: 0.01780580161139369, 512: 0.03115477201063186, 1024: 0.07443338260054588}

# batch size 160, rolling, quantized 
prefill_bucket_size_to_s = {64: 0.006821704190224409, 128: 0.01175499300006777, 256: 0.018776051187887787, 512: 0.03392685519065708, 1024: 0.06476318498607725}

prefill_bucket_size_to_ms = {k: p*1000 for k, p in prefill_bucket_size_to_s.items() }

# batch size 60, ful cache, bfloat
system_time_per_decode_token_ms =26.55/60

# batch size 96, ful cache, quantized
system_time_per_decode_token_ms =26.0/96

# batch size 96, rolling, bfloat
system_time_per_decode_token_ms =28.18/96

# batch size 160, rolling, quantized 
system_time_per_decode_token_ms =30/160

def do_simulation(prefill_bucket_size_to_ms, system_time_per_decode_token_ms):

    def next_power_of_2(x):  
        return 1 if x == 0 else 2**(x - 1).bit_length()

    def tokens_in_input_str(s):
        return_val =  int(1.3 * len(s.split()))
        #print(f"{s=} -> {return_val=}")
        return return_val

    convo_numbers = []
    # Please update with your own data file path
    loaded_share_gpt = json.load(open('~/data/ShareGPT_V3_unfiltered_cleaned_split.json', 'r'))
    for example in loaded_share_gpt:
        if len(example['conversations']) < 2:
            continue
        input_tokens = tokens_in_input_str(example['conversations'][0]['value'])
        output_tokens = tokens_in_input_str(example['conversations'][1]['value'])
        convo_numbers.append( (input_tokens, output_tokens))

    num_convos = len(convo_numbers)
    kept_convos = [c for c in convo_numbers if c[0] <= CUTOFF_INPUT and c[1] <= CUTOFF_OUTPUT]

    mean_input = sum([c[0] for c in kept_convos]) / len(kept_convos)
    mean_output = sum([c[1] for c in kept_convos]) / len(kept_convos)

    print(f"Total {num_convos=} but only kept {kept_convos=}. Out of kept, {mean_input=}, {mean_output=}")

    total_prefill_system_ms = 0
    total_generate_system_ms = 0

    total_system_output_tokens = 0
    for convo in kept_convos:
        input_tok, output_tok = convo
        bucket = max(128,next_power_of_2(input_tok))
        generate_system_ms = output_tok * system_time_per_decode_token_ms 
        prefill_system_ms = prefill_bucket_size_to_ms[bucket]

        print(f"{convo=} {bucket=}, {prefill_system_ms=:.2f}, {generate_system_ms=:.2f}")

        total_prefill_system_ms += prefill_system_ms
        total_generate_system_ms += generate_system_ms

    total_time_ms = total_prefill_system_ms + total_generate_system_ms
    input_tokens = sum([c[0] for c in kept_convos])

    output_tokens = sum([c[1] for c in kept_convos])
    print(f"Output tokens {output_tokens} in {total_time_ms/1000:.2f} seconds, for {output_tokens/(total_time_ms/1000):.2f} out tok/s")

    total_prefill_sec = total_prefill_system_ms/1000
    total_generate_sec = total_generate_system_ms/1000

    print(f"Total time {total_time_ms/1000:.2f} seconds, split {total_prefill_sec=:.2f} seconds and {total_generate_sec=:.2f} seconds")

    idealized_prefill_sec = 1.1 * input_tokens/1024 * prefill_bucket_size_to_ms[1024] / 1000

    prefill_savings_sec = total_prefill_sec-idealized_prefill_sec


    idealized_generate_sec = total_generate_sec/2 # (Roughly save 75% on KV cache high cost on the rest)
    generate_savings_sec = total_generate_sec - idealized_generate_sec

    print(f"we think prefill will take {total_prefill_sec=:.2f}, we could get it to {idealized_prefill_sec=:.2f} so we'd save {prefill_savings_sec=:.2f} seconds ")
    print(f"with sparsity we could go from  {total_generate_sec=:.2f}, we could get it to {idealized_generate_sec=:.2f} so we'd save {generate_savings_sec=:.2f} seconds ")

    idealized_overall_time = idealized_generate_sec + idealized_prefill_sec

    print(f"Idealized out tokens {output_tokens} in {idealized_overall_time:.2f} seconds, for {output_tokens/idealized_overall_time:.2f} out tok/s")
    print('prfill', prefill_bucket_size_to_ms)
    print('decode step', system_time_per_decode_token_ms)
