from typing import Callable
import dataclasses
from collections import defaultdict
import jax
from jax import numpy as jnp

from jetstream.engine import engine_api

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main.py")

@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array 
  true_length: int


class OfflineInference:

  def __init__(self, engine: engine_api.Engine):
    self.engine = engine
    self.decode_state = engine.init_decode_state()
    self.params = engine.load_params()

    self.batch_size = engine.env.batch_size
    self.max_decode_length = engine.max_decode_length
    metadata = engine.get_tokenizer()
    self.tokenizer = engine.build_tokenizer(metadata)

  def warmup(self, max_length=2048):
    interesting_buckets = [
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]
    input_data = [
      InputData(str(i), jnp.ones(length, dtype=jnp.int32), true_length=length-1)
      for i, length in enumerate(interesting_buckets)
      if length <= max_length
    ]
    self.batch_inference(input_data)

  def batch_inference_with_callback(
    self, data: InputData, 
    emit_first_token: Callable[[str, int], bool],
    emit_token: Callable[[str, int], bool],
  ):
    """callback is a function that takes id and token. It will be called once per output

    token.
    """
    def prefill(slot, tokens, true_length):
      prefill_result, first_token = self.engine.prefill(
          params=self.params, 
          padded_tokens=tokens, 
          true_length=true_length
      )
      self.decode_state = self.engine.insert(
        prefill_result, self.decode_state, slot=slot)
      return first_token.data[0][0].item()

    empty_slots = list(range(self.batch_size))
    valid_slots = set()
    slot_to_id = dict()

    def decode():
      log.info('decode')
      nonlocal slot_to_id
      self.decode_state, result_tokens = self.engine.generate(
        self.params, self.decode_state)
      result_tokens = result_tokens.convert_to_numpy()
      # NOTA: result_tokens is of type ResultTokens is not subscriptable.
      newly_empty = []
      for slot, id_ in slot_to_id.items():
        token, is_valid, length = result_tokens.data[slot]
        log.info(f'slot is {slot}, length is {length}')
        if is_valid:
          should_finish = emit_token(id_, token.item())
        if should_finish or length >= self.max_decode_length:
          newly_empty.append(slot)

      # Add slots of those that are empty to emtpy
      for slot in newly_empty:
        del slot_to_id[slot]
        empty_slots.append(slot)

    for row in data:
      log.info(f'empty_slots {len(empty_slots)}')
      if empty_slots:
        log.info('prefill')
        slot = empty_slots.pop()
        first_token = prefill(slot, row.tokens, row.true_length)
        should_terminate = emit_first_token(row.id, first_token)
        if not should_terminate:
          slot_to_id[slot] = row.id
        else:
          empty_slots.append(slot) # dont use the slot 
      else:
        while not empty_slots:
          decode()

    while slot_to_id:
      log.info(f'slot to id {len(slot_to_id)}')
      decode()

  def batch_inference(self, data: InputData):
    """data is list of obj with id, tokens, and true length
    
    
    """
    ans = defaultdict(list)
    def callback(id_, token):
      ans[id_].append(token)
      return token == self.tokenizer.eos_id
    self.batch_inference_with_callback(data, emit_first_token=callback, emit_token=callback)
    return ans