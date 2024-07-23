from typing import Callable
import dataclasses
from collections import defaultdict
import jax
from jax import numpy as jnp
import numpy as np

from jetstream.engine import engine_api

import logging

log = logging.getLogger(__name__)


@dataclasses.dataclass
class InputData:
  id: str
  tokens: jax.Array
  true_length: int


class OfflineInference:

  def __init__(self, engine: engine_api.Engine, params=None):
    self.engine = engine
    self.decode_state = None
    if params is None:
      params = engine.load_params()
    self.params = params

    self.batch_size = engine.env.batch_size
    self.max_decode_length = engine.max_decode_length
    metadata = engine.get_tokenizer()
    self.tokenizer = engine.build_tokenizer(metadata)
    self.dummy = False

    self._cached_pref = {}
    self._cached_generate = None

  def init_decode_state(self):
    if self.decode_state is None:
      self.decode_state = self.engine.init_decode_state()

  def warmup(self, max_length=2048):
    self.init_decode_state()
    interesting_buckets = [
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
    ]
    for length in interesting_buckets:
      if length > max_length:
        break
      log.info(f"Compiling prefill: {length}")
      input_data = jax.ShapeDtypeStruct((length,), jnp.dtype("int32"))
      self._cached_pref[length] = (
          jax.jit(self._prefill_insert, donate_argnums=(4,))
          .lower(
            self.params, 
            tokens=input_data, 
            slot=0, 
            true_length=length - 1, 
            decode_state=self.decode_state)
          .compile()
      )

    log.info(f"Compiling decode")
    self._cached_generate = (
        jax.jit(self.engine.generate, donate_argnums=(1,))
        .lower(self.params, self.decode_state)
        .compile()
    )

  def _prefill_insert(self, params, tokens, slot, true_length, decode_state):
    """return decodestate."""
    prefill_result, first_token = self.engine.prefill(
        params=params, padded_tokens=tokens, true_length=true_length
    )
    decode_state = self.engine.insert(prefill_result, decode_state, slot=slot)
    return first_token, decode_state

  def batch_inference_with_callback(
      self,
      data: InputData,
      emit_first_token: Callable[[str, int], bool],
      emit_token: Callable[[str, int], bool],
  ):
    """callback is a function that takes id and token. It will be called once per output

    token.
    """

    def prefill(slot, tokens, true_length):
      nonlocal self
      if self.dummy:
        log.debug("dummy prefill")
        return 123

      prefill_fn = self._prefill_insert
      if (cached := self._cached_pref.get(len(tokens))) is not None:
        prefill_fn = cached

      first_token, self.decode_state = prefill_fn(
          self.params, tokens=tokens, slot=slot, 
          true_length=true_length, decode_state=self.decode_state
      )
      return first_token.data[0][0].item()

    empty_slots = list(range(self.batch_size))
    slot_to_id = {}

    dummy_length = 1

    def decode():
      log.debug("decode")
      nonlocal self
      nonlocal slot_to_id
      nonlocal dummy_length
      if self.dummy:
        log.debug("Dummy generate")
        res = engine_api.ResultTokens(
            data=np.array([[123, 1, dummy_length]] * self.batch_size),
            tokens_idx=(0, 0),
            valid_idx=(0, 0),
            length_idx=(0, 0),
            samples_per_slot=(0, 0),
        )
        dummy_length += 1
        self.decode_state, result_tokens = self.decode_state, res
      else:
        gen_fn = self.engine.generate
        if self._cached_generate is not None:
          gen_fn = self._cached_generate
        self.decode_state, result_tokens = gen_fn(
            self.params, self.decode_state
        )

      result_tokens = result_tokens.convert_to_numpy()

      newly_empty = []
      for slot, id_ in slot_to_id.items():
        token, is_valid, length = result_tokens.data[slot]
        log.debug(f"slot is {slot}, length is {length}")
        should_finish = False
        if is_valid:
          should_finish = emit_token(id_, token.item())
        if should_finish or length >= self.max_decode_length:
          newly_empty.append(slot)

      # Add slots of those that are empty to emtpy
      for slot in newly_empty:
        del slot_to_id[slot]
        empty_slots.append(slot)

    for row in data:
      log.debug(f"empty_slots {len(empty_slots)}")
      while not empty_slots:
        # If slots are all full, decode until there are free slots
        # to insert
        decode()
      # do one insert
      log.debug(f"prefill {row.id}")
      slot = empty_slots.pop()
      first_token = prefill(slot, row.tokens, row.true_length)
      should_terminate = emit_first_token(row.id, first_token)
      if not should_terminate:
        slot_to_id[slot] = row.id
      else:
        empty_slots.append(slot)  # dont use the slot

    while slot_to_id:
      log.debug(f"slot to id {len(slot_to_id)}")
      decode()

  def batch_inference(self, data: InputData):
    """data is list of obj with id, tokens, and true length"""
    ans = defaultdict(list)

    def callback(id_, token):
      nonlocal ans
      ans[id_].append(token)
      return token == self.tokenizer.eos_id

    self.batch_inference_with_callback(
        data, emit_first_token=callback, emit_token=callback
    )
    return ans
