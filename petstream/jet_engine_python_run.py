"""Run script of the Jet Stream API."""

from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import jax
from jax import lax
from jax import numpy as jnp

from petstream.external import engine_api
from petstream import jax_wrapper as jw
from petstream import jet_engine as je
import os

FLAGS = flags.FLAGS

_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path',
    'petstream/pets/tokenizer.model',
    'The tokenizer model path',
    required=False,
)
_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path', None, 'Directory for .pth checkpoints', required=False
)
_BF16_ENABLE = flags.DEFINE_bool(
    'bf16_enable', False, 'Whether to enable bf16', required=False
)
_CONTEXT_LENGTH = flags.DEFINE_integer(
    'context_length', 1024, 'The context length', required=False
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 1, 'The batch size', required=False
)
_PROFILING_OUTPUT =flags.DEFINE_string(
    'profiling_output',
    '',
    'The profiling output',
    required=False,
)

class JetEngineWrapper:
  """Wraps a JetEngine."""

  def __init__(
      self,
      devices: list[Any],
      tokenizer_path: str,
      checkpoint_path: str,
      bf16_enable: bool = False,
      context_length: int = 1024,
      batch_size: int = 1,
  ):
    self.engine = je.create_pytorch_engine(
        devices=devices,
        tokenizer_path=tokenizer_path,
        ckpt_path=checkpoint_path,
        bf16_enable=bf16_enable,
        param_size='tiny',
        context_length=context_length,
        batch_size=batch_size,
    )

  def _init_decode_state(self) -> jw.LoopState:
    return jw.LoopState(
        jnp.zeros((self.engine.param.max_batch_size, 1), dtype=jnp.int32),
        jw.initial_result(self.engine.param),
        jw.init_decode_state(
            self.engine.param,
            jw.init_cache,
            # jw.init_random_cache,
        ),
    )

  def _prefill_and_insert(
      self, weights, input_tokens, slot: int, decode_state: jw.LoopState
  ) -> jw.LoopState:
    """Prefills and inserts a slot."""

    prefix = self.engine.prefill(
        params=weights,
        prefill_inputs=input_tokens,
        true_length=jnp.array(len(input_tokens)),
    )
    print(f'Prefix: {prefix}, decode_state: {decode_state}')
    return self.engine.insert(
        prefix,
        decode_state,
        jnp.int32(slot),
    )

  def wrap_for_loop(self) -> Any:
    """Wraps a for loop as a function."""

    def func(weights, input_tokens):
      def body(loop_state):
        loop_state = jax.lax.cond(
            # Insert when the first slot has generated 1 token
            loop_state.decode_state.gen_len[0][0] == 1,
            lambda: self._prefill_and_insert(
                weights, input_tokens, 1, loop_state
            ),
            lambda: loop_state,
        )
        jax.debug.print('Jet loop_state before: {x}', x=loop_state)

        loop_state, _ = self.engine.generate(weights, loop_state)
        return loop_state

      def condition(loop_state):
        return loop_state.decode_state.gen_len[0][0] < 10

      init_decode_state = self._init_decode_state()
      insert_result = self._prefill_and_insert(
          weights, input_tokens, 0, init_decode_state
      )

      jax.debug.print('Jet Insert_result: {x}', x=insert_result)
      result = lax.while_loop(condition, body, insert_result).res
      return result

    return func


def main(argv: Sequence[str]) -> None:
  """Produces SavedModel from PytorchXLA compiled model."""
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

  for flag in FLAGS.flags_by_module_dict()[argv[0]]:
    logging.info('flag.name=%s, flag.value=%r', flag.name, flag.value)
  del argv

  num_of_partitions = jax.device_count()
  logging.info('Jet %s partitions exported!', num_of_partitions)

  jax.config.update('jax_traceback_filtering', 'off')

  jet = JetEngineWrapper(
      devices=jax.devices(),
      tokenizer_path=_TOKENIZER_PATH.value,
      bf16_enable=_BF16_ENABLE.value,
      context_length=_CONTEXT_LENGTH.value,
      batch_size=_BATCH_SIZE.value,
      checkpoint_path=_CKPT_PATH.value,
  )
  weights = jet.engine.load_params()

  jax_model = jet.wrap_for_loop()
  jax_model = jax.jit(jax_model)

  imported = jet.engine.imported_model
  # For inputs, always use batch size of 1
  batch_size = 1
  input_tokens = jnp.arange(
      batch_size * imported.model_args.max_seq_len, dtype=jnp.int32
  ).reshape((
      batch_size,
      imported.model_args.max_seq_len,
  ))
  if _PROFILING_OUTPUT.value:
   jax.profiler.start_trace(_PROFILING_OUTPUT.value)
  jax.block_until_ready(jax_model(weights, input_tokens))
  if _PROFILING_OUTPUT.value:
   jax.profiler.stop_trace()
  return


if __name__ == '__main__':
  app.run(main)
