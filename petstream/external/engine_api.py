"""Defines the API described in go/inference_engine_v2_api.

These functions are the accelerator functions which an outer sampling loop
could want to call, enabling token batched, disaggregated inference.
"""

import abc
from typing import Any, Optional, Tuple, Union

from flax import struct
import jax
import numpy as np

from petstream.external import tokenizer_pb2


# The model parameters - their partitioning will be unique for different prefill
# and decode topoologies.
Params = Any
# The result of a prefill operation, often a batch size 1 KVCache.
Prefix = Any
# The inputs into a generation step, often a prefill and generate cache tuple.
DecodeState = Any
# Accelerator representation of tokens.
DeviceTokens = Any
# Cpus asscociated with the mesh.
CpuDevices = Any


# pylint: disable=g-doc-args
@struct.dataclass
class ResultTokens(abc.ABC):
  """Class to store returned tokens in.

  We store everything in one array, and keep indexes - because copying
  a single array to host is much faster.
  Each tuple represents the indices of the relevant data.
  """

  # Shape: [batch, tokens.shape[1] + validity.shape[1] + lengths.shape[1]]
  data: Union[jax.Array, np.ndarray]
  # The range of indices which contain tokens.
  tokens_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the validity of
  # the tokens.
  valid_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  # The range of indices which contain the lengths up till now of the lengths
  # of each generated sequence.
  length_idx: tuple[int, int] = struct.field(
      pytree_node=False,
  )
  samples_per_slot: int = struct.field(
      pytree_node=False,
  )

  def copy_to_host_async(self: 'ResultTokens') -> None:
    """Copy to host asynchronously."""
    self.data.copy_to_host_async()

  def convert_to_numpy(self: 'ResultTokens') -> 'ResultTokens':
    """Converts to numpy."""
    return ResultTokens(
        np.array(self.data),
        self.tokens_idx,
        self.valid_idx,
        self.length_idx,
        self.samples_per_slot,
    )

  def get_result_at_slot(
      self, slot: int
  ) -> Tuple[
      Union[np.ndarray, jax.Array],
      Union[jax.Array, np.ndarray],
      Union[jax.Array, np.ndarray],
  ]:
    """Returns the token at a given slot.
    
    Args:
      slot: An integer from [0, n) representing an index into the batch.

    Note: implementations of this method must correctly handle
    microbatches, if microbatches are used.
    """
    # Potentially get multiple beams for given slot.
    slot_tokens = self.data[
        slot * self.samples_per_slot : (slot + 1) * self.samples_per_slot,
        self.tokens_idx[0]:self.tokens_idx[1],
    ]
    slot_valid = self.data[
        slot * self.samples_per_slot : (slot + 1) * self.samples_per_slot,
        self.valid_idx[0]:self.valid_idx[1],
    ]
    # Only get a 1D representation here
    slot_lengths = self.data[
        slot * self.samples_per_slot : (slot + 1) * self.samples_per_slot,
        self.length_idx[0]:self.length_idx[1],
    ][:, 0]
    # Mask out any non valid tokens.
    return slot_tokens, slot_valid, slot_lengths


class Engine(abc.ABC):
  """The computational core of the generative model server.

  Engine defines an API that models must adhere to as they plug into the
  Wiz efficient serving infrastructure.
  """

  @abc.abstractmethod
  def prefill(
      self,
      *,
      params: Params,
      existing_prefix: Optional[Prefix] = None,
      padded_tokens: jax.Array,
      true_length: int,
  ) -> Prefix:
    """Computes a kv-cache for a set of tokens conditional on existing cache.

    existing_prefix (if provided) represents a prefix that has already been
    processed by the underlying model. tokens is logically appended
    to the text represented by `existing_prefix`. This method returns a new
    kv_cache (typically) for the resulting text.
    """

  @abc.abstractmethod
  def generate(
      self,
      params: Params,
      decode_state: DecodeState
  ) -> Tuple[DecodeState, ResultTokens]:
    """Generates tokens for each sequence being decoded in parallel.

    Generate takes a batch of pre-computed kv-caches, and computes:
      - the predicted next token for each of the sequences
      - an updated set of kv-caches

    In the case of pipelining, this will handle N cycles (where each cycle
    consists of each microbatch progressing through every stage), in
    non-pipelined code this is a full forward pass. In both cases, this accounts
    for a full embed-layerstack-unembed-sample operation.
    """

  @abc.abstractmethod
  def insert(
      self,
      prefix: Prefix,
      decode_state: DecodeState,
      slot: int,
  ) -> DecodeState:
    """Adds `new_request` into `caches` at 'slot'.

    When decoding multiple requests in parallel, when one request finishes, a
    new request must be slotted into the recently vacated spot: `insert`!

    This can occur in between and async to generate calls, and takes a lock over
    that row of the cache.

    The slot may represent a tuple of positions (e.g. microbatch, pipeline stage
    and batch), but at the engine interface level all of these are exposed as
    a [0, n) range of slots and converted internally.
    """

  @abc.abstractmethod
  def load_params(self, *args, **kwargs) -> Params:
    """Loads parameters.

    May not be used in full production form, where weights are part of the saved
    model, but commonly used in the pathways setting.
    """

  @abc.abstractmethod
  def get_prefix_destination_sharding(self) -> Any:
    """Returns the shardings necessary to transfer data between engines."""

  @abc.abstractmethod
  def get_tokenizer(
      self,
  ) -> tokenizer_pb2.TokenizerParameters:
    """Returns the info to construct a sentencepiece tokenizer in py/c++."""

  @abc.abstractmethod
  def init_decode_state(self, *args, **kwargs) -> DecodeState:
    """Initialises any state which a generation step transforms."""

  @property
  @abc.abstractmethod
  def max_concurrent_decodes(self) -> int:
    """Total capacity."""

  @property
  @abc.abstractmethod
  def samples_per_slot(self) -> int:
    """Total samples per slot."""

  @property
  @abc.abstractmethod
  def max_prefill_length(self) -> int:
    """Maximum prefill length."""

  @property
  @abc.abstractmethod
  def mesh(self) -> jax.sharding.Mesh:
    """Mesh which the engine is running on."""

  @property
  @abc.abstractmethod
  def colocated_cpus(self) -> Union[list[CpuDevices], None]:
    """CPU devices colocated with the engine's accelerators."""
