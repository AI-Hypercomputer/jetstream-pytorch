# pylint: disable-all
"""Tokenizer."""

import logging
from typing import List

from google3.pyglib import gfile
from google3.third_party.sentencepiece.src.python import sentencepiece_processor

logger = logging.getLogger()


class Tokenizer:
  """tokenizing and encoding/decoding text using SentencePiece."""

  def __init__(self, model_path: str):
    """Initializes the Tokenizer with a SentencePiece model.

    Args:
        model_path (str): The path to the SentencePiece model file.
    """
    # reload tokenizer
    assert gfile.Exists(model_path), model_path
    self.sp_model = sentencepiece_processor.SentencePieceProcessor()
    self.sp_model.Load(model_path)
    logger.info("Reloaded SentencePiece model from %s", model_path)

    # BOS / EOS token IDs
    # SentencePieceProcessor doesn't have vocab_size, use GetPieceSize() as
    # replacement.
    # self.n_words: int = self.sp_model.vocab_size()
    self.n_words: int = self.sp_model.GetPieceSize()
    self.bos_id: int = self.sp_model.bos_id()
    self.eos_id: int = self.sp_model.eos_id()
    self.pad_id: int = self.sp_model.pad_id()
    logger.info(
        "#words: %s - BOS ID: %s - EOS ID: %s",
        self.n_words,
        self.bos_id,
        self.eos_id,
    )
    # assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

  def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    """Encodes a string into a list of token IDs.

    Args:
        s (str): The input string to be encoded.
        bos (bool): Whether to prepend the beginning-of-sequence token.
        eos (bool): Whether to append the end-of-sequence token.

    Returns:
        List[int]: A list of token IDs.
    """
    assert isinstance(s, str)
    t = self.sp_model.EncodeAsIds(s)
    if bos:
      t = [self.bos_id] + t
    if eos:
      t = t + [self.eos_id]
    return t

  def decode(self, t: List[int]) -> str:
    """Decodes a list of token IDs into a string.

    Args:
        t (List[int]): The list of token IDs to be decoded.

    Returns:
        str: The decoded string.
    """
    return self.sp_model.DecodeIds(t)
