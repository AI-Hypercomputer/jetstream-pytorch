from jetstream.engine import tokenizer_api, token_utils


class HFTokenizerAdapter(tokenizer_api.Tokenizer):
  """Implementation of Tokenizer interface backed by HF tokenizer."""

  def __init__(self, tokenizer):
    self.tokenizer = tokenizer

  def encode(self, s: str, **kwargs):
    """Tokenize a string.
    Args:
        s: String to tokenize.
        **kwargs: Additional keyword arguments.
    Returns:
        tokens: Tokenized into integers.
        true_length: Actual length of the non-padded sequence
          if padding is used.
    """
    res = self.tokenizer.encode(s, add_special_tokens=False)
    return token_utils.pad_tokens(
        res, self.bos_id, self.pad_id, jax_padding=True
    )

  def decode(self, token_ids: list[int], **kwargs) -> str:
    """Processess input token ids to generate a string.
    Args:
      token_ids: List of token ids.
      **kwargs: Additional keyword arguments.
    Returns:
      str: String generated from the token ids.
    """
    return self.tokenizer.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.tokenizer.eos_token_id

  @property
  def bos_id(self) -> int:
    """ID of BOS token."""
    return self.tokenizer.bos_token_id

  @property
  def stop_tokens(self) -> set[int]:
    """ID of the stop token."""
    return {self.eos_id}
