# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
from jetstream_pt.third_party.llama import model_original
from flax import struct
from jetstream_pt.third_party.llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
  role: Role
  content: str


class CompletionPrediction(TypedDict, total=False):
  generation: str
  tokens: List[str]  # not required
  logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
  generation: Message
  tokens: List[str]  # not required
  logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


@struct.dataclass
class DecodeStateOriginal:
  prev_pos: int
  cur_pos: int
  tokens: torch.tensor
  out_tokens: List[List[int]]
  logits: torch.tensor
  input_text_mask: torch.tensor
  prompt_tokens: List[List[int]]


class LlamaOriginal:

  @staticmethod
  def build(
      tokenizer_path: str,
      model_args: model_original.ModelArgs,
      seed: int = 1,
  ) -> "LlamaOriginal":

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # seed must be the same in all processes
    torch.manual_seed(seed)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = model_original.Transformer(model_args)

    return LlamaOriginal(model, tokenizer)

  def __init__(self, model: model_original.Transformer, tokenizer: Tokenizer):
    self.model = model
    self.tokenizer = tokenizer

  @torch.inference_mode()
  def prefill(
      self,
      prompt_tokens: List[List[int]],
      max_gen_len: int,
  ) -> DecodeStateOriginal:
    """
    Do greedy search on CPU and return tokens only.
    """

    params = self.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = self.tokenizer.pad_id
    tokens = torch.full(
        (bsz, total_len), pad_id, dtype=torch.long, device="cpu"
    )
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

    prev_pos = 0
    input_text_mask = tokens != pad_id

    cur_pos = min_prompt_len
    logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    next_token = torch.argmax(logits[:, -1], dim=-1)

    next_token = next_token.reshape(-1)
    # only replace token if prompt has already been generated
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token

    prev_pos = cur_pos

    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      start = len(prompt_tokens[i])
      toks = toks[start : start + 1]
      probs = None
      out_tokens.append(toks)
      out_logprobs.append(probs)
    state = DecodeStateOriginal(
        prev_pos=cur_pos,
        cur_pos=cur_pos + 1,
        tokens=tokens,
        out_tokens=out_tokens,
        logits=logits,
        input_text_mask=input_text_mask,
        prompt_tokens=prompt_tokens,
    )

    return state

  @torch.inference_mode()
  def decode(
      self,
      decode_state: DecodeStateOriginal,
  ) -> DecodeStateOriginal:

    prev_pos = decode_state.prev_pos
    cur_pos = decode_state.cur_pos
    tokens = decode_state.tokens
    input_text_mask = decode_state.input_text_mask
    prompt_tokens = decode_state.prompt_tokens

    logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
    next_token = torch.argmax(logits[:, -1], dim=-1)

    next_token = next_token.reshape(-1)
    # only replace token if prompt has already been generated
    next_token = torch.where(
        input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
    )
    tokens[:, cur_pos] = next_token
    prev_pos = cur_pos

    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      toks = toks[cur_pos : cur_pos + 1]
      probs = None
      out_tokens.append(toks)
      out_logprobs.append(probs)

    state = DecodeStateOriginal(
        prev_pos=cur_pos,
        cur_pos=cur_pos + 1,
        tokens=tokens,
        out_tokens=out_tokens,
        logits=logits,
        input_text_mask=input_text_mask,
        prompt_tokens=prompt_tokens,
    )

    return state

  @torch.inference_mode()
  def generate(
      self,
      prompt_tokens: List[List[int]],
      max_gen_len: int,
  ) -> List[List[int]]:
    """
    Do greedy search on CPU and return tokens only.
    """

    params = self.model.params
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len
    total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

    pad_id = self.tokenizer.pad_id
    tokens = torch.full(
        (bsz, total_len), pad_id, dtype=torch.long, device="cpu"
    )
    for k, t in enumerate(prompt_tokens):
      tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device="cpu")
    input_text_mask = tokens != pad_id

    for cur_pos in range(min_prompt_len, total_len):
      logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
      next_token = torch.argmax(logits[:, -1], dim=-1)

      next_token = next_token.reshape(-1)
      # only replace token if prompt has already been generated
      next_token = torch.where(
          input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
      )
      tokens[:, cur_pos] = next_token
      eos_reached |= (~input_text_mask[:, cur_pos]) & (
          next_token == self.tokenizer.eos_id
      )
      prev_pos = cur_pos
      if all(eos_reached):
        break

    out_tokens, out_logprobs = [], []
    for i, toks in enumerate(tokens.tolist()):
      # cut to max gen len
      start = len(prompt_tokens[i])
      toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
      probs = None
      # cut to eos tok if any
      if self.tokenizer.eos_id in toks:
        eos_idx = toks.index(self.tokenizer.eos_id)
        toks = toks[:eos_idx]
      out_tokens.append(toks)
      out_logprobs.append(probs)
    return out_tokens

  def text_completion(
      self,
      prompts: List[str],
      temperature: float = 0.6,
      top_p: float = 0.9,
      max_gen_len: Optional[int] = None,
      logprobs: bool = False,
      echo: bool = False,
  ) -> List[CompletionPrediction]:
    """
    Perform text completion for a list of prompts using the language generation model.

    Args:
        prompts (List[str]): List of text prompts for completion.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

    Note:
        This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    if max_gen_len is None:
      max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = [
        self.tokenizer.encode(x, bos=True, eos=False) for x in prompts
    ]
    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
        echo=echo,
    )
    return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

  def chat_completion(
      self,
      dialogs: List[Dialog],
      temperature: float = 0.6,
      top_p: float = 0.9,
      max_gen_len: Optional[int] = None,
      logprobs: bool = False,
  ) -> List[ChatPrediction]:
    """
    Generate assistant responses for a list of conversational dialogs using the language generation model.

    Args:
        dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
            If not provided, it's set to the model's maximum sequence length minus 1.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

    Returns:
        List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

    Raises:
        AssertionError: If the last message in a dialog is not from the user.
        AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

    Note:
        This method generates assistant responses for the provided conversational dialogs.
        It employs nucleus sampling to introduce controlled randomness in text generation.
        If logprobs is True, token log probabilities are computed for each generated token.

    """
    if max_gen_len is None:
      max_gen_len = self.model.params.max_seq_len - 1
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:
      unsafe_requests.append(
          any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
      )
      if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
      assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
          [msg["role"] == "assistant" for msg in dialog[1::2]]
      ), (
          "model only supports 'system', 'user' and 'assistant' roles, "
          "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
      )
      dialog_tokens: List[int] = sum(
          [
              self.tokenizer.encode(
                  f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                  bos=True,
                  eos=True,
              )
              for prompt, answer in zip(
                  dialog[::2],
                  dialog[1::2],
              )
          ],
          [],
      )
      assert (
          dialog[-1]["role"] == "user"
      ), f"Last message must be from user, got {dialog[-1]['role']}"
      dialog_tokens += self.tokenizer.encode(
          f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
          bos=True,
          eos=False,
      )
      prompt_tokens.append(dialog_tokens)

    generation_tokens, generation_logprobs = self.generate(
        prompt_tokens=prompt_tokens,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        logprobs=logprobs,
    )
    if logprobs:
      return [
          {
              "generation": {
                  "role": "assistant",
                  "content": self.tokenizer.decode(t)
                  if not unsafe
                  else UNSAFE_ERROR,
              },
              "tokens": [self.tokenizer.decode(x) for x in t],
              "logprobs": logprobs_i,
          }
          for t, logprobs_i, unsafe in zip(
              generation_tokens, generation_logprobs, unsafe_requests
          )
      ]
    return [
        {
            "generation": {
                "role": "assistant",
                "content": self.tokenizer.decode(t)
                if not unsafe
                else UNSAFE_ERROR,
            }
        }
        for t, unsafe in zip(generation_tokens, unsafe_requests)
    ]
