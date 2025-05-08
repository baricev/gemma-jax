"""
Minimal, wrapper around SentencePiece that imitates the subset of
the Hugging Face tokenizer API used in the Gemma/JAX scripts.

Supports:
    - explicit bos_token_id handling (default 2)
    - uses EncodeAsIds / DecodeIds directly to avoid slow Python paths
    - preserves the Gemma whitespace sentinel when ingesting pre-split tokens

on examples from: https://github.com/google-deepmind/gemma/tree/main/gemma/gm/text
"""

from pathlib import Path
from typing import Any, Union, Sequence, Iterable

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import sentencepiece as spm
from typing import Optional

# --- Remaining processing functions ---
PAD_ID = 0
EOS_ID = 1
BOS_ID = 2

# --- Dialogue prompt/ answer wrappers ---
PROMPT_TEMPLATE = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
ANSWER_TEMPLATE = "{}<end_of_turn>"

# Unicode U+2581 "Lower One Eighth Block" used by SentencePiece to mark spaces.
_WHITESPACE_CHAR = "▁"

__all__ = ["SentencePieceTokenizer"]


class SentencePieceTokenizer:
  """
  SentencePiece wrapper that imitates the subset of
  the Hugging Face tokenizer API used in the Gemma/JAX scripts.
  """

  def __init__(
      self,
      model_file: Union[str, Path],  # path to the SentencePiece "tokenizer.model" file
      *,
      pad_token_id: int = 0,
      eos_token_id: int = 1,
      bos_token_id: int = 2,
  ) -> None:
    self.sp = spm.SentencePieceProcessor(model_file=str(model_file))

    # Public attrs for drop-in parity with HF
    self.pad_token_id = pad_token_id
    self.eos_token_id = eos_token_id
    self.bos_token_id = bos_token_id

  # Encoding helpers
  def encode(
      self,
      text: str | list[str],
      *,
      add_special_tokens: bool = True,
      **__,  # ignore HF-style kwargs
  ) -> list[int]:
    if isinstance(text, str):
      ids = self.sp.EncodeAsIds(text)
    else:  # already split into pieces
      ids = [self.sp.PieceToId(t.replace(" ", _WHITESPACE_CHAR)) for t in text]

    if add_special_tokens:
      ids = (
          ([self.bos_token_id] if self.bos_token_id is not None else [])
          + ids
          + ([self.eos_token_id] if self.eos_token_id is not None else [])
      )
    else:
      # Gemma 3 always expects a BOS [2] token
      ids = ([self.bos_token_id] if self.bos_token_id is not None else []) + ids

    return ids

  # Explicit list-of-texts variant:
  def batch_encode(self, texts: Sequence[str], add_special_tokens: bool = True) -> list[list[int]]:
    return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

  # Decoding helpers
  def decode(
      self,
      ids: Sequence[int],
      *,
      skip_special_tokens: bool = True,
      **__,
  ) -> str:
    if skip_special_tokens:
      ids = [i for i in ids if i not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
    return self.sp.DecodeIds(ids)

  def batch_decode(
      self,
      ids: np.ndarray | Iterable[Sequence[int]],
      *,
      skip_special_tokens: bool = True,
      **__,
  ) -> list[str]:
    if isinstance(ids, np.ndarray):
      ids = ids.tolist()
    return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in ids]

  # HF-style callable interface
  def __call__(
      self,
      texts: str | Sequence[str],
      *,
      return_tensors: str = "jax",
      padding: bool | str = True,
      padding_side: str = "right",
      max_length: int | None = None,
      add_special_tokens: bool = True,
      **__,
  ) -> dict[str, Any]:
    if isinstance(texts, str):
      texts = [texts]

    encs = [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
    max_len = max_length or max(len(e) for e in encs)

    def _pad(seq: Sequence[int]) -> list[int]:
      diff = max_len - len(seq)
      if diff <= 0:
        return seq[:max_len]
      pad = [self.pad_token_id] * diff
      return list(seq) + pad if padding_side == "right" else pad + list(seq)

    padded = np.asarray([_pad(e) for e in encs], dtype=np.int32)

    if return_tensors == "jax":
      import jax.numpy as jnp

      padded = jnp.asarray(padded)

    return {"input_ids": padded}

  # Factory for consistency with HuggingFace
  @classmethod
  def from_pretrained(cls, path: str | Path, **kwargs) -> "SentencePieceTokenizer":
    model_path = Path(path) / "tokenizer.model" if Path(path).is_dir() else path
    if not Path(model_path).exists():
      raise FileNotFoundError(model_path)
    return cls(model_path, **kwargs)


# --- Dialogue prompt/ answer wrappers ---
def format_prompt(prompt: str) -> str:
  return PROMPT_TEMPLATE.format(prompt)


def format_answer(answer: str) -> str:
  return ANSWER_TEMPLATE.format(answer)


# --- Example tokenizer loading ---
# sp_tokenizer = SentencePieceTokenizer("/some/absolute/path/tokenizer.model")
# hf_tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")


# --- Basic Tokenization Functions ---
def encode_text(text: Any, tokenizer: Any, max_length: int | None = None, add_special_tokens=True) -> jax.Array:
  """Encode text into token IDs using the tokenizer. Works with both HF and SentencePiece tokenizer."""
  return tokenizer(
      text if isinstance(text, list) else [text],  # required for HF
      truncation=True,  # Truncate to the model's max length
      return_tensors="jax",  # Return NumPy arrays (works with JAX)
      pad_to_max_length=True,
      # padding_side="right",   # default setting in HF
      max_length=max_length,
      add_special_tokens=add_special_tokens,  # Gemma 3 expects a BOS [2] token, we also add EOS [1] token to the end of the sequence
  )["input_ids"]


def decode_tokens(tokens: jax.Array, tokenizer: Any, skip_special_tokens=True) -> list[str]:
  """Decode token IDs  with either a HF or SentencePiece tokenizer."""
  # tokens: (B, L)  int32/bf16 on device
  tokens_host = np.asarray(jax.device_get(tokens), dtype=np.int32)
  return tokenizer.batch_decode(tokens_host, skip_special_tokens=skip_special_tokens)


# --- Tokenization and Padding use by model code  ---
def process_and_pad_inputs(
    input_text: Any,
    max_sequence_length: Optional[int],
    cache_len: Optional[int],
    tokenizer: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Tokenize and pad input text for the model using SentencePiece tokenizer.

  returning input ids, position ids, and attention mask.
  """

  def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

  # Use the encode text wrapper to handle both HF and SentencePiece tokenizers
  raw_ids = encode_text(input_text, tokenizer, max_length=max_sequence_length or None, add_bos_token_only=True)

  seq_len = raw_ids.shape[1]

  max_num_tokens = max_sequence_length or seq_len
  cache_length = cache_len or max_num_tokens
  assert cache_length >= max_num_tokens, "Cache length must be >= max_num_tokens."

  input_ids = jnp.pad(raw_ids, ((0, 0), (0, max_num_tokens - seq_len)), constant_values=PAD_ID)
  attn_mask = input_ids != PAD_ID

  # Build position ids using the Gemma 3 repository function
  position_ids = build_positions(attn_mask)

  # or simply use the following line:
  # position_ids = jnp.cumsum(attn_mask, axis=-1) - 1
  # Note: this appr. returns negative position ids (-1), if leading padding tokens are present:  [-1, -1, 0, 1, 1]  vs. [0, 0, 0, 1, 1]

  causal_attn = jnp.tril(jnp.ones((max_num_tokens, max_num_tokens), dtype=bool))
  causal_attn = attn_mask[:, None, :] & causal_attn[None, :, :]
  causal_attn = jnp.pad(
      causal_attn,
      ((0, 0), (0, 0), (0, cache_length - max_num_tokens)),
      constant_values=0,
  )
  return input_ids, position_ids, causal_attn
