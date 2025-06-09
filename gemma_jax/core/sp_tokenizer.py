# %%
"""
Minimal, wrapper around SentencePiece that imitates the subset of
the Hugging Face tokenizer API used in the Gemma/JAX scripts.

Supports:
    - explicit bos_token_id handling (default 2)
    - uses EncodeAsIds / DecodeIds directly to avoid slow Python paths
    - preserves the Gemma whitespace sentinel when ingesting pre-split tokens

Based on examples from: https://github.com/google-deepmind/gemma/tree/main/gemma/gm/text
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
      add_bos_id: bool = True,
      add_eos_id: bool = True,
      return_raw_tokens: bool = False,
      **__,  # ignore HF-style kwargs
  ) -> list[int]:
    if isinstance(text, str):
      ids = self.sp.EncodeAsIds(text)
    else:  # already split into pieces
      ids = [self.sp.PieceToId(t.replace(" ", _WHITESPACE_CHAR)) for t in text]
    if return_raw_tokens:
      return ids

    # If either add_bos_id or add_eos_id is explicitly set, ignore add_special_tokens
    if (add_bos_id is False) or (add_eos_id is False):
      if add_bos_id:
        ids = [self.bos_token_id] + ids
      if add_eos_id:
        ids = ids + [self.eos_token_id]
    else:
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
  def batch_encode(
      self, texts: Sequence[str], add_special_tokens: bool = True, add_bos_id: bool = True, add_eos_id: bool = True
  ) -> list[list[int]]:
    return [
        self.encode(t, add_special_tokens=add_special_tokens, add_bos_id=add_bos_id, add_eos_id=add_eos_id) for t in texts
    ]

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
      add_bos_id: bool = True,
      add_eos_id: bool = True,
      **__,
  ) -> dict[str, Any]:
    if isinstance(texts, str):
      texts = [texts]

    encs = [
        self.encode(t, add_special_tokens=add_special_tokens, add_bos_id=add_bos_id, add_eos_id=add_eos_id) for t in texts
    ]
    max_len = max_length or max(len(e) for e in encs)

    def _pad(seq: Sequence[int]) -> list[int]:
      diff = max_len - len(seq)
      if diff <= 0:
        return seq[:max_len]
      pad = [self.pad_token_id] * diff
      return list(seq) + pad if padding_side == "right" else pad + list(seq)

    padded_list = [_pad(e) for e in encs] # Pad each sequence to max_length

    if return_tensors == "jax":
      import jax.numpy as jnp

      padded = jnp.asarray(padded_list)
    elif return_tensors == "np":
      padded = np.asarray(padded_list, dtype=np.int32)
    elif return_tensors == "None":
      padded = encs  # Return list of lists
    else:
      raise ValueError(f"Unsupported return_tensors: {return_tensors}")
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
def encode_text(
    text: Any,
    tokenizer: Any,
    max_length: int | None = None,
    add_special_tokens: bool = True,
    add_bos_id: bool = True,
    add_eos_id: bool = True,
    return_tensors="jax",
    pad_to_max_length: bool = True,
    truncation: bool = True,
) -> jax.Array:
  """Encode text into token IDs using the tokenizer. Works with both HF and SentencePiece tokenizer."""
  return tokenizer(
      text if isinstance(text, list) else [text],  # required for HF
      truncation=True,  # Truncate to the model's max length
      return_tensors=return_tensors,  # Return NumPy arrays (works with JAX)
      pad_to_max_length=True,
      # padding_side="right",   # default setting in HF
      max_length=max_length,
      add_special_tokens=add_special_tokens,
      add_bos_id=add_bos_id,
      add_eos_id=add_eos_id,
  )["input_ids"]


def decode_tokens(tokens: jax.Array, tokenizer: Any, skip_special_tokens=True) -> list[str]:
  """Decode token IDs  with either a HF or SentencePiece tokenizer."""
  # tokens: (B, L)  int32/bf16 on device
  tokens_host = np.asarray(jax.device_get(tokens), dtype=np.int32)
  return tokenizer.batch_decode(tokens_host, skip_special_tokens=skip_special_tokens)


# --- Tokenization and Padding use by model code  ---
def process_and_pad_inputs(
    input_text: Any,
    chunk_length: int | None,
    cache_len: int | None,
    tokenizer: Any,
    add_bos_id: bool = True,
    add_eos_id: bool = True,
    return_tensors="jax",
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Tokenize and pad input text for the model using SentencePiece tokenizer.

  returning input ids, position ids, and attention mask.
  """

  def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

  # Use the encode text wrapper to handle both HF and SentencePiece tokenizers
  raw_ids = encode_text(
      input_text,
      tokenizer,
      max_length=chunk_length or None,
      add_special_tokens=True,
      add_bos_id=add_bos_id,
      add_eos_id=add_eos_id,
      return_tensors=return_tensors,
  )

  seq_len = raw_ids.shape[1]

  max_num_tokens = chunk_length or seq_len
  cache_length = cache_len or max_num_tokens
  assert cache_length >= max_num_tokens, "Cache length must be >= max_num_tokens."

  input_ids = jnp.pad(raw_ids, ((0, 0), (0, max_num_tokens - seq_len)), constant_values=PAD_ID)
  attn_mask = input_ids != PAD_ID

  # Build position ids using the Gemma 3 repository function
  position_ids = build_positions(attn_mask)

  causal_attn = jnp.tril(jnp.ones((max_num_tokens, max_num_tokens), dtype=bool))
  causal_attn = attn_mask[:, None, :] & causal_attn[None, :, :]
  causal_attn = jnp.pad(
      causal_attn,
      ((0, 0), (0, 0), (0, cache_length - max_num_tokens)),
      constant_values=0,
  )
  return input_ids, position_ids, causal_attn


def old_encode_raw_ids(
    input_text: Any,
    tokenizer: Any,
    add_special_tokens: bool = False,
    add_bos_id: bool = True,
    add_eos_id: bool = True,
) -> Any:

  import math

  def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

  # No EOS token, as it breaks generation
  raw_ids = encode_text(
      input_text,
      tokenizer,
      max_length=None,
      add_special_tokens=add_special_tokens,
      add_bos_id=add_bos_id,
      add_eos_id=add_eos_id,
  )

  batch_size, seq_len = raw_ids.shape

  position_ids = build_positions(raw_ids != PAD_ID)
  attn_mask = raw_ids != PAD_ID
  causal_attn = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
  causal_attn = attn_mask[:, None, :] & causal_attn[None, :, :]

  return raw_ids, position_ids, attn_mask, causal_attn


def encode_batch_to_multiple(
    input_text: Any,
    multiple: int,
    tokenizer: Any,
    add_bos_id: bool = True,
    add_eos_id: bool = True,
) -> Any:

  import math

  def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

  raw_ids, raw_position_ids, raw_attn_mask, raw_causal_attn = encode_raw_ids(
      input_text, tokenizer, add_bos_id=add_bos_id, add_eos_id=add_eos_id
  )

  batch_size, seq_len = raw_ids.shape
  num_chunks = math.ceil(seq_len / multiple)

  # Pad tokens to multiple of chunk_len so shapes stay static.
  pad_len = num_chunks * multiple - seq_len
  if pad_len > 0:
    input_ids = jnp.pad(raw_ids, ((0, 0), (0, pad_len)), constant_values=PAD_ID)
  else:
    input_ids = raw_ids

  position_ids = build_positions(input_ids != PAD_ID)
  max_seq_len = seq_len + pad_len

  attn_mask = input_ids != PAD_ID
  causal_attn = jnp.tril(jnp.ones((max_seq_len, max_seq_len), dtype=bool))
  causal_attn = attn_mask[:, None, :] & causal_attn[None, :, :]

  # return (raw_ids, raw_position_ids, raw_attn_mask, raw_causal_attn), (input_ids, num_chunks, pad_len, position_ids, attn_mask, causal_attn)
  return input_ids, num_chunks, pad_len, position_ids, attn_mask, causal_attn


def encode_raw_ids(
    input_text: Any,
    tokenizer: Any,
    add_bos_id: bool = True,
    add_eos_id: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Encode input_text using the raw SentencePiece tokenizer (tokenizer.sp).
  Returns: input_ids, position_ids, attention_mask, causal_attention_mask
  """
  # Handle single string or list of strings
  if isinstance(input_text, str):
    input_text = [input_text]

  # Encode each input using SentencePiece directly
  ids = []
  for t in input_text:
    seq = tokenizer.sp.EncodeAsIds(t)
    # If either add_bos_id or add_eos_id is explicitly set, ignore add_special_tokens
    if (add_bos_id is not True) or (add_eos_id is not True):
      if add_bos_id:
        seq = [tokenizer.bos_token_id] + seq
      if add_eos_id:
        seq = seq + [tokenizer.eos_token_id]
    else:
      # Default: add both BOS and EOS
      seq = [tokenizer.bos_token_id] + seq + [tokenizer.eos_token_id]
    ids.append(seq)

  # Pad to max length in batch
  max_len = max(len(seq) for seq in ids)
  input_ids = np.array([seq + [tokenizer.pad_token_id] * (max_len - len(seq)) for seq in ids], dtype=np.int32)
  input_ids = jnp.asarray(input_ids)

  # Attention mask: 1 for non-pad, 0 for pad
  attn_mask = (input_ids != tokenizer.pad_token_id).astype(jnp.int32)

  # Position ids: cumsum over attn_mask, minus 1 for each position (so padding gets -1)
  def build_positions(mask: jax.Array) -> jax.Array:
    pos = jnp.cumsum(mask, axis=-1)
    return pos - (pos >= 1)

  position_ids = build_positions(attn_mask)

  # Causal attention mask: (batch, seq, seq)
  causal_mask = jnp.tril(jnp.ones((max_len, max_len), dtype=bool))
  causal_attn = attn_mask[:, None, :] & causal_mask[None, :, :]

  return input_ids, position_ids, attn_mask, causal_attn

# %%
