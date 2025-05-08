"""Multi-turn batched conversation state management and response generation.

This module provides a bare-bones implementation of a parallel, multi-turn conversation
state management system using JAX.
Conversation turns are stored in a batched state, and the model's responses
are generated using various sampling strategies (greedy, top-k, top-p).
"""

import os
import time
from typing import Any, Optional, Callable, Union

import jax
import jax.numpy as jnp
from jax.random import categorical
from jax import Array
from flax import struct
from dataclasses import replace

from gemma_jax.core.sp_tokenizer import (
    encode_text,
    decode_tokens,
    process_and_pad_inputs,
    format_prompt,
    format_answer,
    PAD_ID,
    EOS_ID,
    BOS_ID,
)

from gemma_jax.core.inference import (
    greedy_sample,
    greedy_sample_single_step,
    sample_temperature,
    sample_top_k,
    sample_top_p,
)

# --- Role IDs ---
ROLE_USER, ROLE_MODEL, ROLE_SYSTEM = 0, 1, 2
ROLE_STR = {ROLE_USER: "user", ROLE_MODEL: "model", ROLE_SYSTEM: "system"}

# --- Log of conversation turns ---
conversation_log: list[dict[str, Any]] = []


# --- Model Response Functions ---
def extract_response(generated_text: str, *args: Any, **kwargs: Any) -> Any:
  """Extract the model's response from the generated text."""
  start = generated_text.rfind("<start_of_turn>model")
  if start != -1:
    response = generated_text[start + len("<start_of_turn>model") :]
    end = response.find("<end_of_turn>")
    if end != -1:
      response = response[:end]
    response = response.strip()
  else:
    response = generated_text

  # Check if the response contains <thinking> tags or other metatokens
  # Handle thinking sections
  thinking_content = ""
  system_content = ""
  visible_content = response

  if "<thinking>" in response and "</thinking>" in response:
    thinking_start = response.find("<thinking>")
    thinking_end = response.find("</thinking>")

    if thinking_start != -1 and thinking_end != -1 and thinking_end > thinking_start:
      thinking_content = response[thinking_start + len("<thinking>") : thinking_end].strip()

      # Create visible content without the thinking section
      visible_content = (response[:thinking_start] + response[thinking_end + len("</thinking>") :]).strip()

    return (
        response.strip(),
        visible_content.strip(),
        thinking_content.strip(),
    )

  if "<system>" in response and "</system>" in response:
    system_start = response.find("<system>")
    system_end = response.find("</system>")

    if system_start != -1 and system_end != -1 and system_end > system_start:
      system_content = response[system_start + len("<system>") : system_end].strip()

      # Create visible content without the system section
      visible_content = (response[:system_start] + response[system_end + len("</system>") :]).strip()

      return (
          response.strip(),
          visible_content.strip(),
          system_content.strip(),
      )

  # If no thinking section is found, return the response as is
  return response.strip()


# --- Batched Conversation State Management using JAX Structs ---
@struct.dataclass
class ConversationState:
  """Batched, JIT friendly conversation buffer.

  Args:
      tokens   : (B, max_tok)    flat token ids
      offsets  : (B, max_turn+1) start index of each turn
      roles    : (B, max_turn)   role id per turn
      tok_ptr  : (B,)            write pointer into tokens
      turn_ptr : (B,)            write pointer into offsets / roles
      trace    : optional parallel buffer (same shape as tokens)
      max_tok  : int           max number of tokens in buffer
      max_turn : int           max number of turns in buffer

  """

  tokens: jnp.ndarray
  offsets: jnp.ndarray
  roles: jnp.ndarray
  tok_ptr: jnp.ndarray  # (B,)
  turn_ptr: jnp.ndarray  # (B,)
  max_tok: int
  max_turn: int
  trace: Optional[jnp.ndarray] = None  # (B, max_tok) or None

  @property
  def n_tokens(self: Any) -> Any:
    return self.tok_ptr  # Per conversation vector.

  @property
  def n_turns(self: Any) -> Any:
    return self.turn_ptr  # Per turn vector.


def create_empty_state_batched(
    batch_size: int,
    cache_length: int,
    max_turns: int,
    with_trace: bool = False,
    pad_id: int = 0,
) -> ConversationState:
  """Create an empty batched conversation state."""
  batch_size = batch_size
  cache_length = cache_length
  max_turns = max_turns

  tok_buf = jnp.full((batch_size, cache_length), pad_id, jnp.int32)
  off_buf = jnp.zeros((batch_size, max_turns + 1), jnp.int32)
  role_buf = jnp.zeros((batch_size, max_turns), jnp.int8)
  trace_buf = jnp.full_like(tok_buf, pad_id) if with_trace else None
  zeros = jnp.zeros((batch_size,), jnp.int32)

  return ConversationState(
      tok_buf,
      off_buf,
      role_buf,
      zeros,
      zeros,
      cache_length,
      max_turns,
      trace_buf,
  )  # type: ignore[call-arg]


def _append_tokens(
    state: Any,
    new_tokens: jnp.ndarray,  # (B, L) already padded with PAD_ID
    role: jnp.ndarray,  # (B,)  int8
    trace_tok: Optional[jnp.ndarray] = None,  # (B, max_tok) or None
) -> Any:
  """Append new tokens to the conversation state."""
  B, L = new_tokens.shape
  assert B == state.tokens.shape[0], "batch size mismatch"

  # Bounds check (vectorised)
  if not jnp.all(state.tok_ptr + L <= state.max_tok):
    raise AssertionError("Token buffer exhausted")
  if not jnp.all(state.turn_ptr + 1 <= state.max_turn):
    raise AssertionError("Turn buffer exhausted")

  # write tokens
  tok_upd = jax.vmap(lambda buf, ptr, toks: jax.lax.dynamic_update_slice(buf, toks, (ptr,)))(
      state.tokens, state.tok_ptr, new_tokens
  )

  trace_upd = state.trace
  if trace_tok and state.trace:
    trace_upd = jax.vmap(lambda buf, ptr, toks: jax.lax.dynamic_update_slice(buf, toks, (ptr,)))(
        state.trace, state.tok_ptr, trace_tok
    )

  # update offsets and roles
  off_upd = state.offsets.at[jnp.arange(B), state.turn_ptr + 1].set(state.tok_ptr + L)
  roles_upd = state.roles.at[jnp.arange(B), state.turn_ptr].set(role)

  return replace(
      state,
      tokens=tok_upd,
      offsets=off_upd,
      roles=roles_upd,
      trace=trace_upd,
      tok_ptr=state.tok_ptr + L,
      turn_ptr=state.turn_ptr + 1,
  )


def _add_turn_batched(
    state: ConversationState,
    role: int,
    texts: list[str],  # length = B
    tokenizer: Any,
) -> ConversationState:
  """Add a new turn to the batched conversation state."""
  assert len(texts) == state.tokens.shape[0], "batch length mismatch"
  if role == ROLE_USER:
    texts = [format_prompt(t) for t in texts]
  elif role == ROLE_MODEL:
    texts = [format_answer(t) for t in texts]

  tok = encode_text(texts, tokenizer)
  # Ensure common length L
  max_len = tok.shape[1]
  # Pad manually to fixed L so vmap writing works
  if max_len < state.max_tok:
    tok = jnp.pad(tok, ((0, 0), (0, state.max_tok - max_len)), constant_values=0)
    tok = tok[:, :max_len]  # keep actual length
  roles = jnp.full((len(texts),), role, jnp.int8)

  return _append_tokens(state, tok, roles)


# --- Chat Function ---
def chat(
    prefill_partial: Callable,
    process_inputs_partial: Callable,
    setup_scan_fn: Callable,
    scan_generate_partial_fn: Callable,
    tokenizer: Any,
    config: Any,
    state: ConversationState,
    user_texts: list[str],
    generate_steps: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    role: int = ROLE_USER,
    skip_special_tokens: bool = True,
    **kw: Any,
) -> tuple[ConversationState, list[str]]:
  """Run one user -> model exchange on *B* conversations in parallel. Returns (new_state, list[str] replies)."""
  B = len(user_texts)
  assert B == state.tokens.shape[0], "batch size mismatch"

  assert generate_steps <= state.max_tok, "num of generate is larger than max_tok (CACHE_LENGTH)"
  if generate_steps != config.num_new_tokens:
    print(
        f"Warning: generate_steps is set to {generate_steps}, ",
        f"but config.num_new_tokens is {config.num_new_tokens}.",
    )

  state = _add_turn_batched(state, role, user_texts, tokenizer)
  # Append to host log
  conversation_log.append({"role": "user" if role == ROLE_USER else "system", "text": user_texts})

  # Apply Gemma3 dialogue template to prompts
  user_texts = [format_prompt(t) for t in user_texts]

  # Build prompt tensors once
  prompt_ids, pos_ids, attn_mask = process_inputs_partial(user_texts)  # shapes (B, S)

  # Prefill
  logits, kv_cache = prefill_partial(prompt_ids, pos_ids, attn_mask)

  # Sample tokens
  logits_for_sampling = logits[:, -1:, :]  # (B, 1, V)
  rng = jax.random.key(int(time.time() * 1e6))
  subkeys = jax.random.split(rng, B)

  def sample_one(k: Any, l: Any) -> Any:
    # jax.vmap: needs a leading dimension as sampling functions expect logits.ndim == 3
    l = l.reshape(1, 1, -1)

    if temperature <= 0.0:
      return greedy_sample_single_step(l)[0]
    if 0.0 < top_p <= 1.0:
      return sample_top_p(k, l, p=top_p, temperature=temperature)[0]
    if top_k > 0:
      return sample_top_k(k, l, k=top_k, temperature=temperature)[0]
    return sample_temperature(k, l, temperature=temperature)[0]

  next_tok = jax.vmap(sample_one)(subkeys, logits_for_sampling).reshape(B, 1)

  _, _, _, carry = setup_scan_fn(
      prompt_ids,
      pos_ids,
      next_tok,
      kv_cache,
      cache_length=config.cache_length,
  )

  # Generate step
  new_carry, _ = jax.lax.scan(scan_generate_partial_fn, carry, None, length=generate_steps)

  full_ids = new_carry[0]  # (B, T)

  # (4) Decode
  replies = decode_tokens(full_ids, tokenizer, skip_special_tokens=skip_special_tokens)
  replies_clean = [extract_response(r) for r in replies]

  # (5) Append model turn
  state = _add_turn_batched(state, ROLE_MODEL, replies_clean, tokenizer)
  conversation_log.append({"role": "model", "text": replies_clean})

  return state, replies_clean


# --- Serialization ---
def _state_to_text(state: ConversationState, tokenizer: Any, strip_special: bool = True) -> str:
  """Convert the conversation state to a readable text format."""
  toks = jax.device_get(state.tokens[: state.n_tokens])
  txt = decode_tokens(toks, tokenizer)

  pieces = []
  for i in range(state.n_turns):
    beg, end = state.offsets[i], state.offsets[i + 1]
    role = ROLE_STR[int(state.roles[i])]
    pieces.append(f"<{role}>\n" + "".join(txt[beg:end]))
  return "\n".join(pieces)


def _save_state(state: ConversationState, path: str) -> None:
  """Save the conversation state to a file."""
  host = jax.tree_util.tree_map(lambda x: jax.device_get(x), state)
  jnp.savez(
      path,
      tokens=host.tokens,
      offsets=host.offsets,
      roles=host.roles,
      tok_ptr=host.tok_ptr,
      turn_ptr=host.turn_ptr,
      max_tok=state.max_tok,
      max_turn=state.max_turn,
      trace=host.trace if host.trace is not None else [],
  )


def _load_state(path: str) -> ConversationState:
  """Load a conversation state from a file."""
  npz = jnp.load(path, allow_pickle=True)
  return ConversationState(
      tokens=npz["tokens"],
      offsets=npz["offsets"],
      roles=npz["roles"],
      tok_ptr=npz["tok_ptr"],
      turn_ptr=npz["turn_ptr"],
      max_tok=int(npz["max_tok"]),
      max_turn=int(npz["max_turn"]),
      trace=npz["trace"] if npz["trace"].size else None,
  )  # type: ignore[call-arg]


def _state_to_text_batched(state: ConversationState, tokenizer: Any, strip_special: bool = True) -> list[str]:
  """Convert the batched conversation state to readable text format."""
  outs = []
  B = state.tokens.shape[0]
  host = jax.device_get(state)
  for b in range(B):
    pieces = []
    toks = host.tokens[b, : host.tok_ptr[b]]
    txt = decode_tokens(toks, tokenizer)

    for i in range(host.turn_ptr[b]):
      beg, end = host.offsets[b, i], host.offsets[b, i + 1]
      role = ROLE_STR[int(host.roles[b, i])]
      pieces.append(f"<{role}>\n" + "".join(txt[beg:end]))
    outs.append("\n".join(pieces))
  return outs
