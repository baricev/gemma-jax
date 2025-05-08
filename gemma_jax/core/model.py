"""Experimental model and inference code for text-only Gemma-3 written in a functional style in vanilla JAX."""

import os
import time
from functools import partial
from pathlib import Path
from typing import Any
from collections.abc import Callable

import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gemma_jax.core.cache import (KVCache, LayoutType, update_cache_layer, update_cache_layer_by_layer, update_cache_layers)
from gemma_jax.core.rope import (
    apply_rope,
    apply_rope_cached,
    apply_rope_outer_product,
)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# --- Constants ---
PAD_ID: int = 0
EOS_ID: int = 1
BOS_ID: int = 2
K_MASK: float = -2.3819763e38
ATTENTION_TYPE_GLOBAL: int = 1
ATTENTION_TYPE_LOCAL_SLIDING: int = 2
GEMMA3_ATTENTION_PATTERN: tuple[int, ...] = (2, 2, 2, 2, 2, 1)
PROMPT_TEMPLATE = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
ANSWER_TEMPLATE = "{}<end_of_turn>"


# --- Attention Mask (Sliding Window) ---
def _sliding_mask(
    position_ids: Array,  # (B, T) int32
    total_tokens: int,
    *,
    cache_len: int,
    window: int,
) -> Array:
  """
  Return a boolean mask with shape (B, T, cache_len) that is True for
  cache positions within W (window_size) tokens of each query token.
  """
  # 1. Re-create absolute positions that live in the ring-buffer cache.
  #    If total_tokens has not yet wrapped we can use a simple arange.
  cache_pos = jax.lax.cond(
      total_tokens <= cache_len,
      lambda _: jnp.arange(cache_len, dtype=jnp.int32),
      lambda _: (
          # rotated ring buffer: oldest token is (total tokens - cache_len)
          (jnp.arange(cache_len, dtype=jnp.int32) + total_tokens - cache_len)
      ),
      operand=None,
  )  # (cache_len,)

  # 2. Compute signed distance between each (query, key) pair.
  #    diff shape: (B, T, cache_len)
  diff = cache_pos[None, None, :] - position_ids[:, :, None]

  # 3. Inside window  we attend,  otherwise  mask out
  mask = (diff > -window) & (diff < window)
  return mask


# --- Attention Mask (auto-regressive stage) ---  # variants, equivalent/ for experimentation
def _build_attn_masks(
    time_step: int,
    seq_len: int,
    input_mask: Array,  # (B, seq_len) bool
) -> Array:
  """
  Produce (B, 1, seq_len) causal masks needed at each generation from input_mask:(B, seq_len) bool.
  Works entirely on device with broadcasting; no per step Python allocation.
  """
  # Broadcast compare:  shape (seq_len,)
  causal = jnp.arange(seq_len, dtype=jnp.int32) <= time_step

  # Combine with the per-example padding mask and give final shape
  return (causal[None, :] & input_mask).reshape(input_mask.shape[0], 1, seq_len)


def _compute_attn_mask(batch_size: int, seq_len: int, current_index: int) -> Array:
  """Produce (B, 1, seq_len) causal masks needed at each generation."""
  # Create sequence indices
  indices = jnp.arange(seq_len)

  # Generate mask: True for indices <= current_index, False otherwise
  mask_1d = indices <= current_index

  # Reshape and broadcast to target shape (batch_size, 1, seq_len)
  return jnp.broadcast_to(mask_1d.reshape(1, 1, seq_len), (batch_size, 1, seq_len))


# --- Core Model Functions ---
def decode(model: Any, x: Array) -> Array:
  """Decode hidden states back to logits."""
  return jnp.dot(x, model.input_embedding_table.T)


def rms_norm(x: Array, scale: Array, epsilon: float = 1e-6) -> Array:
  """Apply root mean square normalization to the input tensor."""
  var = jnp.mean(x * x, axis=-1, keepdims=True)
  return x * jax.lax.rsqrt(var + epsilon) * (1 + scale)


def encode(model: Any, x: Array, config: Any) -> Array:
  """Static-shape, gather-based embedding lookup (no advanced-indexing)."""
  emb = jnp.take(model.input_embedding_table, x, axis=0)  # (B,T,D)
  return emb * jnp.sqrt(jnp.asarray(config.embed_dim, emb.dtype))


def feed_forward(layer: Any, x: Array) -> Array:
  """Memory-lean gated-GELU FFN"""
  gate_out = jnp.einsum("...d,shd->...sh", x, layer.gating_weights)  # s=2
  act, gate = jnp.split(gate_out, 2, axis=-2)  # (...,1,H)
  hidden = jax.nn.gelu(act, approximate=True) * gate
  return jnp.einsum("...h,hd->...d", hidden.squeeze(-2), layer.output_weights)


def is_global_layer(layer_idx: int) -> bool:  # Gemma-3 pattern
  return GEMMA3_ATTENTION_PATTERN[layer_idx % len(GEMMA3_ATTENTION_PATTERN)] == 1


def self_attention(
    tokens: Array,  # no op
    x: Array,
    positions: Array,
    attn_mask: Array,
    write_index: int,
    layer_idx: int,
    layer: Any,
    cache: KVCache,
    rope_cache: Any,  # dummy value [for benchmarking RoPE implementations]
    config: Any,
    layout: LayoutType,
) -> tuple[jax.Array, Any]:

  query = jnp.einsum("btd,ndh->btnh", x, layer.q_proj)
  key, value = jnp.einsum("bsd,ckdh->cbskh", x, layer.kv_proj)
  query = rms_norm(query, layer.attn_query_norm_scale)
  key = rms_norm(key, layer.attn_key_norm_scale)

  attn_type = ATTENTION_TYPE_GLOBAL if is_global_layer(layer_idx) else ATTENTION_TYPE_LOCAL_SLIDING
  rope_idx = 0
  base_frequency = config.local_base_frequency
  scale_factor = config.local_scale_factor

  if attn_type == ATTENTION_TYPE_GLOBAL:
    base_frequecy = config.global_base_frequency  # 1_000_000
    scale_factor = config.global_scale_factor  # 8.0

  if rope_cache is not None:
    query = apply_rope_cached(query, positions, rope_cache[rope_idx][0], rope_cache[rope_idx][1])
    key = apply_rope_cached(key, positions, rope_cache[rope_idx][0], rope_cache[rope_idx][1])

  else:
    query = apply_rope(
        query,
        positions,
        base_frequency=base_frequency,
        scale_factor=scale_factor,
    )
    key = apply_rope(key, positions, base_frequency=base_frequency, scale_factor=scale_factor)

  if attn_type == ATTENTION_TYPE_LOCAL_SLIDING:
    total = write_index + config.padded_input_size
    attn_mask = attn_mask & _sliding_mask(
        positions,
        total,
        cache_len=config.cache_length,
        window=config.window_size,
    )

  query_scaled = query * config.query_pre_attn_scalar
  B, T, N, H = query_scaled.shape
  G = N // config.num_kv_heads
  query_scaled = query_scaled.reshape((B, T, config.num_kv_heads, G, H))

  cache_key, cache_value, cache = update_cache_layer(
      cache,
      key,
      value,
      write_index=write_index,
      layer=layer_idx,
      kind=layout,
  )

  cache_key = cache_key.astype(jnp.float32)
  cache_value = cache_value.astype(jnp.float32)

  logits = jnp.einsum("btkgh,bskh->btkgs", query_scaled, cache_key)
  logits = logits.reshape((B, T, -1, logits.shape[-1]))

  padded_logits = jnp.where(jnp.expand_dims(attn_mask, -2), logits, K_MASK)
  attn_weights = jax.nn.softmax(padded_logits, axis=-1)
  probs = attn_weights.reshape((B, T, config.num_kv_heads, G, logits.shape[-1]))
  attn_out = jnp.einsum("btkgs,bskh->btkgh", probs, cache_value)
  attn_out = attn_out.reshape((B, T, N, H))

  # Downcast output to match input dtype (bfloat16 or int8)
  attn_out = attn_out.astype(x.dtype)
  output = jnp.einsum("btnh,nhd->btd", attn_out, layer.output_proj)
  return output, cache


@partial(jax.jit, static_argnames=("config", "layout"))
def forward_fn(
    tokens: jax.Array,
    positions: jax.Array,
    attn_mask: jax.Array,
    write_index: int,
    model: Any,
    cache: Any,
    rope_cache: Any,  # dummy value [for benchmarking RoPE implementations]
    config: Any,
    layout: LayoutType,
) -> tuple[jax.Array, Any]:
  x = encode(model, tokens, config)

  for idx, layer in enumerate(model.blocks):
    norm_x = rms_norm(x, layer.pre_attention_norm_scale)
    attn_out, cache = self_attention(
        tokens,
        norm_x,
        positions,
        attn_mask,
        write_index,
        idx,
        layer,
        cache,
        rope_cache,
        config,
        layout,
    )
    x = x + rms_norm(attn_out, layer.post_attention_norm_scale)
    norm_x = rms_norm(x, layer.pre_ffw_norm_scale)
    x = x + rms_norm(feed_forward(layer, norm_x), layer.post_ffw_norm_scale)
  x = rms_norm(x, model.final_norm_scale)
  logits = decode(model, x)
  return logits, cache


# --- Auto-regressive generation ---
def setup_scan_fn(
    input_ids: Array,
    position_ids: Array,
    next_tokens: Array,
    prefill_cache: Any,
    cache_length: int,
) -> tuple[Array, int, Array, tuple[Any, ...]]:
  """Set up the initial state for scan-based generation."""
  batch_size = input_ids.shape[0]

  current_index = int(jnp.max(position_ids[:, -1]))
  all_generated = jnp.zeros((batch_size, cache_length), dtype=jnp.int32)
  all_generated = all_generated.at[:, : input_ids.shape[1]].set(input_ids)
  all_generated = all_generated.at[:, current_index + 1].set(next_tokens.reshape(-1))
  current_pos = (position_ids[:, -1] + 1).reshape(-1, 1)

  initial_carry = (
      all_generated,
      current_index + 1,
      current_pos,
      prefill_cache,
  )
  return all_generated, current_index, current_pos, initial_carry


def scan_generate_step(
    carry: tuple[Any, ...],
    _,
    *,
    model: Any,
    rope_cache: Any,
    config: Any,
    layout: LayoutType,
) -> tuple[tuple[Any, ...], Array]:
  """
  'carry' = (tokens, cur_idx, next_pos, cache)

  Produces the next token and updated state; **no host ops** inside.
  """
  tokens, cur_idx, cur_pos, cache = carry
  B = tokens.shape[0]
  cache_length = config.cache_length
  cur_tok = jax.lax.dynamic_slice(tokens, (0, cur_idx), (B, 1))

  attn_mask = _build_attn_masks(
      cur_idx,
      cache_length,
      jnp.ones((B, cache_length), dtype=jnp.bool_),
  )

  logits, new_cache = forward_fn(
      cur_tok,
      cur_pos,
      attn_mask,
      cur_idx,
      model,
      cache,
      rope_cache,
      config,
      layout,
  )
  next_token = jnp.argmax(logits[:, 0, :], axis=-1)
  tokens = jax.lax.dynamic_update_slice(tokens, next_token.reshape(-1, 1), (0, cur_idx + 1))

  return (
      tokens,
      cur_idx + 1,
      (cur_pos + 1).reshape(-1, 1),
      new_cache,
  ), next_token
