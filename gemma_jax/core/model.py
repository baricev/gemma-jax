# %%

"""Experimental model and inference code for text-only Gemma-3 written in a functional style in vanilla JAX."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

import dataclasses
import enum
from functools import partial
from typing import Any, NamedTuple

from gemma_jax.core.cache import (KVCache, update_cache_layer)
from gemma_jax.core.rope import ( apply_rope_cached,)

# -----------------------------------------------------------------------------
# Transformer class
# -----------------------------------------------------------------------------

class Layer(NamedTuple):
    attn_key_norm_scale       : Array       # (head_dim,)                             (H,)
    attn_query_norm_scale     : Array       # (head_dim,)                             (H,)
    output_proj               : Array       # (num_heads, head_dim, embed_dim)        (N,H,D)
    kv_proj                   : Array       # (2, num_kv_heads, embed_dim, head_dim)  (2,K,D,H)
    q_proj                    : Array       # (num_heads, embed_dim, head_dim)        (N,D,H)
    gating_weights            : Array       # (2, mlp_hidden_dim, embed_dim)          (2,F,D)
    output_weights            : Array       # (mlp_hidden_dim, embed_dim)             (F,D)
    post_attention_norm_scale : Array       # (embed_dim,)                            (D,)
    post_ffw_norm_scale       : Array       # (embed_dim,)                            (D,)
    pre_attention_norm_scale  : Array       # (embed_dim,)                            (D,)
    pre_ffw_norm_scale        : Array       # (embed_dim,)                            (D,)


class Gemma3(NamedTuple):
    input_embedding_table     : Array       # (vocab_size, embed_dim)                 (V,D)
    mm_input_projection       : Array       # (embed_dim, embed_dim)                  (ViT_embed_dim, D)
    mm_soft_embedding_norm    : Array       # (embed_dim,)                            (ViT_embed_dim,)
    final_norm_scale           : Array       # (embed_dim,)                            (D,)
    blocks                    : tuple[Layer, ...]
 

def apply_rope(
    inputs: Array,
    positions: Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> Array:
  """Applies RoPE.

  Let B denote batch size, L denote sequence length, N denote number of heads,
  and H denote head dimension. Note that H must be divisible by 2.

  Args:
    inputs: Array of shape [B, L, N, H].
    positions:  Array of shape [B, L].
    base_frequency: Base frequency used to compute rotations.
    scale_factor: The scale factor used for positional interpolation, allowing
      an expansion of sequence length beyond the pre-trained context length.

  Returns:
    Array of shape [B, L, N, H].
  """
  head_dim = inputs.shape[-1]
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = base_frequency**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)



"""JAX functional implementations of transformer sub-modules."""

K_MASK = -2.3819763e38  # Large negative number for masking logits.
DEFAULT_ROPE_BASE_FREQUENCY = 10_000
DEFAULT_ROPE_SCALE_FACTOR = 1.0

ROPE_INDEX_LOCAL = 0
ROPE_INDEX_GLOBAL = 1

State = Any


class AttentionType(enum.Enum):
  """Type of attention used by a block."""

  GLOBAL = 1
  LOCAL_SLIDING = 2

ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)

def make_attention_layers_types(
    num_layers: int,
    pattern: tuple[AttentionType, ...]=ATTENTION_PATTERN,
) -> tuple[int, ...]:
  """Returns the list of attention types for every layers."""
  pattern_size = len(pattern)
  out = pattern * (num_layers // pattern_size)
  if num_layers % pattern_size != 0:
    out += pattern[: num_layers % pattern_size]
  return tuple(out)

# -----------------------------------------------------------------------------
# Helper dataclasses for parameters
# -----------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class AttentionConfig:
  num_heads: int
  num_kv_heads: int
  embed_dim: int
  head_dim: int
  hidden_dim: int
  attn_type: AttentionType
  query_pre_attn_scalar: float
  rope_base_frequency: int = DEFAULT_ROPE_BASE_FREQUENCY
  rope_scale_factor: float = DEFAULT_ROPE_SCALE_FACTOR
  window_size: int | None = None
  cache_length: int = 1024


def _layer_config(config: Any, attn_type: int) -> AttentionConfig:
  return AttentionConfig(
      num_heads=config.num_heads,
      num_kv_heads=config.num_kv_heads,
      embed_dim=config.embed_dim,
      head_dim=config.head_dim,
      hidden_dim=config.hidden_dim,
      attn_type=attn_type,
      query_pre_attn_scalar=config.query_pre_attn_scalar,
      rope_base_frequency=(
          config.local_base_frequency
          if attn_type == AttentionType.LOCAL_SLIDING
          else config.global_base_frequency
      ),
      rope_scale_factor=(
          config.local_scale_factor
          if attn_type == AttentionType.LOCAL_SLIDING
          else config.global_scale_factor
      ),
      window_size=(
          config.window_size if attn_type == AttentionType.LOCAL_SLIDING else None
      ),
      cache_length=config.cache_length,
  )


# -----------------------------------------------------------------------------
# Normalisation and basic ops
# -----------------------------------------------------------------------------
def rms_norm(x: Array, scale: Array, epsilon: float = 1e-6) -> Array:
  var = jnp.mean(x * x, axis=-1, keepdims=True)
  return x * jax.lax.rsqrt(var + epsilon) * (1 + scale)

def rms_norm(x: Array, scale: Array, epsilon: float = 1e-6, dtype: jnp.dtype = jnp.bfloat16) -> Array:
  """Upcast to float32, compute RMS, then downcast to jnp.bfloat16."""
  x = x.astype(jnp.float32)
  var = jnp.mean(x * x, axis=-1, keepdims=True)
  x = x * jax.lax.rsqrt(var + epsilon) * (1 + scale.astype(x.astype(jnp.float32)))
  return x.astype(dtype)


# -----------------------------------------------------------------------------
# Embedder
# -----------------------------------------------------------------------------

def encode(model: Any, x: Array, config: Any) -> Array:
  """Static-shape, gather-based embedding lookup (no advanced-indexing)."""
  emb = jnp.take(model.input_embedding_table, x, axis=0)  # (B,T,D)
  return emb * jnp.sqrt(jnp.asarray(config.embed_dim, emb.dtype))

def decode(model: Any, x: Array) -> Array:
  """Decode hidden states back to logits."""
  return jnp.dot(x, model.input_embedding_table.T) # (B,T,D) -> (B,T,V)

def embedder_encode_vision(params: Any, x: jax.Array) -> jax.Array:
  """Project vision embeddings into the text embedding space."""
  if params.mm_soft_embedding_norm_scale is None:
    raise ValueError('Vision encoder not configured.')
  x = rms_norm(x, params.mm_soft_embedding_norm_scale)
  x = jnp.einsum('...tm,md->...td', x, params.mm_input_projection)
  return x


# -----------------------------------------------------------------------------
# Feed forward
# -----------------------------------------------------------------------------

def feed_forward(model: Any, x: Array) -> Array:
  """Memory-lean gated-GELU FFN"""
  gate_out = jnp.einsum("btd, gfd->btgf", x, model.gating_weights)  # g=2
  act, gate = jnp.split(gate_out, 2, axis=-2)  # (...,1,F)   BTGF -> [BT1F, BT1F]
  hidden = jax.nn.gelu(act, approximate=True) * gate
  return jnp.einsum("btf,fd->btd", hidden.squeeze(-2), model.output_weights)

# -----------------------------------------------------------------------------
# Attention helpers
# -----------------------------------------------------------------------------

def _new_apply_rope(x: Array, pos: Array, *, base: int, scale: float) -> Array:
  H = x.shape[-1]
  half = H // 2
  freq = base ** (jnp.arange(half) * 2 / H)
  theta = pos[..., None] / freq / scale
  sin, cos = jnp.sin(theta), jnp.cos(theta)
  sin, cos = sin[..., None, :], cos[..., None, :]
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _create_sliding_mask(
    segment_pos: jnp.ndarray,
    end_index: int,
    cache_len: int,
    sliding_window_size: int,
) -> jax.Array:
  """Mask for sliding window attention."""
  total_tokens = end_index + segment_pos.shape[1]

  def _reconstruct_rotated_cache_positions():
    cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
    cache_positions = (
        jnp.zeros_like(cache_positions)
        .at[cache_positions % cache_len]
        .set(cache_positions)
    )
    return cache_positions

  cache_positions = jax.lax.cond(
      total_tokens <= cache_len,
      lambda: jnp.arange(cache_len),
      _reconstruct_rotated_cache_positions,
  )

  cache_positions = cache_positions[None, None, :]
  segment_pos = segment_pos[:, :, None]
  mask = cache_positions > segment_pos - sliding_window_size
  mask *= cache_positions < segment_pos + sliding_window_size
  return mask

@partial(jax.jit, static_argnames=("window",))
def _extract_window(mat: Array, end_idx: Array, *, window: int) -> Array:
  def _one(m, e):
    start = jnp.maximum(e - window, 0)
    return jax.lax.dynamic_slice_in_dim(m, start, window, axis=0)
  return jax.vmap(_one)(mat, end_idx)


# -----------------------------------------------------------------------------
# Attention masks
# -----------------------------------------------------------------------------

def build_sliding_mask(
    position_ids: Array,  # (B, T) int32
    total_tokens: int,
    *,
    cache_len: int,
    # window_size: int,
    sliding_window_size: int,
) -> Array:
  """
  Return a boolean mask with shape (B, T, cache_len) that is True for
  cache positions within W (window_size) tokens of each query token.
  """
  window = sliding_window_size

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


def build_gen_step_attn_masks(
    time_step: int | Array, # (), or a batched 2D array of time steps (B,1)   int32
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


def build_combined_attn_mask(attn_mask: Array, attn_type: AttentionType, positions: Array, seq_lens: Array, config: AttentionConfig) -> Array:
  attn_mask_BTS = attn_mask
  if attn_type == AttentionType.LOCAL_SLIDING:
    total = jnp.max(seq_lens)
    sliding_mask_BTS = build_sliding_mask(
        positions,
        total,
        cache_len=config.cache_length,
        sliding_window_size=config.window_size,
    )
    attn_mask_BTS = attn_mask_BTS & sliding_mask_BTS

  return attn_mask_BTS

# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------

def qkv_projection(x: Array, q_proj: Array, kv_proj: Array) -> tuple[Array, Array]:
  query = jnp.einsum("btd,ndh->btnh", x, q_proj)
  key, value = jnp.einsum("bsd,ckdh->cbskh", x, kv_proj)
  return query, key, value

def output_projection(x: Array, output_proj: Array) -> Array:
  return jnp.einsum("btnh,nhd->btd", x, output_proj)

@jax.jit
def multi_head_attention(
    q: Array,                      # (B, T, N, H) query
    k: Array,                      # (B, S, K, H) key
    v: Array,                      # (B, S, K, H) value
    attn_mask_BTS: Array,          # (B, T, S) attention mask
) -> Array:
  """
  Compute multi-head attention with scaled dot-product attention.

  Returns the output of the attention layer (B, T, N, H)
  """
  B, T, N, H = q.shape
  _, S, K, _ = k.shape
  G = N // K
  
  q = q.reshape((B, T, K, G, H))
  scores = jnp.einsum("btkgh,bskh->btkgs", q, k)
  scores = scores.reshape((B, T, -1, S))

  attn_weights = jax.nn.softmax(
    jnp.where(jnp.expand_dims(attn_mask_BTS, -2), scores, K_MASK).astype(jnp.float32),
    axis=-1
  ).astype(jnp.bfloat16)


  probs = attn_weights.reshape((B, T, K, G, S))

  attn_out = jnp.einsum("btkgs,bskh->btkgh", probs, v)
  attn_out = attn_out.reshape((B, T, N, H))

  return attn_out

@partial(jax.jit, static_argnums=(9, 10, 11))
def self_attention(
    state: State,           # no-op for now
    x: Array,
    seq_lens: Array,

    positions: Array,
    attn_mask_BTS: Array,

    write_index: Array,
    layer: Any,
    cache: KVCache,
    rope_cache: Any,        # optional [used for benchmarking RoPE implementations]
    config: AttentionConfig,
    auto_regressive: bool,  # no-op for now
    layer_idx: int,
) -> tuple[jax.Array, Any]:

  query, key, value = qkv_projection(x, layer.q_proj, layer.kv_proj)

  query = rms_norm(query, layer.attn_query_norm_scale)
  key = rms_norm(key, layer.attn_key_norm_scale)

  attn_type = config.attn_type
  base_frequency = config.rope_base_frequency
  scale_factor = config.rope_scale_factor

  if rope_cache is not None:
    rope_idx = ROPE_INDEX_GLOBAL if attn_type == AttentionType.GLOBAL else ROPE_INDEX_LOCAL

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

  cache_key, cache_value, cache = update_cache_layer(
      cache,
      key,
      value,
      write_pos_B=write_index,
      seq_lens_B=seq_lens,
      layer=layer_idx,
  )


  query_scaled = query * config.query_pre_attn_scalar

  attn_out = multi_head_attention(
      query_scaled.astype(jnp.float32),  # Ensure float32 for precision
      cache_key.astype(jnp.float32),  
      cache_value.astype(jnp.float32), 
      attn_mask_BTS,
  ).astype(x.dtype)

  attn_out = output_projection(attn_out, layer.output_proj)

  return attn_out, cache


@partial(jax.jit, static_argnums=(9, 10))
def forward_fn(
    state,                # no-op for now
    input_ids,
    seq_lens,

    positions,
    attn_mask,

    write_index,

    model,
    cache,
    rope_cache,            # optional 
    config,
    auto_regressive=False, # no-op for now
):
  # 
  x = encode(model, input_ids, config)

  # Use bfloat16 for computation
  x = x.astype(jnp.bfloat16)

  attention_types = make_attention_layers_types( num_layers=config.num_layers)


  for idx, layer in enumerate(model.blocks):

    layer_cfg = _layer_config(config, attention_types[idx])
    attn_mask = build_combined_attn_mask(attn_mask, layer_cfg.attn_type, positions, seq_lens, layer_cfg)

    norm_x = rms_norm(x, layer.pre_attention_norm_scale)


    attn_out, cache = self_attention(
        state,
        norm_x,
        seq_lens,

        positions,
        attn_mask,

        write_index,

        layer,
        cache,
        rope_cache,
        layer_cfg,
        auto_regressive,
        idx,
    )
    x = x + rms_norm(attn_out, layer.post_attention_norm_scale)
    norm_x = rms_norm(x, layer.pre_ffw_norm_scale)
    x = x + rms_norm(feed_forward(layer, norm_x), layer.post_ffw_norm_scale)

  x = rms_norm(x, model.final_norm_scale) # (batch_size, seq_len, embed_dim)
  return x , cache


@jax.jit
def setup_scan_fn(state, input_ids, prefill_cache):
    batch_size = input_ids.shape[0]

    def build_positions(mask):                # (B,T) → positions
        pos = jnp.cumsum(mask, -1)
        return pos - (pos >= 1)

    prompt_pos = build_positions(input_ids != 0)
    last_pos    = prompt_pos[:, -1]           # (B,)
    seq_lens_B  = (input_ids != 0).sum(-1)
    last_tokens = input_ids[jnp.arange(batch_size), last_pos]  # (B,)

    carry = (
        last_tokens,         # current token (B,)
        seq_lens_B,                                          # current seq-lens (B,)
        last_pos + 1,                                        # write indices (B,)
        last_pos[:, None],                                   # absolute pos  (B,1)
        0,                               # step counter (scalar)
        prefill_cache,
        state,
    )
    return carry

@partial(jax.jit, static_argnames=( "config"))
def scan_generate_step(carry, _, *, model, rope_cache, config):
    (current_tok_B, seq_len_B, write_idx_B, abs_pos_B1,
     step, kv_cache, model_state) = carry

    batch   = current_tok_B.shape[0]
    cache_L = config.cache_length

    attn_mask = build_gen_step_attn_masks(
        write_idx_B[:, None], cache_L,
        jnp.ones((batch, cache_L), dtype=jnp.bool_)
    )

    x_emb, kv_cache = forward_fn(
        model_state,
        current_tok_B[:, None],        # (B,1)
        seq_len_B,
        abs_pos_B1,
        attn_mask,
        write_index=write_idx_B,
        model=model,
        cache=kv_cache,
        rope_cache=rope_cache,
        config=config,
        auto_regressive=True,
    )

    logits_BV = decode(model, x_emb)[:, 0]   # (B,V)
    next_B    = jnp.argmax(logits_BV, -1)    # sample on device


    new_carry = (
        next_B,
        seq_len_B + 1,
        (write_idx_B + 1) % cache_L,
        abs_pos_B1 + 1,
        step + 1,
        kv_cache,
        model_state,
    )
    return new_carry, next_B                 # ← return the token, not a big buffer@partial(jax.jit, static_argnames=("config",))
# %%
