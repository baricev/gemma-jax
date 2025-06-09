# %%
"""
KV-cache utilities for Gemma‑3 in functional JAX style (fixed - no ring buffer).
"""

from dataclasses import dataclass
from typing import Any, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.sharding import NamedSharding, PartitionSpec

# ============================================================================
# KVCache Data Structure
# ============================================================================

@jax.tree_util.register_pytree_node_class
@dataclass
class KVCache:
    """Sharded key/value tensors for transformer attention.

    Attributes:
      key: Cached keys, shape (num_layers, batch, max_seq_len, num_kv_heads, head_dim)
      value: Cached values, same shape as key
      sequence_lengths: Valid sequence length per batch element, shape (batch,)
      write_positions: Next write position per batch element, shape (batch,)
    """

    key: Array  # (L, B, S, K, H)
    value: Array  # (L, B, S, K, H)
    sequence_lengths: Array  # (B,)
    write_positions: Array  # (B,)

    # ──────────────────────────────────────────────────────────────────────
    # pytree helpers
    # ──────────────────────────────────────────────────────────────────────
    def tree_flatten(self):
        return (self.key, self.value, self.sequence_lengths, self.write_positions), None

    @classmethod
    def tree_unflatten(cls, _, leaves):
        return cls(*leaves)

    # ──────────────────────────────────────────────────────────────────────
    # convenience
    # ──────────────────────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, ...]:
        return self.key.shape

    @property
    def batch_size(self) -> int:
        return self.key.shape[1]

    @property
    def max_seq_len(self) -> int:
        return self.key.shape[2]

    @property
    def num_layers(self) -> int:
        return self.key.shape[0]


# ============================================================================
# Public API
# ============================================================================


def init_cache(
    *,
    batch: int,
    max_seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> KVCache:
    """Allocate an initialized KV cache."""

    cache_shape = (num_layers, batch, max_seq_len, num_kv_heads, head_dim)
    zeros = lambda shape: jnp.zeros(shape, dtype=dtype)

    return KVCache(
        key=zeros(cache_shape),
        value=zeros(cache_shape),
        sequence_lengths=jnp.zeros((batch,), dtype=jnp.int32),
        write_positions=jnp.zeros((batch,), dtype=jnp.int32),
    )


def create_cache_partition_spec(key: str, mesh_axes: dict[str, str]) -> PartitionSpec:
    """Return a PartitionSpec for any KVCache field."""

    if key in ("key", "value"):
        return PartitionSpec(None, mesh_axes.get("batch"), None, mesh_axes.get("heads"), None)
    if key in ("sequence_lengths", "write_positions"):
        return PartitionSpec(mesh_axes.get("batch"))
    return PartitionSpec()


def shard_kvcache_with_tree_map(cache: KVCache, mesh: Any, mesh_axes: dict[str, str]) -> KVCache:
    """Shard each field of the KVCache according to a per‑field spec."""

    def put(x: Array, field: str) -> Array:
        spec = create_cache_partition_spec(field, mesh_axes)
        return jax.device_put(x, NamedSharding(mesh, spec))

    return KVCache(
        key=put(cache.key, "key"),
        value=put(cache.value, "value"),
        sequence_lengths=put(cache.sequence_lengths, "sequence_lengths"),
        write_positions=put(cache.write_positions, "write_positions"),
    )


# ============================================================================
# Cache update helpers
# ============================================================================


def _update_ragged(
    key_cache_layer: Array,   # (B, S, K, H)
    val_cache_layer: Array,   # (B, S, K, H)
    key_proj: Array,          # (B, 1, K, H) or (B, K, H)
    value_proj: Array,        # (B, 1, K, H) or (B, K, H)
    write_pos_B: Array,       # (B,)
) -> tuple[Array, Array]:
    """Single‑token (generation) update for ragged batches."""

    # Squeeze T‑dimension if present
    if key_proj.ndim == 4:  # (B, 1, K, H)
        key_proj = jnp.squeeze(key_proj, axis=1)
        value_proj = jnp.squeeze(value_proj, axis=1)

    def update_one(cache_k, cache_v, new_k, new_v, pos):
        new_k = new_k[None, :, :]  # (1, K, H)
        new_v = new_v[None, :, :]
        cache_k = lax.dynamic_update_slice(cache_k, new_k, (pos, 0, 0))
        cache_v = lax.dynamic_update_slice(cache_v, new_v, (pos, 0, 0))
        return cache_k, cache_v

    return jax.vmap(update_one)(key_cache_layer, val_cache_layer, key_proj, value_proj, write_pos_B)


def _update_dense(
    key_cache_layer: Array,   # (B, S, K, H)
    val_cache_layer: Array,   # (B, S, K, H)
    key_proj: Array,          # (B, T, K, H)
    value_proj: Array,        # (B, T, K, H)
    write_pos_B: Array,       # (B,)
    seq_lens_B: Array,        # (B,)
) -> tuple[Array, Array]:
    """Chunk (prefill) update for dense sequences."""

    batch_size, max_cache_len, _, _ = key_cache_layer.shape
    timeline_len = key_proj.shape[1]

    # Guard against over‑length chunks
    if timeline_len > max_cache_len:
        raise ValueError(f"Chunk length exceeds cache capacity ({timeline_len} > {max_cache_len})")

    write_pos_B = write_pos_B.reshape(-1)
    seq_lens_B = seq_lens_B.reshape(-1)

    # Calculate positions (no wraparound)
    token_positions = write_pos_B[:, None] + jnp.arange(timeline_len)[None, :]
    
    # Check for overflow
    max_position = jnp.max(token_positions + seq_lens_B[:, None] - 1)
    # if max_position >= max_cache_len:
    #     raise ValueError(f"Cache overflow: trying to write to position {max_position} but cache size is {max_cache_len}")
    
    cache_indices = token_positions

    valid_mask = jnp.arange(timeline_len)[None, :] < seq_lens_B[:, None]
    cache_indices = jnp.where(valid_mask, cache_indices, -1)

    batch_indices = jnp.arange(batch_size)[:, None]
    batch_indices = jnp.broadcast_to(batch_indices, cache_indices.shape)

    updated_key = key_cache_layer.at[batch_indices, cache_indices].set(key_proj, mode="drop")
    updated_val = val_cache_layer.at[batch_indices, cache_indices].set(value_proj, mode="drop")

    return updated_key, updated_val


# ============================================================================
# Public updater
# ============================================================================


@partial(jax.jit, static_argnames=("layer", "ragged"))
def update_cache_layer(
    cache: KVCache,
    key_proj: Array,      # (B, T, K, H)
    value_proj: Array,    # (B, T, K, H)
    *,
    write_pos_B: Array,   # (B,) or scalar
    seq_lens_B: Array,    # (B,) or scalar (tokens being written)
    layer: int,
    ragged: bool = False,
):
    """Update one layer of the KV cache and return (key_layer, value_layer, new_cache)."""

    if not 0 <= layer < cache.num_layers:
        raise ValueError(f"Layer {layer} out of bounds [0, {cache.num_layers})")

    key_layer = cache.key[layer]
    val_layer = cache.value[layer]

    write_pos_B = jnp.asarray(write_pos_B)
    seq_lens_B = jnp.asarray(seq_lens_B)

    if write_pos_B.ndim == 0:
        write_pos_B = jnp.broadcast_to(write_pos_B, (cache.batch_size,))
    if seq_lens_B.ndim == 0:
        seq_lens_B = jnp.broadcast_to(seq_lens_B, (cache.batch_size,))

    if ragged or key_proj.shape[1] == 1:
        updated_key, updated_val = _update_ragged(
            key_layer, val_layer, key_proj, value_proj, write_pos_B
        )
    else:
        updated_key, updated_val = _update_dense(
            key_layer, val_layer, key_proj, value_proj, write_pos_B, seq_lens_B
        )

    # ── bookkeeping (no ring buffer) ──────────────────────────────────────
    new_seq_lengths = jnp.maximum(
        cache.sequence_lengths, write_pos_B + seq_lens_B
    )
    
    new_write_pos = cache.write_positions + seq_lens_B  # No wraparound
    

    new_cache = KVCache(
        key=cache.key.at[layer].set(updated_key),
        value=cache.value.at[layer].set(updated_val),
        sequence_lengths=new_seq_lengths,
        write_positions=new_write_pos,
    )

    return updated_key, updated_val, new_cache


# ============================================================================
# Utility helpers
# ============================================================================

def get_valid_cache_positions(cache: KVCache) -> Array:
    """Boolean mask of valid positions."""

    positions = jnp.arange(cache.max_seq_len)[None, :]
    return positions < cache.sequence_lengths[:, None]


def reset_cache_positions(cache: KVCache, batch_indices: Optional[Array] = None) -> KVCache:
    """Reset sequence length and write pointer for selected batch elements."""

    if batch_indices is None:
        new_len = jnp.zeros_like(cache.sequence_lengths)
        new_pos = jnp.zeros_like(cache.write_positions)
    else:
        new_len = cache.sequence_lengths.at[batch_indices].set(0)
        new_pos = cache.write_positions.at[batch_indices].set(0)

    return KVCache(
        key=cache.key,
        value=cache.value,
        sequence_lengths=new_len,
        write_positions=new_pos,
    )


def cache_info_string(cache: KVCache) -> str:
    """Return a human‑readable string summarising cache state."""

    gb = (cache.key.nbytes + cache.value.nbytes) / 1024 ** 3
    return (
        f"KVCache Info:\n"
        f"  Shape: {cache.shape}\n"
        f"  Batch size: {cache.batch_size}\n"
        f"  Max sequence length: {cache.max_seq_len}\n"
        f"  Number of layers: {cache.num_layers}\n"
        f"  Current sequence lengths: {cache.sequence_lengths.tolist()}\n"
        f"  Current write positions: {cache.write_positions.tolist()}\n"
        f"  Memory usage: {gb:.2f} GB"
    )

# %%