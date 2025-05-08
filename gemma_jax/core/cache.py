"""
KV-cache utilities for Gemma-3 in functional JAX style.

The module exposes two public helpers:

- `init_cache` - allocate and shard an empty cache
- `update_cache_layer` - in-place update of a **single** layer's key/value slice
- `update_cache_layers` - in-place update of **multiple** layers key/value slices

All layout-specific logic is compiled once up-front, so the hot path that runs
inside `self_attention` contains no Python conditionals and is fully
jax.jit/jax.pmap friendly.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Dict, Callable, Any, Final, NamedTuple, Literal

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing import NamedTuple


# NOTE: Both Layout Classes functionally equivalent.

# LayoutType is an alias for int, used to represent the layout type of the cache.
LayoutType = int


def get_shape_map(config: Any) -> dict[str, int]:
  """Return a dictionary mapping shape dimensions to their sizes."""
  return {
      "L": config.num_layers,
      "B": config.batch_size,
      "K": config.num_kv_heads,
      "S": config.cache_length,
      "H": config.head_dim,
  }


class Layout(NamedTuple):
  """Allowed cache memory layouts."""

  kind: int


# Create Cache Layouts here (memory layout, sharding dimensions, and shape suffix names and mapping)
"""Allowed cache memory layouts."""
SEQUENCE_HEADS: Final[int] = 0  # [L, B, S, K, H]
HEADS_SEQUENCE: Final[int] = 1  # [L, B, K, S, H]

layout_map = {SEQUENCE_HEADS: ("L", "B", "S", "K", "H"), HEADS_SEQUENCE: ("L", "B", "K", "S", "H")}
shard_dims = {SEQUENCE_HEADS: ("B", "K"), HEADS_SEQUENCE: ("K", "B")}
aliases_map = {SEQUENCE_HEADS: "SEQUENCE_HEADS", HEADS_SEQUENCE: "HEADS_SEQUENCE"}
SUPPORTED_LAYOUTS = [SEQUENCE_HEADS, HEADS_SEQUENCE]

# Cache Layout Creation Functions


def cache_shape(
    kind: LayoutType,
    config: Any,
    layout_map: dict[LayoutType, tuple[str, ...]] = layout_map,
    shard_dims: dict[LayoutType, tuple[str, ...]] = shard_dims,  # no op   [API compatibility]
) -> tuple[int, ...]:
  if not kind in SUPPORTED_LAYOUTS:
    raise ValueError(f"Unsupported layout: {kind}")
  shape_map = get_shape_map(config)

  if hasattr(kind, "ordered_shapes"):
    return tuple(shape_map[i] for i in kind.ordered_layout)  # pylint ignore
  else:
    return tuple(shape_map[i] for i in layout_map[kind])


def cache_layout(
    kind: LayoutType,
    layout_map: dict[LayoutType, tuple[str, ...]] = layout_map,
    shard_dims: dict[LayoutType, tuple[str, ...]] = shard_dims,  # no op
) -> tuple[str, ...]:
  if not kind in SUPPORTED_LAYOUTS:
    raise ValueError(f"Unsupported layout: {kind}")
  return layout_map[kind]


def cache_spec(
    kind,
    mesh: Mesh,
    layout_map: dict[LayoutType, tuple[str, ...]] = layout_map,
    shard_dims: dict[LayoutType, tuple[str, ...]] = shard_dims,
) -> P:
  if not kind in SUPPORTED_LAYOUTS:
    raise ValueError(f"Unsupported layout: {kind}")

  partition_mapping = {k: v for k, v in zip(shard_dims[kind], mesh.axis_names)}
  partition_mapping = {**partition_mapping, **layout_map}

  if hasattr(kind, "ordered_shapes"):
    partition_mapping = {k: v for k, v in zip(kind.shard_dims, mesh.axis_names)}
    # return P(tuple(partition_mapping.get(k, None) for k in kind.ordered_layout ))
    spec = tuple(partition_mapping.get(k, None) for k in kind.ordered_layout)
  else:
    spec = tuple(partition_mapping.get(k, None) for k in partition_mapping[kind])

  assert (
      len(spec) != 1
  ), "Partition spec must have at least 2 dimensions. Is it nested?"  # ensure spec tuple is not wrapped in a tuple
  return P(*spec)


def cache_spec_map(
    kind,
    mesh: Mesh,
    layout_map: dict[LayoutType, tuple[str, ...]] = layout_map,
    shard_dims: dict[LayoutType, tuple[str, ...]] = shard_dims,
) -> dict[str, Any]:
  if not kind in SUPPORTED_LAYOUTS:
    raise ValueError(f"Unsupported layout: {kind}")

  partition_mapping = {k: v for k, v in zip(shard_dims[kind], mesh.axis_names)}
  partition_mapping = {**partition_mapping, **layout_map}

  if hasattr(kind, "ordered_shapes"):
    partition_mapping = {k: v for k, v in zip(kind.shard_dims, mesh.axis_names)}

  return {k: partition_mapping.get(k, None) for k in partition_mapping[kind]}


def layout_by_name(
    name: LayoutType,
    _ALIASES: dict[LayoutType, str] = aliases_map,
) -> str:
  try:
    return _ALIASES[name]
  except KeyError as exc:
    raise ValueError(f"Unknown layout: {name}. Supported: {SUPPORTED_LAYOUTS}") from exc


# PyTree container
@jax.tree_util.register_pytree_node_class
@dataclass
class KVCache:
  """Sharded key/value tensors. Outermost dimension is layer."""

  key: Array
  value: Array

  def tree_flatten(self):
    return (self.key, self.value), None

  @classmethod
  def tree_unflatten(cls, _, leaves):
    key, value = leaves
    return cls(key, value)


# Public API helpers
def init_cache(
    *,
    mesh: Mesh,
    config: Any,  # expects .num_layers .batch_size .num_kv_heads .cache_length .head_dim
    dtype: jnp.dtype = jnp.bfloat16,
    kind: LayoutType = HEADS_SEQUENCE,
    layout_map: dict[LayoutType, tuple[str, ...]] = layout_map,
    shard_dims: dict[LayoutType, tuple[str, ...]] = shard_dims,
    aliases_map: dict[LayoutType, str] = aliases_map,
) -> KVCache:
  """Allocate an uninitialized, correctly sharded KV cache."""
  shape_map = get_shape_map(config)

  shape = cache_shape(kind, config, layout_map, shard_dims)
  ps = cache_spec(kind, mesh, layout_map, shard_dims)
  suffixes = cache_layout(kind, layout_map, shard_dims)
  sharding = NamedSharding(mesh, ps)

  layout_name = layout_by_name(kind, aliases_map)
  print(f"Initializing {layout_name} cache with shape {shape}")

  print(f"Cache shape {shape}, partition spec {ps}")
  print(f"Sharding cache with shape {shape} using partition spec: {sharding}")
  print(f"Sharding {layout_name} across {cache_spec_map(kind, mesh)}")

  empty = lambda: jax.device_put(jnp.empty(shape, dtype=dtype), sharding)
  return KVCache(empty(), empty())


# Layout-specialized update kernels
UpdateFn = Callable[[Array, Array, Array, Array, int], Tuple[Array, Array]]


def _make_update_fn(kind: LayoutType) -> UpdateFn:
  """Build a jitted function that writes one (B,T,K,H) slice for key and value."""

  @jax.jit
  def _update(key_slice: Array, val_slice: Array, key_proj: Array, value_proj: Array, write_index: int):
    if kind == SEQUENCE_HEADS:  # [B, S, K, H]
      key_update = jax.lax.dynamic_update_slice(key_slice, key_proj, (0, write_index, 0, 0))
      val_update = jax.lax.dynamic_update_slice(val_slice, value_proj, (0, write_index, 0, 0))
    else:  # HEADS_SEQUENCE [B, K, S, H]
      key_t = jnp.transpose(key_proj, (0, 2, 1, 3))
      val_t = jnp.transpose(value_proj, (0, 2, 1, 3))
      key_update = jax.lax.dynamic_update_slice(key_slice, key_t, (0, 0, write_index, 0))
      val_update = jax.lax.dynamic_update_slice(val_slice, val_t, (0, 0, write_index, 0))
    return key_update, val_update

  return _update


# Compile both kernels once
_UPDATE_FNS: Dict[LayoutType, UpdateFn] = {
    cache_layout: _make_update_fn(cache_layout) for cache_layout in (SEQUENCE_HEADS, HEADS_SEQUENCE)
}


def update_cache_layer_by_layer(
    cache_layer: tuple[Array, Array],
    key_proj: Array,
    value_proj: Array,
    *,
    write_index: int,
    layer: int,
    kind: LayoutType = HEADS_SEQUENCE,
) -> tuple[Array, Array]:
  """
  Functional in-place update for a single layer.

  Returns a new cache layer (tuple consiting of the new key, and value arrays).
  Does *not* return a new KVCache.
  """
  # Bounds checking
  cache_key, cache_value = cache_layer
  if not (0 <= layer < cache_key.shape[0]):
    raise ValueError(f"Layer {layer} out of bounds [0, {cache_key.shape[0]})")

  update_fn = _UPDATE_FNS[kind]
  key_layer = jax.lax.dynamic_index_in_dim(cache_key, layer, keepdims=False)
  val_layer = jax.lax.dynamic_index_in_dim(cache_value, layer, keepdims=False)
  new_key_layer, new_val_layer = update_fn(key_layer, val_layer, key_proj, value_proj, write_index)
  return new_key_layer, new_val_layer


def update_cache_layer(
    cache: KVCache, key_proj: Array, value_proj: Array, *, write_index: int, layer: int, kind: LayoutType = HEADS_SEQUENCE
) -> tuple[Array, Array, KVCache]:
  """Functional in-place update for a single layer. Returns new KVCache."""
  # Bounds checking
  if not (0 <= layer < cache.key.shape[0]):
    raise ValueError(f"Layer {layer} out of bounds [0, {cache.key.shape[0]})")

  update_fn = _UPDATE_FNS[kind]
  key_layer = jax.lax.dynamic_index_in_dim(cache.key, layer, keepdims=False)
  val_layer = jax.lax.dynamic_index_in_dim(cache.value, layer, keepdims=False)
  new_key_layer, new_val_layer = update_fn(key_layer, val_layer, key_proj, value_proj, write_index)
  key = cache.key.at[layer].set(new_key_layer)
  value = cache.value.at[layer].set(new_val_layer)
  return new_key_layer, new_val_layer, KVCache(key, value)


def update_cache_layers(
    cache: KVCache,
    key_proj: Array,
    value_proj: Array,
    *,
    write_index: int,
    layers: list[int],
    kind: LayoutType = HEADS_SEQUENCE,
) -> KVCache:
  """Functional in-place update for multiple layers. Returns new KVCache."""
  update_fn = _UPDATE_FNS[kind]
  key, value = cache.key, cache.value

  # Bounds checking
  for layer in layers:
    if not (0 <= layer < cache.key.shape[0]):
      raise ValueError(f"Layer {layer} out of bounds [0, {cache.key.shape[0]})")

  for layer in layers:
    key_layer = jax.lax.dynamic_index_in_dim(key, layer, keepdims=False)
    val_layer = jax.lax.dynamic_index_in_dim(value, layer, keepdims=False)
    new_key_layer, new_val_layer = update_fn(key_layer, val_layer, key_proj, value_proj, write_index)
    key = key.at[layer].set(new_key_layer)
    value = value.at[layer].set(new_val_layer)
  return KVCache(key, value)
