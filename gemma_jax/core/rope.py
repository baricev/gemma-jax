#%%
"""Experimental RoPE (Rotary Positional Embedding) implementations for JAX.

Examples included:
1. Cached RoPE: Uses precomputed sine and cosine tables.
2. Computed RoPE: Uses complex numbers and outer products to compute RoPE on-the-fly (less efficient but mathematically clearer).
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from typing import Any

Config = Any


# --- Cached RoPE implementation ---
def precompute_rope_embeddings(
    head_dim: int,
    max_seq_len: int,
    base_freq: float,
    scale_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
  """Precompute the Rope embeddings for the given configuration."""
  half_dim = head_dim // 2
  freq = base_freq ** (-2 * jnp.arange(half_dim) / head_dim)
  positions = jnp.arange(max_seq_len, dtype=jnp.float32) / scale_factor
  sinusoid = jnp.outer(positions, freq)
  return jnp.sin(sinusoid), jnp.cos(sinusoid)


def apply_rope_cached(
    x: jax.Array,
    positions: jax.Array,
    sin_cache: jax.Array = None,
    cos_cache: jax.Array = None,
) -> jax.Array:
  """Apply RoPE using cached sin/cos tables."""
  sin_vals = sin_cache[positions]
  cos_vals = cos_cache[positions]
  sin_vals = sin_vals[:, :, None, :]
  cos_vals = cos_vals[:, :, None, :]
  x1, x2 = jnp.split(x, 2, axis=-1)
  return jnp.concatenate([x1 * cos_vals - x2 * sin_vals, x2 * cos_vals + x1 * sin_vals], axis=-1).astype(x.dtype)

def init_rope_cache(config: Config) -> jax.Array:
  """Load the Rope cache for the given configuration."""
  if config.cache_length is None:
    raise ValueError("config.cache_length cannot be None for RoPE cache loading")
  local_rope = precompute_rope_embeddings(
      config.head_dim,
      config.cache_length,
      config.local_base_frequency,
      config.local_scale_factor,
      # config.rope_base_frequency,
      # config.rope_scale_factor,
  )
  global_rope = precompute_rope_embeddings(
      config.head_dim,
      config.cache_length,
      config.global_base_frequency,
      config.global_scale_factor,
      # config.global_rope_base_frequency,
      # config.global_rope_scale_factor,
  )
  return jnp.array([local_rope, global_rope])


def load_rope_cache(mesh: Mesh, config: Config) -> jax.Array:
  """Load the Rope cache for the given configuration."""
  if config.cache_length is None:
    raise ValueError("config.cache_length cannot be None for RoPE cache loading")

  rope_cache = init_rope_cache(config)

  # Shard the rope cache using the mesh and mesh_axes:
  rope_sharding = NamedSharding(mesh, P(None, None, "model", None))
  rope_cache = jax.device_put(rope_cache, rope_sharding)
  return rope_cache


# --- Computed RoPE implementation (complex number) ---


def apply_rope_outer_product(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Simple RoPE implementation using complex numbers and outer product.

  Args:
      inputs: Array of shape [B, L, N, H]
      positions: Array of shape [B, L]
      base_frequency: Base frequency for the rotations
      scale_factor: Scaling factor for positions (default 1.0)

  Returns:
      Array with RoPE applied, same shape as inputs
  """
  B, L, N, H = inputs.shape
  half_head = H // 2

  # Create frequency components using outer product
  positions_1d = positions.reshape(-1)  # Flatten to [B*L]
  freqs = base_frequency ** (2 * jnp.arange(half_head) / H)
  inv_scale_freqs = 1.0 / (freqs * scale_factor)

  # Outer product: positions [B*L] x inv_scale_freqs [H/2] -> [B*L, H/2]
  sinusoid_inp = jnp.outer(positions_1d, inv_scale_freqs)
  sinusoid_inp = sinusoid_inp.reshape(B, L, 1, half_head)  # Reshape for broadcasting

  # Compute rotation factors
  sin, cos = jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)
  rotator = cos + 1j * sin  # Combine into complex rotation factors

  # Split input and apply rotation
  x1, x2 = inputs[..., :half_head], inputs[..., half_head:]
  complex_input = x1 + 1j * x2
  rotated = complex_input * rotator

  # Recombine and return
  rotated_x1, rotated_x2 = rotated.real, rotated.imag
  return jnp.concatenate([rotated_x1, rotated_x2], axis=-1).astype(inputs.dtype)


# --- DeepMind RoPE implementation ---
# apply_rope from: https://github.com/google-deepmind/gemma/blob/main/gemma/gm/math/_positional_embeddings.py
def apply_rope(
    inputs: jax.Array,
    positions: jax.Array,
    *,
    base_frequency: int,
    scale_factor: float = 1.0,
) -> jax.Array:
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

  sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)

# %%
