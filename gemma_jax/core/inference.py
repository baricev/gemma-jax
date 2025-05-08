"""" Inference and sampling functions for text generation.  """

from functools import partial
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.random import categorical


def greedy_sample(logits: jax.Array, positions: jax.Array) -> jax.Array:
  """Select highest probability token from last positions in each sequence. (Ragged sampling, prefill)"""
  B = logits.shape[0]
  last_pos = positions[:, -1]
  selected = logits[jnp.arange(B), last_pos]
  return jnp.argmax(selected, axis=-1).reshape(-1, 1)


def greedy_sample_one_hot(logits: jax.Array, positions: jax.Array) -> jax.Array:
  """Select highest probability token using one-hot trick for efficient indexing."""
  last_pos = positions[:, -1]  # (B,)
  # Create one-hot mask for last positions (B, seq_len)
  mask = jax.nn.one_hot(last_pos, logits.shape[1], axis=1)
  # Select final logits using matrix multiplication (B, vocab_size)
  final_logits = jnp.einsum("bl,blv->bv", mask, logits)
  # Return argmax with preserved dimensions (B, 1)
  return jnp.argmax(final_logits, axis=-1, keepdims=True)


def greedy_sample_single_step(logits: jax.Array) -> jax.Array:
  """Sample tokens greedily by selecting the token with the highest probability."""
  if not isinstance(logits, jax.Array):
    raise TypeError(f"Logits must be a jax.Array, got {type(logits)}")
  if logits.ndim != 3:
    raise ValueError(f"Logits must have 3 dimensions (batch, seq_len, vocab), got {logits.ndim}")
  # Find the index of the maximum logit along the vocabulary axis
  sampled_indices = jnp.argmax(logits, axis=-1)
  return sampled_indices.astype(jnp.int32)


def sample_temperature(key: jax.Array, logits: jax.Array, temperature: float = 1.0) -> jax.Array:
  """Sample tokens using a temperature-scaled distribution."""
  if not isinstance(key, jax.Array) or not hasattr(key, "dtype") or key.dtype != jax.random.key(0).dtype:
    raise TypeError(f"key must be a JAX Array, got {type(key)}")
  if not isinstance(logits, jax.Array):
    raise TypeError(f"Logits must be a jax.Array, got {type(logits)}")
  if logits.ndim != 3:
    raise ValueError(f"Logits must have 3 dimensions (batch, seq_len, vocab), got {logits.ndim}")
  if not isinstance(temperature, (float, int)) or temperature < 0:
    raise ValueError(f"Temperature must be a non-negative float, got {temperature}")

  # Avoid division by zero; very small temperature effectively mimics greedy.
  effective_temperature = jnp.maximum(temperature, 1e-9)
  # Scale logits
  scaled_logits = logits / effective_temperature

  # Sample from the scaled logits distribution (jax.random.categorical expects logits)
  sampled_indices = categorical(key, scaled_logits, axis=-1)
  return sampled_indices.astype(jnp.int32)


def sample_top_k(key: jax.Array, logits: jax.Array, k: int, temperature: float = 1.0) -> jax.Array:
  """Sample tokens from the top-k logits."""
  if not isinstance(key, jax.Array) or not hasattr(key, "dtype") or key.dtype != jax.random.key(0).dtype:
    raise TypeError(f"key must be a JAX Array, got {type(key)}")
  if not isinstance(logits, jax.Array):
    raise TypeError(f"Logits must be a jax.Array, got {type(logits)}")
  if logits.ndim != 3:
    raise ValueError(f"Logits must have 3 dimensions (batch, seq_len, vocab), got {logits.ndim}")
  if not isinstance(k, int) or k <= 0:
    raise ValueError(f"k must be a positive integer, got {k}")
  if not isinstance(temperature, (float, int)) or temperature < 0:
    raise ValueError(f"Temperature must be a non-negative float, got {temperature}")

  vocab_size = logits.shape[-1]
  # Ensure k is not larger than vocabulary size
  k = min(k, vocab_size)

  # Apply temperature
  effective_temperature = jnp.maximum(temperature, 1e-9)
  scaled_logits = logits / effective_temperature

  # Find the top k logits and their indices (sorted descending by value)
  top_k_logits, top_k_indices = jax.lax.top_k(scaled_logits, k=k)

  # Sample from the top-k logits distribution
  sampled_relative_indices = categorical(key, top_k_logits, axis=-1)

  # Map the relative indices back to the original vocabulary indices
  sampled_indices = jnp.take_along_axis(top_k_indices, sampled_relative_indices[..., None], axis=-1).squeeze(-1)

  return sampled_indices.astype(jnp.int32)


def sample_top_p(
    key: jax.Array,
    logits: jax.Array,
    p: float,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
) -> jax.Array:
  """Sample tokens from the top-p (nucleus) logits."""
  if not isinstance(key, jax.Array) or not hasattr(key, "dtype") or key.dtype != jax.random.key(0).dtype:
    raise TypeError(f"key must be a JAX Array, got {type(key)}")
  if not isinstance(logits, jax.Array):
    raise TypeError(f"Logits must be a jax.Array, got {type(logits)}")
  if logits.ndim != 3:
    raise ValueError(f"Logits must have 3 dimensions (batch, seq_len, vocab), got {logits.ndim}")
  if not isinstance(p, float) or not (0.0 < p <= 1.0):
    raise ValueError(f"p must be a float in (0, 1], got {p}")
  if not isinstance(temperature, (float, int)) or temperature < 0:
    raise ValueError(f"Temperature must be a non-negative float, got {temperature}")
  if not isinstance(min_tokens_to_keep, int) or min_tokens_to_keep < 1:
    raise ValueError(f"min_tokens_to_keep must be a positive integer, got {min_tokens_to_keep}")

  # Apply temperature
  effective_temperature = jnp.maximum(temperature, 1e-9)
  scaled_logits = logits / effective_temperature

  # Calculate probabilities
  probs = jax.nn.softmax(scaled_logits, axis=-1)

  # Sort probabilities descending and get corresponding indices/logits
  sorted_indices = jnp.argsort(probs, axis=-1)[..., ::-1]
  sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
  sorted_logits = jnp.take_along_axis(scaled_logits, sorted_indices, axis=-1)

  # Calculate cumulative probabilities
  cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)

  # Create mask: exclude tokens Thresholding.
  # Shift mask ensures token triggering threshold is excluded.
  cutoff_mask = cumulative_probs > p
  exclude_mask = jnp.roll(cutoff_mask, shift=1, axis=-1)
  exclude_mask = exclude_mask.at[..., 0].set(False)  # Always keep the highest prob token

  # Ensure min_tokens_to_keep included:
  # Keep if NOT excluded by p OR in top min_tokens_to_keep
  nucleus_mask_sorted = jnp.logical_or(
      jnp.logical_not(exclude_mask),
      jnp.arange(logits.shape[-1]) < min_tokens_to_keep,
  )

  # Apply the mask to the *sorted* logits, setting excluded tokens to -inf
  masked_sorted_logits = jnp.where(nucleus_mask_sorted, sorted_logits, -jnp.inf)

  # Sample from the masked, sorted logits distribution
  sampled_sorted_indices = categorical(key, masked_sorted_logits, axis=-1)

  # Map the sampled index (relative to the sorted list)
  # back to the original vocabulary index
  sampled_indices = jnp.take_along_axis(sorted_indices, sampled_sorted_indices[..., None], axis=-1).squeeze(-1)

  return sampled_indices.astype(jnp.int32)
