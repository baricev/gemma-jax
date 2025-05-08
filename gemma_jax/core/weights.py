"""Experimental sharding and loading code for Gemma3.

This module provides helper functions for:
  - loading and formatting checkpoints,
  - creating a device mesh,
  - computing PartitionSpec trees,
  - sharding parameters or NamedTuples models.

"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import categorical
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax import Array

import orbax.checkpoint as ocp
from dataclasses import replace
import functools
from functools import partial
import os
import time
from typing import Any, NamedTuple, TypeVar, Union

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

PAD_ID: int = 0
EOS_ID: int = 1
K_MASK: float = -2.3819763e38
ATTENTION_TYPE_GLOBAL: int = 1
ATTENTION_TYPE_LOCAL_SLIDING: int = 2
GEMMA3_ATTENTION_PATTERN: tuple[int, ...] = (2, 2, 2, 2, 2, 1)

# --- Type Definitions ---
T = TypeVar("T")
Leaf = Any
FlatDict = dict[str, Array]
NestedDict = dict[str, Any]

# --- Constants ---

# Text decoder constants
PAD_ID = 0
EOS_ID = 1
K_MASK = -2.3819763e38

ATTENTION_TYPE_GLOBAL = 1
ATTENTION_TYPE_LOCAL_SLIDING = 2

GEMMA3_ATTENTION_PATTERN = (
    2,
    2,
    2,
    2,
    2,
    1,  # LOCAL_SLIDING * 5, GLOBAL * 1
)


# --- Model Configuration ---

GEMMA3_VARIANTS = np.array(
    [
        # 1B, 4B, 12B, 27B
        [1, 4, 12, 27],  # model_size
        [26, 34, 48, 62],  # num_layers
        [4, 8, 16, 32],  # num_heads
        [1, 4, 8, 16],  # num_kv_heads
        [1152, 2560, 3840, 5376],  # embed_dim (9*128, 20*128, 30*128, 42*128)
        # hidden_dim (6*9*128, 4*20*128, 4*30*128, 4*42*128)
        [6912, 15360, 20480, 21504],
        [256, 256, 256, 128],  # head_dim (2*128, 2*128, 2*128, 1*128)
    ],
    np.int32,
)


class VariantConfig(NamedTuple):
  model_size: int
  num_layers: int
  num_heads: int
  num_kv_heads: int
  embed_dim: int
  hidden_dim: int
  head_dim: int


def get_gemma3_variants(variant: int) -> list[int]:
  """Retrieve configuration parameters for a specific Gemma3 model variant.

  Args:
      The model size variant (e.g., 1, 4, 12, 27).
  Returns:
      A list containing [num_layers, num_heads, num_kv_heads, embed_dim, hidden_dim, head_dim].
  """

  # find the index of the variant in the gemma3_variants array
  index = np.where(GEMMA3_VARIANTS[0] == variant)[0][0]
  # return the corresponding values for num_layers, num_heads, num_kv_heads, embed_dim, hidden_dim
  config = GEMMA3_VARIANTS[1:, index]
  return config.tolist()


class Config(NamedTuple):
  batch_size: int
  padded_input_size: int
  model_size: int
  num_layers: int
  num_heads: int
  num_kv_heads: int
  embed_dim: int
  hidden_dim: int
  head_dim: int
  # Remaining fields are the same except for the 1B model which does not use GQA
  vocab_size: int
  query_pre_attn_scalar: float
  attention_types: tuple[int, ...]
  attention_pattern: tuple[int, ...]
  local_base_frequency: int
  global_base_frequency: int
  global_scale_factor: float
  local_scale_factor: float
  final_logit_softcap: float | None
  window_size: int
  transpose_gating_einsum: bool
  attn_logits_soft_cap: float | None
  use_post_attn_norm: bool
  use_post_ffw_norm: bool
  use_qk_norm: bool
  vision_encoder: tuple[int, ...] | None  # Placeholder type
  mm_extra_vocab_size: int | None
  cache_length: int = 1024
  use_rope_cache: bool = False
  rope_cache: Array | None = None
  max_position_length: int = 1024
  max_gen_length: int = 1024
  generate_steps: int = 1


def query_pre_attn_norm(model_size: int) -> float:
  """Calculate the pre-attention normalization scalar for the query.

  Scales by '1/sqrt(embed_dim // num_heads)' if 27B model,
  otherwise by '1/sqrt(head_dim=256)'.
  """
  head_dim_27b = 128
  embed_dim_27b = 5376
  num_heads_27b = 32
  return (embed_dim_27b / num_heads_27b) ** -0.5 if model_size == 27 else (head_dim_27b * 2) ** -0.5


def make_attn_layers_types(
    pattern: tuple[int, ...],
    num_layers: int,
) -> tuple[int, ...]:
  """Returns attention patterns for each layer by repeating the input pattern.

  Args:
    pattern: The base pattern sequence to repeat.
    num_layers: Total number of attention layers needed.

  Returns:
    A tuple of attention patterns with length equal to num_layers.
  """
  full_repeats, remainder = divmod(num_layers, len(pattern))
  return tuple(pattern * full_repeats + pattern[:remainder])


def create_gemma3_config(
    model_size: int,
    batch_size: int,
    padded_input_size: int,
    cache_length: int = 1024,
    window_size: int = 1024,
    use_rope_cache: bool = False,
    rope_cache: Array | None = None,
    max_position_length: int = 1024,
    max_gen_length: int = 1024,
    generate_steps: int = 1,
) -> Config:
  """Create Configuration for Gemma3 series of models."""
  if model_size not in [1, 4, 12, 27]:
    raise ValueError(f"Unsupported model size: {model_size}")

  # Get the configuration for the specified model size
  (
      num_layers,
      num_heads,
      num_kv_heads,
      embed_dim,
      hidden_dim,
      head_dim,
  ) = get_gemma3_variants(model_size)
  assert (
      len(make_attn_layers_types((2, 2, 2, 2, 2, 1), num_layers)) == num_layers
  ), "Attention layers types mismatch with num_layers"

  return Config(
      model_size=model_size,
      batch_size=batch_size,
      padded_input_size=padded_input_size,
      cache_length=cache_length,
      window_size=window_size,
      num_layers=num_layers,
      num_heads=num_heads,
      num_kv_heads=num_kv_heads,
      embed_dim=embed_dim,
      hidden_dim=hidden_dim,
      head_dim=head_dim,
      query_pre_attn_scalar=query_pre_attn_norm(model_size),
      attention_types=make_attn_layers_types((2, 2, 2, 2, 2, 1), num_layers),
      attention_pattern=make_attn_layers_types((2, 2, 2, 2, 2, 1), num_layers),
      vocab_size=262144,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attn_logits_soft_cap=None,
      final_logit_softcap=None,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      local_scale_factor=1.0,
      global_base_frequency=1_000_000,
      global_scale_factor=8.0,
      vision_encoder=None,  # if text_only else gemma_vision.SigLiPFromPatches(),
      mm_extra_vocab_size=0,  # if text_only else 128,
      use_rope_cache=use_rope_cache,
      rope_cache=rope_cache,
      max_position_length=max_position_length,
      max_gen_length=max_gen_length,
      generate_steps=generate_steps,
  )


# --- Model Definition ---
class Layer(NamedTuple):
  attn_key_norm_scale: Array  # (head_dim,) - Replicated
  attn_query_norm_scale: Array  # (head_dim,) - Replicated
  output_proj: Array  # (num_heads, head_dim, embed_dim) -> Shard embed_dim
  kv_proj: Array  # (2, K, D, H) -> Shard embed_dim
  q_proj: Array  # (N, D, H) -> Shard embed_dim #
  gating_weights: Array  # (2, F, D) -> Shard hidden_dim (D)
  output_weights: Array  # (num_heads, head_dim, embed_dim) -> Shard embed_dim
  post_attention_norm_scale: Array  # (embed_dim,) - Replicated
  post_ffw_norm_scale: Array  # (embed_dim,) - Replicated
  pre_attention_norm_scale: Array  # (embed_dim,) - Replicated
  pre_ffw_norm_scale: Array  # (embed_dim,) - Replicated


class Gemma3(NamedTuple):
  # (vocab_size, embed_dim) -> Shard embed_dim
  input_embedding_table: Array
  # Ignored mm fields for text-only
  mm_input_projection: Array
  mm_soft_embedding_norm: Array
  final_norm_scale: Array  # (embed_dim,) - Replicated
  blocks: list[Layer]  # list of Layers


# --- Create Gemma3 model ---
def initialize_model(params: FlatDict, num_layers: int, config: Config | None = None) -> Gemma3:
  """Load and instantiate the Gemma3 model. This is essential an ordered dictionary of parameters."""
  return Gemma3(
      params["transformer.embedder.input_embedding"],
      params["transformer.embedder.mm_input_projection.w"],
      params["transformer.embedder.mm_soft_embedding_norm.scale"],
      params["transformer.final_norm.scale"],
      [
          Layer(
              params[f"transformer.layer_{idx}.attn._key_norm.scale"],
              params[f"transformer.layer_{idx}.attn._query_norm.scale"],
              params[f"transformer.layer_{idx}.attn.attn_vec_einsum.w"],
              params[f"transformer.layer_{idx}.attn.kv_einsum.w"],
              params[f"transformer.layer_{idx}.attn.q_einsum.w"],
              params[f"transformer.layer_{idx}.mlp.gating_einsum.w"],
              params[f"transformer.layer_{idx}.mlp.linear.w"],
              params[f"transformer.layer_{idx}.post_attention_norm.scale"],
              params[f"transformer.layer_{idx}.post_ffw_norm.scale"],
              params[f"transformer.layer_{idx}.pre_attention_norm.scale"],
              params[f"transformer.layer_{idx}.pre_ffw_norm.scale"],
          )
          for idx in range(num_layers)
      ],
  )


# --- Param/ Pytree Utility Functions ---
def handle_key(k: Any) -> str:
  """Extract a string representation from various key types used in Pytrees."""
  if hasattr(k, "idx"):
    return str(k.idx)
  elif hasattr(k, "key"):
    return k.key
  elif hasattr(k, "name"):
    return k.name
  else:
    return str(k)


def get_key(k: Union[tuple[Any, ...], Any], replace_str: str = "", with_str: str = "") -> str:
  key = ".".join([handle_key(x) for x in k]) if isinstance(k, tuple) else handle_key(k)
  return key.replace(replace_str, with_str) if replace_str else key


def flatten_nested_pytree(tree: Any) -> FlatDict:
  flat, _ = jax.tree_util.tree_flatten_with_path(tree)
  return {get_key(k): v for k, v in flat}


# --- PartitionSpecs for Model ---
def get_params_partition_spec_as_dict(config: Config) -> dict[str, Any]:
  """Get the PartitionSpec Pytree matching the raw params dictionary."""
  embed_spec = P(None, "model")
  norm_spec = P()  # Replicated

  # For each layer in the transformer block
  layer_spec = {
      "attn_key_norm_scale": norm_spec,
      "attn_query_norm_scale": norm_spec,
      "output_proj": P(None, None, "model"),
      "kv_proj": P(None, None, "model", None),  # [2,K,D,H]
      "q_proj": P(None, "model", None),  # [H,D,K]
      "gating_weights": P(None, "model", None),  # [2,F,D]
      "output_weights": P("model", None),
      "post_attention_norm_scale": norm_spec,
      "post_ffw_norm_scale": norm_spec,
      "pre_attention_norm_scale": norm_spec,
      "pre_ffw_norm_scale": norm_spec,
  }
  # Build a pytree matching your Gemma3 named tuple structure.
  params_spec = {
      "input_embedding_table": embed_spec,
      "mm_input_projection": embed_spec,
      "mm_soft_embedding_norm": norm_spec,
      "final_norm_scale": norm_spec,
      "blocks": [layer_spec] * config.num_layers,
  }
  return params_spec


def remap_params(params: FlatDict, num_layers: int, config: Config | None = None) -> NestedDict:
  """Remap the raw flat params dictionary into a shallowly nested parameters dictionary."""

  p = {}
  p["input_embedding_table"] = params["transformer.embedder.input_embedding"]
  p["mm_input_projection"] = params["transformer.embedder.mm_input_projection.w"]
  p["mm_soft_embedding_norm"] = params["transformer.embedder.mm_soft_embedding_norm.scale"]
  p["final_norm_scale"] = params["transformer.final_norm.scale"]
  p["blocks"] = [
      {
          "attn_key_norm_scale": params[f"transformer.layer_{idx}.attn._key_norm.scale"],
          "attn_query_norm_scale": params[f"transformer.layer_{idx}.attn._query_norm.scale"],
          "output_proj": params[f"transformer.layer_{idx}.attn.attn_vec_einsum.w"],
          "kv_proj": params[f"transformer.layer_{idx}.attn.kv_einsum.w"],
          "q_proj": params[f"transformer.layer_{idx}.attn.q_einsum.w"],
          "gating_weights": params[f"transformer.layer_{idx}.mlp.gating_einsum.w"],
          "output_weights": params[f"transformer.layer_{idx}.mlp.linear.w"],
          "post_attention_norm_scale": params[f"transformer.layer_{idx}.post_attention_norm.scale"],
          "post_ffw_norm_scale": params[f"transformer.layer_{idx}.post_ffw_norm.scale"],
          "pre_attention_norm_scale": params[f"transformer.layer_{idx}.pre_attention_norm.scale"],
          "pre_ffw_norm_scale": params[f"transformer.layer_{idx}.pre_ffw_norm.scale"],
      }
      for idx in range(num_layers)
  ]
  return p


def get_params_partition_spec(config: Config) -> Gemma3:
  """Get the PartitionSpec Pytree matching the Gemma3 NamedTuple."""
  # Shard embedding dim ('model'), replicate norms ()
  embed_spec = P(None, "model")
  norm_spec = P()  # Replicated

  layer_spec = Layer(
      attn_key_norm_scale=norm_spec,  # Replicate (head_dim,)
      attn_query_norm_scale=norm_spec,  # Replicate (head_dim,)
      output_proj=P(None, None, "model"),  # Shard embed_dim (num_heads, head_dim, embed_dim)
      kv_proj=P(None, None, "model", None),  # Shard embed_dim(D) (2, K, D, H)
      q_proj=P(None, "model", None),  # Shard hidden_dim (2, hidden_dim, embed_dim)
      gating_weights=P(None, "model", None),  # Shard hidden_dim (2, hidden_dim, embed_dim)
      output_weights=P("model", None),  # Shard hidden_dim (hidden_dim, embed_dim)
      post_attention_norm_scale=norm_spec,  # Replicate (embed_dim,)
      post_ffw_norm_scale=norm_spec,  # Replicate (embed_dim,)
      pre_attention_norm_scale=norm_spec,  # Replicate (embed_dim,)
      pre_ffw_norm_scale=norm_spec,  # Replicate (embed_dim,)
  )
  return Gemma3(
      # Shard embed_dim (vocab_size, embed_dim)
      input_embedding_table=embed_spec,
      final_norm_scale=norm_spec,  # Replicate (embed_dim,)
      mm_input_projection=embed_spec,  # Replicate (embed_dim, embed_dim)
      mm_soft_embedding_norm=norm_spec,  # Replicate (embed_dim,)
      blocks=[layer_spec] * config.num_layers,
  )


# --- Device Mesh Creation ---
def create_device_mesh(
    mesh_shape: tuple[int | None, int | None] | None = None,
    axis_names: tuple[str, str] = ("data", "model"),
) -> Mesh:
  """
  Build a 2-D Mesh that is TPU agnostic.

  Examples:
  create_device_mesh()                 # auto for v2-8, v4-32, v4-64
  create_device_mesh((8, 1))           # data-parallel only
  create_device_mesh((None, 4))        # force 4-way model parallelism
  """
  devices = jax.devices()
  num_devices = len(devices)

  if mesh_shape is None or mesh_shape == (None, None):
    # Factorization
    side = int(math.sqrt(num_devices))
    while side > 1 and num_devices % side:
      side -= 1
    batch_axis = side
    model_axis = num_devices // side
  else:
    batch_axis, model_axis = mesh_shape

    # Fill-in any None using what’s left.
    if batch_axis is None and model_axis is None:
      raise ValueError("mesh_shape cannot be (None, None); use None instead.")

    if batch_axis is None:
      if num_devices % model_axis:
        raise ValueError(f"Cannot put {num_devices} devices into (_, {model_axis}).")
      batch_axis = num_devices // model_axis

    if model_axis is None:
      if num_devices % batch_axis:
        raise ValueError(f"Cannot put {num_devices} devices into ({batch_axis}, _).")
      model_axis = num_devices // batch_axis

  required = batch_axis * model_axis
  if required > num_devices:
    raise ValueError(
        f"Need {required} devices for mesh ({batch_axis}×{model_axis}), " f"but only {num_devices} are available."
    )

  mesh_array = np.array(devices[:required]).reshape((batch_axis, model_axis))

  print(
      f"Creating mesh {mesh_array.shape} on {jax.device_count()} physical devices "
      f"(hosts={jax.process_count()}) → axes{axis_names}"
  )
  return Mesh(mesh_array, axis_names=axis_names)


# --- Model Loading ---
def load_and_format_raw_params(path: str, load_siglip: bool = False, dtype: jnp.dtype | None = None) -> FlatDict:
  """Load parameters from a checkpoint and returns a flat dictionary.

  Remaps keys (replacing "/" with ".") and optionally excludes SigLIP keys.
  Optionally casts parameters to the specified dtype.
  """

  @functools.cache
  def _load_params(p: str) -> Any:
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(p)

  def _remap_fn(key: str) -> str:
    return key.replace("/", ".")

  print(f"Loading raw parameters from: {path}")
  params = _load_params(path)
  flat_params = flatten_nested_pytree(params)

  if not load_siglip:
    print("Removing top-level 'SigLiPFromPatches_0' key.")
    flat_params = {k: v for k, v in flat_params.items() if not k.startswith("SigLiPFromPatches_0")}

  remapped_params = {_remap_fn(k): v for k, v in flat_params.items()}

  if dtype:
    print(f"Casting parameters to {dtype}...")
    start_cast_time = time.time()
    remapped_params = jax.tree_util.tree_map(lambda x: x.astype(dtype), remapped_params)
    print(f"Parameter casting complete in {time.time() - start_cast_time:.2f}s")

  return remapped_params


def load_model(path: str, mesh: Mesh, config: Config, dtype: jnp.dtype | None = None) -> Gemma3:
  """Load parameters from checkpoint, structures them into a Gemma3 NamedTuple,.

  and shards the model across the provided device mesh.
  Args:
      path: Path to the checkpoint directory.
      mesh: The JAX device mesh for sharding.
      config: The model configuration.
      dtype: Optional dtype to cast parameters to (e.g., jnp.bfloat16).

  Returns:
      The sharded Gemma3 model Pytree.
  Raises:
      ValueError: If parameter structuring fails.
  """
  start_time = time.time()
  raw_flat_params = load_and_format_raw_params(path, load_siglip=False, dtype=dtype)
  print(f"Parameter loading complete in {time.time() - start_time:.2f}s")

  start_time = time.time()
  try:
    model_tree = initialize_model(raw_flat_params, num_layers=config.num_layers)
  except ValueError as e:
    print(f"Parameter conversion failed: {e}")
    raise
  print(f"Model structuring complete in {time.time() - start_time:.2f}s")

  params_spec = get_params_partition_spec(config)
  target_sharding = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), params_spec)

  print("Sharding model...")
  start_time = time.time()

  def _shard(pytree: Any) -> Any:
    return jax.device_put(pytree, target_sharding)

  with mesh:
    sharded_model = _shard(model_tree)
    # Ensure the embedding table is ready
    sharded_model.input_embedding_table.block_until_ready()
  print(f"Model sharding complete in {time.time() - start_time:.2f}s")
  return sharded_model


def load_params(path: str, mesh: Mesh, config: Config, dtype: jnp.dtype | None = None) -> NestedDict:
  """Load parameters from checkpoint, structures them into a shallowly nested parameters dictionary,
  and shards the parameters across the provided device mesh.

  Args:
      path: Path to the checkpoint directory.
      mesh: The JAX device mesh for sharding.
      config: The model configuration.
      dtype: Optional dtype to cast parameters to (e.g., jnp.bfloat16).

  Returns:
      The sharded parameters dictionary.
  Raises:
      ValueError: If parameter structuring fails.
  """
  start_time = time.time()
  raw_flat_params = load_and_format_raw_params(path, load_siglip=False, dtype=dtype)
  print(f"Parameter loading complete in {time.time() - start_time:.2f}s")

  start_time = time.time()
  try:
    params_tree = remap_params(raw_flat_params, num_layers=config.num_layers)
  except ValueError as e:
    print(f"Parameter conversion failed: {e}")
    raise
  print(f"Parameter structuring complete in {time.time() - start_time:.2f}s")

  params_spec = get_params_partition_spec_as_dict(config)
  target_sharding = jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), params_spec)

  print("Sharding parameters...")
  start_time = time.time()

  def _shard(pytree: Any) -> Any:
    return jax.device_put(pytree, target_sharding)

  with mesh:
    params = _shard(params_tree)
    # Ensure the embedding table is ready
    params["input_embedding_table"].block_until_ready()
  print(f"Parameter sharding complete in {time.time() - start_time:.2f}s")
  return params
