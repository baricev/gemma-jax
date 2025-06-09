# %% 

"""gemma3_sharding_refactored.py
================================
A JAX-idiomatic helper module for sharding **Gemma-3** checkpoints.

This streamlined version
* keeps *no-boiler-plate* pytree compatibility by using plain
  `typing.NamedTuple` for both **PartitionSpec** trees and runtime model
  states, and
* exposes a tiny public API (`create_config`, `create_device_mesh`,
  `load_model`, `load_params`).

Place the file anywhere on your `$PYTHONPATH`, `pip install orbax` for
checkpoint restore support, and you can immediately shard Gemma-3
checkpoints across CPU, GPU, or TPU meshes.
"""
from __future__ import annotations

import functools
import logging
import math
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

__all__ = [
    "Config",
    "create_config",
    "create_device_mesh",
    "load_model",
    "load_params",
]

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s] %(message)s")
_logger = logging.getLogger("gemma3.shard")

# -----------------------------------------------------------------------------
# Constants & enums
# -----------------------------------------------------------------------------

class SpecialToken(IntEnum):
    PAD = 0
    EOS = 1


class AttentionType(IntEnum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


# Base six-layer attention pattern ------------------------------------------------
_BASE_PATTERN: tuple[AttentionType, ...] = (
    AttentionType.LOCAL_SLIDING,
) * 5 + (AttentionType.GLOBAL,)

# Gemma-3 variant table -----------------------------------------------------------
# fmt: off
_GEMMA3_VARIANTS = np.array([
    #  1B   4B    12B    27B
    [   1,   4,    12,    27],   # model size (B)
    [  26,  34,    48,    62],   # layers
    [   4,   8,    16,    32],   # heads
    [   1,   4,     8,    16],   # kv-heads
    [1152, 2560,  3840,  5376],  # embed-dim
    [6912,10240, 15360, 21504],  # mlp-hidden_dim (6*9*128, 4*20*128, 4*30*128, 4*42*128)
    [ 256,  256,   256,   128],  # head-dim
], dtype=np.int32)
# fmt: on

_ROW = {
    "model_size": 0,
    "num_layers": 1,
    "num_heads": 2,
    "num_kv_heads": 3,
    "embed_dim": 4,
    "hidden_dim": 5,
    "head_dim": 6,
}

# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Config:
    # Architecture
    model_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    hidden_dim: int
    head_dim: int

    # Runtime
    batch_size: int
    window_size: int
    cache_length: int
    chunk_length: int
    # max_gen_length: int
    generate_steps: int

    # Attention
    query_pre_attn_scalar: float
    attention_pattern: tuple[AttentionType, ...] = _BASE_PATTERN

    # Vocab / logits
    vocab_size: int = 262_144
    final_logit_softcap: float | None = None

    # Norm flags
    use_post_attn_norm: bool = True
    use_post_ffw_norm: bool = True
    use_qk_norm: bool = True

    # Rotary cache
    use_rope_cache: bool = False
    rope_cache: Array | None = None

    # Extra (vision etc.)
    vision_encoder: tuple[int, ...] | None = None
    mm_extra_vocab_size: int = 0

    # Frequencies
    local_base_frequency: int = 10_000
    local_scale_factor: float = 1.0
    global_base_frequency: int = 1_000_000
    global_scale_factor: float = 8.0

    # Misc
    transpose_gating_einsum: bool = True
    attn_logits_soft_cap: float | None = None

    # # Alias for back-compat
    # @property
    # def attention_types(self):
    #     return self.attention_pattern

# -----------------------------------------------------------------------------
# Config factory
# -----------------------------------------------------------------------------

def _tile_pattern(n: int) -> tuple[AttentionType, ...]:
    rep, rem = divmod(n, len(_BASE_PATTERN))
    return _BASE_PATTERN * rep + _BASE_PATTERN[:rem]

def get_query_pre_attn_norm(model_size: int) -> float:
  """Calculate the pre-attention normalization scalar for the query.

  Scales by '1/sqrt(embed_dim // num_heads)' if 27B model,
  otherwise by '1/sqrt(head_dim=256)'.
  """
  head_dim_27b = 128
  embed_dim_27b = 5376
  num_heads_27b = 32
  return (embed_dim_27b / num_heads_27b) ** -0.5 if model_size == 27 else (head_dim_27b * 2) ** -0.5


def create_config(*, model_size: int, batch_size: int, chunk_length: int,
                  window_size: int, cache_length: int, generate_steps: int,
                  **kw) -> Config:
    """Return a fully specified :class:`Config`.  Extra `kw` override defaults."""
    if model_size not in (1, 4, 12, 27):
        raise ValueError("Model size must be 1, 4, 12, or 27 (billions).")

    col = int(np.where(_GEMMA3_VARIANTS[_ROW["model_size"]] == model_size)[0][0])
    base = {k: int(_GEMMA3_VARIANTS[row, col]) for k, row in _ROW.items() if k != "model_size"}

    # q_scalar = ( (base["embed_dim"] / base["num_heads"]) ** -0.5 if model_size == 27 else (256 * 2) ** -0.5) #BUG
    q_scalar = get_query_pre_attn_norm(model_size)

    cfg = Config(
        model_size=model_size,
        batch_size=batch_size,
        chunk_length=chunk_length,
        window_size=window_size,
        cache_length=cache_length,
        # max_gen_length=max_gen_length,
        generate_steps=generate_steps,
        attention_pattern=_tile_pattern(base["num_layers"]),
        query_pre_attn_scalar=q_scalar,
        **base,
    )
    return replace(cfg, **kw)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_T = TypeVar("_T")
FlatDict = dict[str, Array]
NestedDict = dict[str, Any]


def _flatten_pytree_with_dots(tree) -> FlatDict:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)

    def _p2s(p):
        parts: list[str] = []
        for k in p:
            parts.append(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))))
        return ".".join(parts)

    return {_p2s(path): v for path, v in flat}

# -----------------------------------------------------------------------------
# Device mesh creation
# -----------------------------------------------------------------------------

def create_device_mesh(shape: tuple[int | None, int | None] | None = None, *, axis_names=("data", "model")) -> Mesh:
    devs = jax.devices()
    if {d.platform for d in devs} == {"cpu"}:  # trivial 1×1 mesh on CPU
        return Mesh(np.array(devs).reshape((1, 1)), axis_names=axis_names)

    n = len(devs)
    if shape is None or shape == (None, None):  # heuristic square factor
        side = int(math.sqrt(n))
        while side > 1 and n % side:
            side -= 1
        data, model = side, n // side
    else:
        data, model = shape
        if data is None and model is None:
            raise ValueError("shape cannot be (None, None)")
        if data is None:
            if n % model:
                raise ValueError(f"{n} devs can't fit (_, {model})")
            data = n // model
        if model is None:
            if n % data:
                raise ValueError(f"{n} devs can't fit ({data}, _)")
            model = n // data
    if data * model > n:
        raise ValueError("Mesh larger than available devices")

    mesh = Mesh(np.array(devs[: data * model]).reshape((data, model)), axis_names=axis_names)
    _logger.info("Mesh %dx%d constructed on %d devices", data, model, n)
    return mesh

# -----------------------------------------------------------------------------
# Checkpoint loading utils
# -----------------------------------------------------------------------------

@functools.cache
def _restore_ckpt(path: str | Path):
    return ocp.PyTreeCheckpointer().restore(str(path))


def _load_flat_params(path: str | Path, *, dtype: jnp.dtype | None = None, keep_siglip: bool = False) -> FlatDict:
    _logger.info("Reading checkpoint %s", path)
    raw = _restore_ckpt(path)
    flat = _flatten_pytree_with_dots(raw)
    if not keep_siglip:
        flat = {k: v for k, v in flat.items() if not k.startswith("SigLiPFromPatches_0")}
    if dtype:
        flat = jax.tree_util.tree_map(lambda x: x.astype(dtype), flat)
    return {k.replace("/", "."): v for k, v in flat.items()}

# -----------------------------------------------------------------------------
# Gemma-3 runtime structures
# -----------------------------------------------------------------------------

# fmt: off
class Layer(NamedTuple):
    attn_key_norm_scale         :  Array   # (head_dim,)                             (H,)
    attn_query_norm_scale       :  Array   # (head_dim,)                             (H,)
    output_proj                 :  Array   # (num_heads, head_dim, embed_dim)        (N,H,D)
    kv_proj                     :  Array   # (2, num_kv_heads, embed_dim, head_dim)  (2,K,D,H)
    q_proj                      :  Array   # (num_heads, embed_dim, head_dim)        (N,D,H)
    gating_weights              :  Array   # (2, mlp_hidden_dim, embed_dim)          (2,F,D)
    output_weights              :  Array   # (mlp_hidden_dim, embed_dim)             (F,D)
    post_attention_norm_scale   :  Array   # (embed_dim,)                            (D,)
    post_ffw_norm_scale         :  Array   # (embed_dim,)                            (D,)
    pre_attention_norm_scale    :  Array   # (embed_dim,)                            (D,)
    pre_ffw_norm_scale          :  Array   # (embed_dim,)                            (D,)


class Gemma3(NamedTuple):
    input_embedding_table       :  Array   # (vocab_size, embed_dim)                 (V,D)
    mm_input_projection         :  Array   # (embed_dim, embed_dim)                  (D,D)
    mm_soft_embedding_norm      :  Array   # (embed_dim,)                            (D,)
    final_norm_scale             :  Array   # (embed_dim,)                            (D,)
    blocks                      :  tuple[Layer, ...]
# fmt: on

# -----------------------------------------------------------------------------
# Param-to-pytree conversions
# -----------------------------------------------------------------------------

def _build_model_pytree(params: FlatDict, *, num_layers: int) -> Gemma3:
    def _layer(idx: int) -> Layer:
        p = f"transformer.layer_{idx}."
        return Layer(
            params[p + "attn._key_norm.scale"],
            params[p + "attn._query_norm.scale"],
            params[p + "attn.attn_vec_einsum.w"],
            params[p + "attn.kv_einsum.w"],
            params[p + "attn.q_einsum.w"],
            params[p + "mlp.gating_einsum.w"],
            params[p + "mlp.linear.w"],
            params[p + "post_attention_norm.scale"],
            params[p + "post_ffw_norm.scale"],
            params[p + "pre_attention_norm.scale"],
            params[p + "pre_ffw_norm.scale"],
        )

    return Gemma3(
        params["transformer.embedder.input_embedding"],
        params["transformer.embedder.mm_input_projection.w"],
        params["transformer.embedder.mm_soft_embedding_norm.scale"],
        params["transformer.final_norm.scale"],
        tuple(_layer(i) for i in range(num_layers)),
    )


# fmt: off
def _build_shallow_dict(params: FlatDict, *, num_layers: int) -> NestedDict:
    """Return a 2-level dict mirroring the NamedTuple but easy to JSON-dump."""
    out: NestedDict = {
        "input_embedding_table"         : params["transformer.embedder.input_embedding"],
        "mm_input_projection"           : params["transformer.embedder.mm_input_projection.w"],
        "mm_soft_embedding_norm"        : params["transformer.embedder.mm_soft_embedding_norm.scale"],
        "final_norm_scale"               : params["transformer.final_norm.scale"],
        "blocks"                        : [],
    }
    for i in range(num_layers):
        p = f"transformer.layer_{i}."
        out["blocks"].append({
            "attn_key_norm_scale"       : params[p + "attn._key_norm.scale"],
            "attn_query_norm_scale"     : params[p + "attn._query_norm.scale"],
            "output_proj"               : params[p + "attn.attn_vec_einsum.w"],
            "kv_proj"                   : params[p + "attn.kv_einsum.w"],
            "q_proj"                    : params[p + "attn.q_einsum.w"],
            "gating_weights"            : params[p + "mlp.gating_einsum.w"],
            "output_weights"            : params[p + "mlp.linear.w"],
            "post_attention_norm_scale" : params[p + "post_attention_norm.scale"],
            "post_ffw_norm_scale"       : params[p + "post_ffw_norm.scale"],
            "pre_attention_norm_scale"  : params[p + "pre_attention_norm.scale"],
            "pre_ffw_norm_scale"        : params[p + "pre_ffw_norm.scale"],
        })
    return out
# fmt: on

# -----------------------------------------------------------------------------
# Sharding wrappers
# -----------------------------------------------------------------------------

def _device_put_with_spec(pytree: _T, spec_tree: _T, mesh: Mesh) -> _T:
    sharding = jax.tree_util.tree_map(lambda s: NamedSharding(mesh, s), spec_tree)
    return jax.device_put(pytree, sharding)

# -----------------------------------------------------------------------------
# Public loading API
# -----------------------------------------------------------------------------

def load_model(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3:
    """Load → build NamedTuple → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree(flat, num_layers=cfg.num_layers)
    spec_pt = _model_spec(cfg)

    _logger.info("Sharding model …")
    t0 = time.perf_counter()
    with mesh:
        sharded = _device_put_with_spec(model_pt, spec_pt, mesh)
        sharded.input_embedding_table.block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return sharded


def load_params(path: str | Path, mesh: Mesh, cfg: Config, *, dtype: jnp.dtype | None = None) -> NestedDict:
    """Load → shallow dict → shard → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    nested = _build_shallow_dict(flat, num_layers=cfg.num_layers)
    pspec = _param_spec_dict(cfg)

    _logger.info("Sharding param-dict …")
    t0 = time.perf_counter()
    with mesh:
        sharded = _device_put_with_spec(nested, pspec, mesh)
        sharded["input_embedding_table"].block_until_ready()
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return sharded

def load_unsharded_model(path: str | Path, cfg: Config, *, dtype: jnp.dtype | None = None) -> Gemma3:
    """Load → build NamedTuple → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    model_pt = _build_model_pytree(flat, num_layers=cfg.num_layers)
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return model_pt

def load_unsharded_params(path: str | Path, cfg: Config, *, dtype: jnp.dtype | None = None, flatten: bool=False) -> NestedDict:
    """Load → shallow dict  → return."""
    t0 = time.perf_counter()
    flat = _load_flat_params(path, dtype=dtype)
    _logger.info("Params loaded in %.2fs", time.perf_counter() - t0)

    nested = _build_shallow_dict(flat, num_layers=cfg.num_layers)
    if flatten:
        flattened = _flatten_pytree_with_dots(nested)
        _logger.info("Flattened to %d keys", len(flattened))
        _logger.info("Done (%.2fs)", time.perf_counter() - t0)

        return flattened
    _logger.info("Nested dict with %d keys", len(nested))
    _logger.info("Done (%.2fs)", time.perf_counter() - t0)
    return nested



# -----------------------------------------------------------------------------
# PartitionSpec templates (NamedTuple PYTree)
# -----------------------------------------------------------------------------

class LayerPSpec(NamedTuple):
    attn_key_norm_scale: P
    attn_query_norm_scale: P
    output_proj: P
    kv_proj: P
    q_proj: P
    gating_weights: P
    output_weights: P
    post_attention_norm_scale: P
    post_ffw_norm_scale: P
    pre_attention_norm_scale: P
    pre_ffw_norm_scale: P


class Gemma3PSpec(NamedTuple):
    input_embedding_table: P
    mm_input_projection: P
    mm_soft_embedding_norm: P
    final_norm_scale: P
    blocks: tuple[LayerPSpec, ...]


# lazy constants
_norm = P()
_emb = P(None, "model")


def _layer_spec() -> Layer:
    return Layer(
        _norm,
        _norm,
        P(None, None, "model"),
        P(None, None, "model", None),
        P(None, "model", None),
        P(None, "model", None),
        P("model", None),
        _norm,
        _norm,
        _norm,
        _norm,
    )


def _model_spec(cfg: Config) -> Gemma3:
    return Gemma3(_emb, _emb, _norm, _norm, tuple(_layer_spec() for _ in range(cfg.num_layers)))


# -----------------------------------------------------------------------------
# Param-spec as dict (for load_params)
# -----------------------------------------------------------------------------

# fmt: off
def _param_spec_dict(cfg: Config) -> NestedDict:
    norm, emb = _norm, _emb
    layer = {
        "attn_key_norm_scale"       : norm,
        "attn_query_norm_scale"     : norm,
        "output_proj"               : P(None, None, "model"),
        "kv_proj"                   : P(None, None, "model", None),
        "q_proj"                    : P(None, "model", None),
        "gating_weights"            : P(None, "model", None),
        "output_weights"            : P("model", None),
        "post_attention_norm_scale" : norm,
        "post_ffw_norm_scale"       : norm,
        "pre_attention_norm_scale"  : norm,
        "pre_ffw_norm_scale"        : norm,
    }
    return {
        "input_embedding_table"     : emb,
        "mm_input_projection"       : emb,
        "mm_soft_embedding_norm"    : norm,
        "final_norm_scale"           : norm,
        "blocks"                    : [layer] * cfg.num_layers,
    }
# fmt: on
# %%
