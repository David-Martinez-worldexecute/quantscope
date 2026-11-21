"""Weight formats used in a run.

Every mode returns a float32 array with the same shape as its input, so the
forward pass is identical apart from the weights. Quantization is simulated in
floating point on purpose: the question is "what does this format cost in
accuracy", not "how fast is a kernel".
"""

from __future__ import annotations

import numpy as np

MODES = ("fp32", "int4", "ternary")


def _as_float32(weights) -> np.ndarray:
    array = np.asarray(weights, dtype=np.float32)
    if array.size == 0:
        raise ValueError("cannot quantize an empty array")
    return array


def rtn(weights, bits: int = 4, group_size: int = 64) -> np.ndarray:
    """Round-to-nearest, group-wise symmetric quantization.

    One scale per group of ``group_size`` contiguous weights, computed from the
    group maximum. Integer codes are immediately dequantized, so the returned
    array is the weight matrix a kernel would see at inference time.
    """
    original = _as_float32(weights)
    flat = original.reshape(-1).copy()
    total = flat.size
    group = max(1, int(group_size))
    pad = (-total) % group
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    blocks = flat.reshape(-1, group)
    qmax = float(2 ** (bits - 1) - 1)
    scale = np.maximum(np.abs(blocks).max(axis=1, keepdims=True) / qmax, 1e-8)
    codes = np.clip(np.rint(blocks / scale), -qmax, qmax)
    restored = (codes * scale).reshape(-1)[:total]
    return restored.reshape(original.shape).astype(np.float32)


def ternary(weights, keep_ratio: float = 0.7) -> np.ndarray:
    """Ternary weights: each value becomes one of {-scale, 0, +scale}.

    ``scale`` is the mean absolute weight and the zero band is
    ``keep_ratio * scale``, so weights close to zero are dropped instead of
    being forced to a sign. The formula is recorded in the manifest docs so two
    "ternary" runs can be told apart.
    """
    original = _as_float32(weights)
    flat = original.reshape(-1)
    scale = float(np.mean(np.abs(flat)))
    if scale <= 0.0:
        return np.zeros_like(original)
    threshold = float(keep_ratio) * scale
    codes = np.where(flat > threshold, 1.0, np.where(flat < -threshold, -1.0, 0.0))
    return (codes * scale).reshape(original.shape).astype(np.float32)


def quantize(weights, mode: str, group_size: int = 64) -> np.ndarray:
    """Dispatch to the named mode. ``fp32`` is an explicit copy, not a no-op."""
    if mode == "fp32":
        return _as_float32(weights).copy()
    if mode == "int4":
        return rtn(weights, bits=4, group_size=group_size)
    if mode == "ternary":
        return ternary(weights)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")


def describe(mode: str, group_size: int = 64) -> dict:
    """Metadata written into the manifest so a mode is reproducible from the text."""
    if mode == "fp32":
        return {"mode": "fp32", "formula": "identity"}
    if mode == "int4":
        return {
            "mode": "int4",
            "formula": "group-wise symmetric round-to-nearest, codes in [-7, 7], dequantized to float32",
            "group_size": int(group_size),
        }
    if mode == "ternary":
        return {
            "mode": "ternary",
            "formula": "codes in {-1, 0, +1}; scale = mean(|w|); zero band = 0.7 * scale",
            "keep_ratio": 0.7,
        }
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
