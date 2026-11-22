"""Ternary weights whose scale groups follow the *input* structure.

The published result this reproduces: on a two-layer Forward-Forward network the
first layer sees ``[784 pixels | 10 one-hot label columns]``. The ten label
columns carry much larger weight norms than the pixel columns, so if a single
scale is shared across the whole row, the pixel weights get rounded to zero and
accuracy drops. Giving each label column its own scale (and keeping the pixel
block as one group) wins most of that drop back.

Two modes, and the only difference between them is the grouping:

``row``
    One scale per output row, computed over the whole input row — the BitNet
    convention, and what a quantiser does when it does not know the input is
    heterogeneous.
``aligned``
    One scale per output row *per input group*: the 784 pixel columns are one
    group, and each of the 10 label columns is its own group (11 groups).

Everything else (weights, data order, learning rate, epochs) is identical, so a
difference between the two runs is attributable to the grouping alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PIXELS = 28 * 28
LABELS = 10
SPANS_ALIGNED = [(0, PIXELS)] + [(PIXELS + index, PIXELS + index + 1) for index in range(LABELS)]


def spans_for(mode: str) -> list[tuple[int, int]] | None:
    """Return the input-column spans for ``mode``; ``None`` means one group per row."""
    if mode == "row":
        return None
    if mode == "aligned":
        return list(SPANS_ALIGNED)
    raise ValueError(f"unknown group mode {mode!r} (expected 'row' or 'aligned')")


def quantize_grouped(
    weight: torch.Tensor,
    spans: list[tuple[int, int]] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ternarise ``weight`` and return ``(quantized, scales)``.

    ``scales`` has shape ``(out_features, n_groups)``; with ``spans=None`` it is
    ``(out_features, 1)``. No bias, no per-tensor fallback, no clipping tricks —
    ``round`` to ``-1 / 0 / +1`` and multiply the scale back.
    """
    if spans is None:
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        quantized = torch.clamp(torch.round(weight / scale), -1.0, 1.0) * scale
        return quantized, scale

    pieces = []
    scales = []
    covered = 0
    for start, stop in spans:
        if start != covered:
            raise ValueError(f"spans must tile the input, gap or overlap at column {covered}")
        block = weight[:, start:stop]
        scale = block.abs().mean(dim=1, keepdim=True).clamp_min(eps)
        pieces.append(torch.clamp(torch.round(block / scale), -1.0, 1.0) * scale)
        scales.append(scale)
        covered = stop
    if covered != weight.size(1):
        raise ValueError(f"spans cover {covered} of {weight.size(1)} input columns")
    return torch.cat(pieces, dim=1), torch.cat(scales, dim=1)


def ternary_ste(weight: torch.Tensor, spans=None) -> torch.Tensor:
    """Quantise on the way forward, identity gradient on the way back."""
    quantized, _ = quantize_grouped(weight, spans)
    return weight + (quantized - weight).detach()


class TernaryLinear(nn.Linear):
    """``nn.Linear`` that ternarises its weight in the forward pass."""

    def __init__(self, in_features: int, out_features: int, spans=None) -> None:
        super().__init__(in_features, out_features)
        self.spans = spans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, ternary_ste(self.weight, self.spans), self.bias)


def zero_fraction(weight: torch.Tensor, spans=None) -> float:
    """Share of quantised weights that land exactly on zero.

    Reported in the manifest because it is the mechanism in one number: sharing a
    scale with the label columns pushes this up.
    """
    quantized, _ = quantize_grouped(weight, spans)
    return float((quantized == 0).float().mean().item())
