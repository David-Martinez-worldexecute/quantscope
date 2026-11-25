# -*- coding: utf-8 -*-
"""The ternary quantiser must honour the group size it is given.

Two defects were found in the published ablation bundles on 2026-11-25, both of the
"parameter never reached the code that used it" kind:

  * the statistics function hard-coded group size 64 inside its body, so every bundle
    reported group-64 numbers no matter which group size the run actually used;
  * it was called while the model was still quantised, so it ternarised an already
    ternarised weight and reported a deflated error (0.3053 instead of 0.5078 on a
    64x128 tensor at group 64).

This file keeps a standalone copy of the quantiser so the public repository can test
the property without importing anything from the lab directory, and it asserts the
statistics against an independent one-shot computation rather than against a shape.
"""
from __future__ import annotations

import unittest

import torch

GROUP_SIZE = 64


def ternary(weight: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Copy of the lab quantiser: per-group absmean scale, round to -1, 0 or +1.

    The group runs along the last dimension; a group whose mean magnitude is zero is
    left at zero rather than divided by it.
    """
    flat = weight.detach().reshape(-1, group_size)
    scale = flat.abs().mean(dim=1, keepdim=True)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantised = torch.round(flat / safe).clamp(-1, 1) * safe
    zero_scale = scale == 0
    if bool(zero_scale.any()):
        quantised = torch.where(zero_scale, torch.zeros_like(quantised), quantised)
    return quantised.reshape(weight.shape)


def statistics(weight: torch.Tensor, group_size: int) -> tuple[float, float]:
    """Zero fraction and relative error of one honest quantisation of the original weight."""
    rebuilt = ternary(weight, group_size)
    zero = float((rebuilt == 0).float().mean())
    error = float((weight - rebuilt).norm() / weight.norm())
    return zero, error


class TernaryGroupSizeTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.weight = torch.randn(64, 128, generator=generator)

    def test_group_size_actually_changes_the_error(self) -> None:
        errors = [statistics(self.weight, group)[1] for group in (32, 64, 128)]
        self.assertEqual(len(set(errors)), len(errors),
                         f"group size did not change the error: {errors}")

    def test_error_stays_near_a_half_across_these_widths(self) -> None:
        # The published claim: the cliff is not explained by coarser groups destroying
        # more of the weight, because the error barely moves across 32 / 64 / 128.
        for group in (32, 64, 128):
            with self.subTest(group=group):
                error = statistics(self.weight, group)[1]
                self.assertGreater(error, 0.45)
                self.assertLess(error, 0.58)

    def test_zero_fraction_is_stable_across_widths(self) -> None:
        zeros = [statistics(self.weight, group)[0] for group in (32, 64, 128)]
        self.assertLess(max(zeros) - min(zeros), 0.05,
                        f"zero fraction swung between widths: {zeros}")

    def test_quantising_twice_reports_a_smaller_error(self) -> None:
        """The direction of the bug, recorded so it cannot be mis-stated again.

        Re-quantising deflates rather than inflates the number: the denominator becomes
        ||q|| instead of ||w||, and the second rounding step is small. Measured here at
        group 64 the honest value is about 0.51 and the double-quantised value about 0.31,
        which looks like a plausible one-shot error, and is why it went unnoticed.
        """
        honest = statistics(self.weight, 64)[1]
        once = ternary(self.weight, 64)
        again = float((once - ternary(once, 64)).norm() / once.norm())
        self.assertLess(again, honest)
        self.assertGreater(honest - again, 0.1)

    def test_zero_scale_group_stays_zero(self) -> None:
        zeros = torch.zeros(2, GROUP_SIZE)
        self.assertTrue(bool((ternary(zeros, GROUP_SIZE) == 0).all()))


if __name__ == "__main__":
    unittest.main()
