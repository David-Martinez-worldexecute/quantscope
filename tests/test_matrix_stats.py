# -*- coding: utf-8 -*-
"""The ternary quantiser must honour the group size it is given.

Two defects were found in the published ablation bundles on 2026-11-25, both of the
"parameter never reached the code that used it" kind:

  * the statistics function hard-coded group size 64 inside its body, so every bundle
    reported group-64 numbers no matter which group size the run actually used;
  * it was called while the model was still quantised, so it ternarised an already
    ternarised weight and reported a deflated error -- 0.3053 instead of 0.5078 on a
    64x128 tensor at group 64.

The quantiser is reimplemented here in plain Python rather than imported, for two
reasons: the CI job that runs the unit tests installs no dependencies, and the public
tree has to stay free of lab paths. Plain lists also make the arithmetic in a test
readable on its own.
"""
from __future__ import annotations

import math
import random
import unittest

GROUP_SIZE = 64


def ternary(values: list[float], group_size: int) -> list[float]:
    """Per-group absmean scale, each weight rounded to -1, 0 or +1 times that scale.

    A group whose mean magnitude is zero stays at zero instead of being divided by it.
    """
    out: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        scale = sum(abs(value) for value in group) / len(group)
        if scale == 0:
            out.extend(0.0 for _ in group)
            continue
        for value in group:
            out.append(max(-1.0, min(1.0, round(value / scale))) * scale)
    return out


def statistics(values: list[float], group_size: int) -> tuple[float, float]:
    """Zero fraction and relative error of one honest quantisation of the original weight."""
    rebuilt = ternary(values, group_size)
    zero = sum(1 for value in rebuilt if value == 0) / len(rebuilt)
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(values, rebuilt)))
    denominator = math.sqrt(sum(a * a for a in values))
    return zero, numerator / denominator if denominator else 0.0


class TernaryGroupSizeTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = random.Random(7)
        self.values = [generator.gauss(0.0, 1.0) for _ in range(64 * 128)]

    def test_group_size_actually_changes_the_error(self) -> None:
        errors = [statistics(self.values, group)[1] for group in (32, 64, 128)]
        self.assertEqual(len(set(round(value, 9) for value in errors)), len(errors),
                         f"group size did not change the error: {errors}")

    def test_error_stays_near_a_half_across_these_widths(self) -> None:
        # The published claim: the cliff is not explained by coarser groups destroying more
        # of the weight, because the error barely moves across 32 / 64 / 128.
        for group in (32, 64, 128):
            with self.subTest(group=group):
                error = statistics(self.values, group)[1]
                self.assertGreater(error, 0.40)
                self.assertLess(error, 0.62)

    def test_zero_fraction_is_stable_across_widths(self) -> None:
        zeros = [statistics(self.values, group)[0] for group in (32, 64, 128)]
        self.assertLess(max(zeros) - min(zeros), 0.05,
                        f"zero fraction swung between widths: {zeros}")

    def test_quantising_twice_reports_a_smaller_error(self) -> None:
        """The direction of the bug, recorded so it cannot be mis-stated again.

        Re-quantising deflates rather than inflates the number: the denominator becomes the
        norm of the quantised weight instead of the original, and the second rounding step is
        small. The deflated figure looks like a plausible one-shot error, which is why it
        survived unnoticed in the published table.
        """
        honest = statistics(self.values, 64)[1]
        once = ternary(self.values, 64)
        again = statistics(once, 64)[1]
        self.assertLess(again, honest)
        self.assertGreater(honest - again, 0.05)

    def test_zero_scale_group_stays_zero(self) -> None:
        zeros = [0.0] * (2 * GROUP_SIZE)
        self.assertTrue(all(value == 0 for value in ternary(zeros, GROUP_SIZE)))

    def test_quantised_values_are_multiples_of_their_group_scale(self) -> None:
        group = GROUP_SIZE
        rebuilt = ternary(self.values, group)
        scale = sum(abs(value) for value in self.values[:group]) / group
        self.assertTrue(any(rebuilt[i] != 0 for i in range(group)))
        for value in rebuilt[:group]:
            self.assertIn(round(value / scale), (-1.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
