"""The figures must be generated, current, and free of local paths.

A figure is a claim rendered as pixels. If someone re-runs the study and forgets to
regenerate the picture, the README keeps showing the old number, and nothing fails.
So: the checked-in SVG must equal what the generator produces from the evidence bundle
today, and no figure may leak a machine path into a public repository.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"
EVIDENCE = ROOT / "docs" / "evidence" / "ff_evidence.json"
IMG = ROOT / "docs" / "img"

FIGURES = (
    "fig-mnist-ternary-cost.svg",
    "fig-cifar-two-training-sets.svg",
    "fig-label-isolation-recovery.svg",
)


def load_generator():
    spec = importlib.util.spec_from_file_location("make_figures_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = load_generator()
        with open(EVIDENCE, encoding="utf-8") as handle:
            cls.bundle = json.load(handle)
        cls.figures = cls.generator.build(cls.bundle)

    def test_all_three_figures_are_shipped(self) -> None:
        self.assertEqual(sorted(self.figures), sorted(FIGURES))
        for name in FIGURES:
            self.assertTrue((IMG / name).is_file(), f"missing {name}")

    def test_checked_in_svg_equals_a_fresh_render(self) -> None:
        for name, body in self.figures.items():
            on_disk = (IMG / name).read_text(encoding="utf-8")
            self.assertEqual(on_disk, body,
                             f"{name} is stale; run python scripts/make_figures.py")

    def test_no_figure_leaks_a_local_path(self) -> None:
        # Built at runtime on purpose: a literal drive path in this file would itself
        # be reported by scripts/check_public.py, which scans the test files too.
        separators = ("/", "\\")
        needles = [f"{drive}:{sep}" for drive in "CH" for sep in separators]
        needles += ["Users/lin", "acceptance_runs"]
        for name, body in self.figures.items():
            for needle in needles:
                self.assertNotIn(needle, body, f"{name} leaks {needle}")

    def test_mnist_figure_quotes_the_bundle(self) -> None:
        body = self.figures["fig-mnist-ternary-cost.svg"]
        summary = self.bundle["precision_matrix"]["summary"]
        self.assertIn(f"{summary['fp32']['mean_accuracy'] * 100:.2f}%", body)
        self.assertIn(f"{summary['ternary']['mean_accuracy'] * 100:.2f}%", body)
        self.assertIn(f"{self.bundle['headline_pp']['blockwise_share_of_gap'] * 100:.1f}%", body)

    def test_cifar_figure_carries_both_training_sets(self) -> None:
        body = self.figures["fig-cifar-two-training-sets.svg"]
        for block in ("cifar", "cifar_full"):
            for variant in ("fp32", "aligned_g11", "unaligned_g11", "ternary_row"):
                value = self.bundle[block]["variants"][variant]["accuracy_mean"] * 100
                self.assertIn(f"{value:.2f}", body, f"{block}.{variant} missing from the figure")

    def test_recovery_figure_refuses_an_impossible_share(self) -> None:
        bundle = json.loads(json.dumps(self.bundle))
        bundle["headline_pp"]["cifar_aligned_share_of_gap"] = 1.4
        with self.assertRaises(AssertionError):
            self.generator.figure_recovery(bundle)


if __name__ == "__main__":
    unittest.main()
