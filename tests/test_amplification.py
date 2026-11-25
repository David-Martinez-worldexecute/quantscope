# -*- coding: utf-8 -*-
"""The amplification checker: quadrature, interaction factor and the drift exponent."""
from __future__ import annotations

import json
import math
import shutil
import unittest

import support
from quantscope import reproduce as reproduce_mod

SCHEMA = "quantscope-amplification-v1"
LAYERS = 3
AXIS = 4


def fit_exponent(curve: list[float]) -> float:
    """Independent re-implementation, so the test does not borrow the checker's own maths."""
    points = [(index, value) for index, value in enumerate(curve) if value > 1e-6]
    if len(points) < 3:
        return 0.0
    count = len(points)
    mean_x = sum(index for index, _ in points) / count
    mean_y = sum(math.log(value) for _, value in points) / count
    denominator = sum((index - mean_x) ** 2 for index, _ in points)
    if denominator == 0:
        return 0.0
    return sum((index - mean_x) * (math.log(value) - mean_y) for index, value in points) / denominator


def prefix(depth: int, curve: list[float], singles: list[dict]) -> dict:
    quadrature = sum(item["final_hidden_drift"] ** 2 for item in singles if item["layer_index"] < depth) ** 0.5
    return {
        "layers_quantised": depth,
        "quantised_matrices": depth * 7,
        "hidden_drift": curve,
        "final_hidden_drift": curve[-1],
        "logit_drift": 1.0 + depth,
        "mean_growth_ratio": 1.2,
        "log_drift_exponent": round(fit_exponent(curve), 6),
        "independent_quadrature": round(quadrature, 6),
        "interaction_factor": round(curve[-1] / quadrature, 4) if quadrature else None,
    }


def amplification_bundle() -> dict:
    singles = [
        {"layer_index": 0, "quantised_matrices": 3, "own_layer_drift": 0.3,
         "final_hidden_drift": 0.3, "logit_drift": 0.4},
        {"layer_index": 1, "quantised_matrices": 4, "own_layer_drift": 0.4,
         "final_hidden_drift": 0.4, "logit_drift": 0.6},
        {"layer_index": 2, "quantised_matrices": 3, "own_layer_drift": 0.5,
         "final_hidden_drift": 0.5, "logit_drift": 0.9},
    ]
    return {
        "schema": SCHEMA,
        "generated_utc": "2026-01-01T00:00:00Z",
        "model": {"name": "synthetic-model"},
        "protocol": {
            "questions": 8,
            "quantisation": "ternary per-group absmean",
            "group_size": 64,
            "device": "cuda",
            "torch": "test",
            "cuda": "test",
            "hidden_states": "index 0 is the embedding output",
            "drift": "relative Frobenius distance",
            "prefix_depths": [1, 2, 3],
            "threads": 4,
        },
        "layers": LAYERS,
        "depth_axis": AXIS,
        "prefixes": [
            prefix(1, [0.0, 0.1, 0.12, 0.2], singles),
            prefix(2, [0.0, 0.2, 0.3, 0.5], singles),
            prefix(3, [0.0, 0.5, 0.9, 1.2], singles),
        ],
        "single_layers": singles,
        "first_drift_crossing": 3,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
        "seconds": 1.0,
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class AmplificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("amp-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict):
        path = self.directory / "amplification.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_the_arithmetic(self) -> None:
        report = reproduce_mod.reproduce(self.write(amplification_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("amplification.prefix.01.final_hidden_drift", ids)
        self.assertIn("amplification.prefix.02.independent_quadrature", ids)
        self.assertIn("amplification.prefix.03.interaction_factor", ids)
        self.assertIn("amplification.prefix.03.log_drift_exponent", ids)

    def test_final_drift_must_match_the_curve(self) -> None:
        bundle = amplification_bundle()
        bundle["prefixes"][1]["final_hidden_drift"] = 0.55
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["amplification.prefix.02.final_hidden_drift"])

    def test_interaction_factor_must_match_the_quadrature(self) -> None:
        bundle = amplification_bundle()
        bundle["prefixes"][2]["interaction_factor"] = round(bundle["prefixes"][2]["interaction_factor"] + 0.01, 4)
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["amplification.prefix.03.interaction_factor"])

    def test_exponent_must_match_the_curve(self) -> None:
        bundle = amplification_bundle()
        bundle["prefixes"][0]["log_drift_exponent"] = 0.9
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["amplification.prefix.01.log_drift_exponent"])

    def test_first_crossing_must_follow_the_curves(self) -> None:
        bundle = amplification_bundle()
        bundle["first_drift_crossing"] = 2
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_curve_must_span_the_whole_axis(self) -> None:
        bundle = amplification_bundle()
        bundle["prefixes"][1]["hidden_drift"] = [0.0, 0.2, 0.5]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_matrix_count_must_follow_the_depth(self) -> None:
        bundle = amplification_bundle()
        bundle["prefixes"][2]["quantised_matrices"] = 20
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_every_layer_needs_a_single_row(self) -> None:
        bundle = amplification_bundle()
        bundle["single_layers"] = bundle["single_layers"][:2]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
