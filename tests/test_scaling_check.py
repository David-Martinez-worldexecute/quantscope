"""Guard the scaling bundle: the collapse set and the verdicts must be recomputable.

The bundle's whole claim is comparative -- ten times the questions leaves the collapse set
unchanged, and every collapsed row still scores its constant-answer baseline -- so the
checker recomputes the collapsed set from the shares, derives each verdict from the accuracy
interval and the gold share, and refuses a bundle whose stored table disagrees.
"""
from __future__ import annotations

import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

SCHEMA = "quantscope-scaling-check-v1"
SMALL = "40"
LARGE = "400"
CONSTANTS = {"A": 0.2, "B": 0.3, "C": 0.3, "D": 0.2}


def row(layers: int, histogram: dict, correct: int, accuracy_ci: tuple) -> dict:
    questions = sum(histogram.values())
    most = max(histogram, key=lambda letter: histogram[letter])
    return {
        "layers": layers,
        "questions": questions,
        "histogram": dict(histogram),
        "correct": correct,
        "accuracy": correct / questions,
        "accuracy_ci": list(accuracy_ci),
        "most_common": most,
        "most_common_count": histogram[most],
        "most_common_share": histogram[most] / questions,
    }


def verdict(sample: str, layers: int, letter: str, accuracy: float) -> dict:
    return {"sample": sample, "layers": layers, "letter": letter, "accuracy": accuracy,
            "constant": CONSTANTS[letter],
            "verdict": f"not proven to differ from always-{letter}"}


def scaling_bundle() -> dict:
    small_rows = [
        row(12, {"A": 4, "B": 3, "C": 2, "D": 1}, 3, (0.1, 0.6)),
        row(18, {"A": 10}, 2, (0.06, 0.5)),
    ]
    large_rows = [
        row(12, {"A": 160, "B": 120, "C": 80, "D": 40}, 120, (0.26, 0.35)),
        row(18, {"A": 400}, 80, (0.164, 0.241)),
    ]
    return {
        "schema": SCHEMA,
        "generated_utc": "2026-11-25T00:00:00Z",
        "z": 1.959963984540054,
        "intervals": "wilson score, 95%, two-sided",
        "constants": {SMALL: dict(CONSTANTS), LARGE: dict(CONSTANTS)},
        "rows": {SMALL: small_rows, LARGE: large_rows},
        "collapsed": {SMALL: [18], LARGE: [18], "same_set": True},
        "collapsed_vs_constants": [
            verdict(SMALL, 18, "A", 0.2),
            verdict(LARGE, 18, "A", 0.2),
        ],
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class ScalingCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("scaling-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict) -> "object":
        path = self.directory / "scaling_check.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_both_samples(self) -> None:
        report = reproduce_mod.reproduce(self.write(scaling_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn(f"scaling.{SMALL}.18.most_common_share", ids)
        self.assertIn(f"scaling.{LARGE}.12.accuracy", ids)
        self.assertIn(f"scaling.{SMALL}.constants_sum", ids)
        self.assertIn(f"scaling.{SMALL}.18.accuracy_vs_constant", ids)

    def test_a_share_that_does_not_match_its_histogram_fails(self) -> None:
        bundle = scaling_bundle()
        bundle["rows"][LARGE][0]["most_common_share"] = 0.5
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), [f"scaling.{LARGE}.12.most_common_share"])

    def test_a_verdict_that_does_not_match_its_row_fails(self) -> None:
        bundle = scaling_bundle()
        bundle["collapsed_vs_constants"][1]["accuracy"] = 0.4
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), [f"scaling.{LARGE}.18.accuracy_vs_constant"])

    def test_constants_that_do_not_sum_to_one_fail(self) -> None:
        bundle = scaling_bundle()
        bundle["constants"][SMALL]["D"] = 0.3
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), [f"scaling.{SMALL}.constants_sum"])

    def test_a_rewritten_verdict_is_refused(self) -> None:
        bundle = scaling_bundle()
        bundle["collapsed_vs_constants"][0]["verdict"] = "above always-A"
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_missing_verdict_is_refused(self) -> None:
        bundle = scaling_bundle()
        bundle["collapsed_vs_constants"].pop()
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_duplicated_verdict_is_refused(self) -> None:
        bundle = scaling_bundle()
        bundle["collapsed_vs_constants"].append(dict(bundle["collapsed_vs_constants"][0]))
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_histogram_that_does_not_sum_to_the_count_is_refused(self) -> None:
        bundle = scaling_bundle()
        bundle["rows"][SMALL][0]["histogram"]["A"] = 5
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_constants_missing_a_letter_are_refused(self) -> None:
        bundle = scaling_bundle()
        bundle["constants"][SMALL].pop("D")
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_bundle_without_a_collapse_is_refused(self) -> None:
        bundle = scaling_bundle()
        for sample in (SMALL, LARGE):
            for entry in bundle["rows"][sample]:
                entry["most_common_share"] = 0.4
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_stored_collapse_set_is_recomputed(self) -> None:
        bundle = scaling_bundle()
        bundle["collapsed"][LARGE] = [12, 18]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
