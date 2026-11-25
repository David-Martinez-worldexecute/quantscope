# -*- coding: utf-8 -*-
"""The granularity checker: a group size that does not collapse must still be checkable."""
from __future__ import annotations

import json
import shutil
import unittest

import support
from quantscope import reproduce as reproduce_mod

SCHEMA = "quantscope-granularity-v1"
QUESTIONS = 10
DEPTH = 2
COLLAPSE = 0.9


def condition(name: str, histogram: dict, correct: int) -> dict:
    count = sum(histogram.values())
    most = max(histogram, key=lambda letter: histogram[letter])
    return {
        "condition": name,
        "count": count,
        "correct": correct,
        "histogram": dict(histogram),
        "most_common": most,
        "most_common_share": histogram[most] / count,
        "quantised_matrices": DEPTH * 7,
    }


def summarise(row: dict) -> dict:
    shares = {item["condition"]: item["most_common_share"] for item in row["conditions"]}
    row = dict(row)
    row["collapsed"] = shares["all_quantised"] >= COLLAPSE
    row["survivors"] = sorted(name for name, share in shares.items() if share < COLLAPSE)
    row["collapsed_keeps"] = sorted(name for name in shares
                                    if name != "all_quantised" and shares[name] >= COLLAPSE)
    return row


def granularity_bundle() -> dict:
    rows = [
        summarise({"group_size": 32, "conditions": [
            condition("all_quantised", {"A": 3, "B": 3, "C": 3, "D": 1}, 3),
            condition("keep_k_proj", {"A": 5, "B": 5}, 5),
        ]}),
        summarise({"group_size": 64, "conditions": [
            condition("all_quantised", {"A": 10}, 2),
            condition("keep_k_proj", {"A": 4, "B": 3, "C": 2, "D": 1}, 4),
        ]}),
    ]
    return {
        "schema": SCHEMA,
        "generated_utc": "2026-01-01T00:00:00Z",
        "model": {"name": "synthetic"},
        "protocol": {
            "questions": QUESTIONS,
            "quantised_layers": DEPTH,
            "collapse_share": COLLAPSE,
            "group_sizes": [32, 64],
            "quantisation": "ternary per-group absmean",
        },
        "rows": rows,
        "collapsed_group_sizes": [size for size in (32, 64)
                                  if next(item for item in rows
                                          if item["group_size"] == size)["collapsed"]],
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class GranularityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("granularity-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict):
        path = self.directory / "granularity.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_both_rows(self) -> None:
        report = reproduce_mod.reproduce(self.write(granularity_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("granularity.g32.all_quantised.most_common_share", ids)
        self.assertIn("granularity.g64.keep_k_proj.most_common_share", ids)

    def test_a_row_that_does_not_collapse_is_accepted(self) -> None:
        bundle = granularity_bundle()
        first = bundle["rows"][0]
        self.assertFalse(first["collapsed"])
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(report["status"], "pass", failing_ids(report))

    def test_share_that_disagrees_with_its_histogram_fails(self) -> None:
        bundle = granularity_bundle()
        bundle["rows"][1]["conditions"][1]["most_common_share"] = 0.5
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["granularity.g64.keep_k_proj.most_common_share"])

    def test_stored_collapsed_flag_must_follow_the_share(self) -> None:
        bundle = granularity_bundle()
        bundle["rows"][0]["collapsed"] = True
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_survivors_must_follow_the_shares(self) -> None:
        bundle = granularity_bundle()
        bundle["rows"][1]["survivors"] = ["all_quantised"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_collapsed_keeps_must_follow_the_shares(self) -> None:
        bundle = granularity_bundle()
        # Row 1 does collapse, and its only kept class has share 0.4, so the honest
        # collapsed_keeps list is empty; claiming otherwise must be refused.
        bundle["rows"][1]["collapsed_keeps"] = ["keep_k_proj"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_collapsed_group_sizes_must_follow_the_rows(self) -> None:
        bundle = granularity_bundle()
        bundle["collapsed_group_sizes"] = [32, 64]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_histogram_that_does_not_sum_to_the_count_is_refused(self) -> None:
        bundle = granularity_bundle()
        bundle["rows"][1]["conditions"][0]["histogram"]["A"] = 7
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_single_group_size_is_refused(self) -> None:
        bundle = granularity_bundle()
        bundle["rows"] = bundle["rows"][:1]
        bundle["protocol"]["group_sizes"] = [32]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_row_counts_must_match_the_protocol(self) -> None:
        bundle = granularity_bundle()
        bundle["protocol"]["questions"] = 20
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
