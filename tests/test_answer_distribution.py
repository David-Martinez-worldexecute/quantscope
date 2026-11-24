"""Guard the answer-distribution bundle: every share must follow from its pick count.

This bundle carries the sharpest claim in the project -- one format answered the same
letter for 99.8% of the questions, so its accuracy is a constant, not a measurement of
the model. That sentence is only worth publishing if the numbers behind it are
recomputable: shares over counts, the modal gold letter over gold counts, and the set of
formats that crossed the collapse threshold.
"""
from __future__ import annotations

import copy
import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

TOTAL = 10
GOLD = {"A": 4, "B": 3, "C": 2, "D": 1}
PICKS = {
    "baseline": {"A": 6, "B": 2, "C": 1, "D": 1},
    "int8": {"A": 7, "B": 1, "C": 1, "D": 1},
    "int4": {"A": 1, "B": 7, "C": 1, "D": 1},
    "ternary": {"A": 0, "B": 0, "C": 0, "D": 10},
}


def distribution_bundle() -> dict:
    modes = {}
    for mode, picks in PICKS.items():
        share = {letter: round(picks[letter] / TOTAL, 6) for letter in picks}
        most = max(picks, key=lambda letter: picks[letter])
        modes[mode] = {
            "count": TOTAL,
            "picks": dict(picks),
            "share": share,
            "distinct_answers": sum(1 for count in picks.values() if count),
            "most_common_pick": most,
            "most_common_share": share[most],
            "gold": dict(GOLD),
            "modal_gold": "A",
            "constant_output_accuracy": round(GOLD["A"] / TOTAL, 6),
            "accuracy": round(GOLD["A"] / TOTAL, 6),
            "file_bytes": 1000,
            "file_sha256": "0" * 64,
        }
    return {
        "schema": reproduce_mod.ANSWER_SCHEMA,
        "questions": TOTAL,
        "source_run": "synthetic",
        "modes": modes,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


class AnswerDistributionTest(unittest.TestCase):
    def write(self, bundle: dict) -> str:
        directory = support.new_dir("answers-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = f"{directory}/answer_distribution.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle)
        return path

    def report(self, bundle: dict) -> dict:
        return reproduce_mod.reproduce(self.write(bundle))

    def test_clean_bundle_passes_and_covers_every_mode(self) -> None:
        report = self.report(distribution_bundle())
        self.assertEqual(report["status"], "pass", report)
        ids = {check["id"] for check in report["checks"]}
        for mode in PICKS:
            for letter in ("A", "B", "C", "D"):
                self.assertIn(f"answers.{mode}.share.{letter}", ids)
            self.assertIn(f"answers.{mode}.most_common_share", ids)
            self.assertIn(f"answers.{mode}.constant_output_accuracy", ids)

    def test_a_share_that_does_not_follow_its_count_fails(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["int4"]["share"]["B"] = 0.5
        report = self.report(bundle)
        self.assertEqual(report["status"], "fail", report)
        failures = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failures, ["answers.int4.share.B"])

    def test_picks_that_do_not_sum_to_the_count_are_rejected(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["baseline"]["picks"]["A"] += 1
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_a_second_collapsed_format_is_rejected(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["int4"]["picks"] = {"A": 0, "B": 0, "C": 0, "D": 10}
        bundle["modes"]["int4"]["share"] = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0}
        bundle["modes"]["int4"]["most_common_pick"] = "D"
        bundle["modes"]["int4"]["most_common_share"] = 1.0
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_a_most_common_pick_that_contradicts_the_counts_is_rejected(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["baseline"]["most_common_pick"] = "B"
        bundle["modes"]["baseline"]["most_common_share"] = bundle["modes"]["baseline"]["share"]["A"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_gold_labels_that_change_between_modes_are_rejected(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["int8"]["gold"] = copy.deepcopy(GOLD)
        bundle["modes"]["int8"]["gold"]["A"] -= 1
        bundle["modes"]["int8"]["gold"]["D"] += 1
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_unknown_mode_is_rejected(self) -> None:
        bundle = distribution_bundle()
        bundle["modes"]["int2"] = copy.deepcopy(bundle["modes"]["int4"])
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)


if __name__ == "__main__":
    unittest.main()
