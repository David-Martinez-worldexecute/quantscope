"""Guard the packing receipt: "same numbers" has to mean exactly the same bytes.

The bundle is what makes the sentence "packing changes how the weights are stored, not
what they compute" checkable. Two things can go wrong and both are silent in a report:
a tensor that differs in the last bit (so the claim is only approximate), and answer
lists that disagree while the stored count still says twelve. The checks here recompute
the difference and the agreement from the stored lists, so neither can drift.
"""
from __future__ import annotations

import copy
import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

ANSWERS = ["D", "A", "A", "B"]


def equivalence_bundle() -> dict:
    modes = {}
    for mode in ("int4", "ternary"):
        modes[mode] = {
            "replaced_layers": 196,
            "count": len(ANSWERS),
            "answers_identical_to_simulated": len(ANSWERS),
            "max_abs_logit_diff": 0.0,
            "simulated_answers": list(ANSWERS),
            "packed_answers": list(ANSWERS),
            "simulated_answers_matching_float32": 2,
        }
    return {
        "schema": reproduce_mod.EQUIVALENCE_SCHEMA,
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": {"name": "Qwen3-0.6B"},
        "protocol": {
            "group_size": 64,
            "questions": len(ANSWERS),
            "reference": "the repository's own simulated quantisation, not float32",
        },
        "matrices_compared": 392,
        "worst_abs_diff": 0.0,
        "discrepancies": [],
        "modes": modes,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


class EquivalenceBundleTest(unittest.TestCase):
    def write(self, bundle: dict) -> str:
        directory = support.new_dir("equivalence-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = f"{directory}/packing_equivalence.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle)
        return path

    def report(self, bundle: dict) -> dict:
        return reproduce_mod.reproduce(self.write(bundle))

    def test_clean_bundle_passes_and_covers_every_mode(self) -> None:
        report = self.report(equivalence_bundle())
        self.assertEqual(report["status"], "pass", report)
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("equivalence.worst_abs_diff", ids)
        for mode in ("int4", "ternary"):
            self.assertIn(f"equivalence.{mode}.answers_identical_to_simulated", ids)
            self.assertIn(f"equivalence.{mode}.max_abs_logit_diff", ids)

    def test_a_nonzero_tensor_difference_fails(self) -> None:
        bundle = equivalence_bundle()
        bundle["worst_abs_diff"] = 1.0e-7
        report = self.report(bundle)
        self.assertEqual(report["status"], "fail", report)
        failures = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failures, ["equivalence.worst_abs_diff"])

    def test_a_listed_discrepancy_is_rejected(self) -> None:
        bundle = equivalence_bundle()
        bundle["discrepancies"] = [{"matrix": "model.layers.0.mlp.up_proj", "abs_diff": 1e-7}]
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_answer_lists_that_disagree_fail(self) -> None:
        bundle = equivalence_bundle()
        bundle["modes"]["int4"]["packed_answers"] = ["A", "A", "A", "B"]
        report = self.report(bundle)
        self.assertEqual(report["status"], "fail", report)
        failures = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failures, ["equivalence.int4.answers_identical_to_simulated"])

    def test_answer_count_that_contradicts_the_protocol_is_rejected(self) -> None:
        bundle = equivalence_bundle()
        bundle["protocol"]["questions"] = 12
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_unknown_mode_is_rejected(self) -> None:
        bundle = equivalence_bundle()
        bundle["modes"]["baseline"] = copy.deepcopy(bundle["modes"]["int4"])
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)


if __name__ == "__main__":
    unittest.main()
