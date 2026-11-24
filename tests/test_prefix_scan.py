"""Guard the layer-prefix sweep: every share, tensor count and collapse label must follow.

The claim this bundle supports is narrow and load-bearing: quantising one block of four
layers never collapsed the model, while a one-layer-at-a-time sweep found a prefix that did.
If the sweep's own counts, its seven tensors per layer, or the set of collapsed prefixes stop
agreeing with each other, the sentence about "where it collapses" has no evidence behind it.
"""
from __future__ import annotations

import copy
import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

TOTAL = 10


def condition(layers: int, total: int, histogram: dict[str, int]) -> dict:
    count = sum(histogram.values())
    most = max(histogram, key=lambda letter: histogram[letter])
    return {
        "layers_quantised": layers,
        "float32_layers": total - layers,
        "linear_layers": reproduce_mod.LINEAR_PER_LAYER * layers,
        "correct": min(histogram[most], count),
        "count": count,
        "histogram": dict(histogram),
        "most_common": most,
        "most_common_share": round(histogram[most] / count, 6),
        "mean_gap": 3.0,
        "mean_abs_logit_drift": 12.0,
    }


def prefix_bundle() -> dict:
    total = 4
    conditions = [
        condition(2, total, {"A": 10, "B": 0, "C": 0, "D": 0}),
        condition(3, total, {"A": 4, "B": 3, "C": 2, "D": 1}),
        condition(4, total, {"A": 1, "B": 0, "C": 0, "D": 9}),
    ]
    collapsed = [block["layers_quantised"] for block in conditions
                 if block["most_common_share"] >= 0.90]
    return {
        "schema": reproduce_mod.PREFIX_SCHEMA,
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": "synthetic",
        "versions": {"python": "3.12", "torch": "0.0", "transformers": "0.0"},
        "protocol": {
            "questions": TOTAL,
            "group_size": 64,
            "start": 2,
            "max_tokens": 96,
            "scoring": "last-position single-token A/B/C/D logits",
            "quantisation": "ternary per-group absmean",
            "collapse_share": 0.90,
            "threads": 4,
            "device": "cpu",
        },
        "baseline": {
            "correct": 4,
            "count": TOTAL,
            "histogram": {"A": 4, "B": 3, "C": 2, "D": 1},
            "most_common": "A",
            "most_common_share": 0.4,
            "mean_gap": 4.0,
        },
        "conditions": conditions,
        "first_collapse": collapsed[0],
        "collapse_lengths": collapsed,
        "collapse_letters": [block["most_common"] for block in conditions
                             if block["layers_quantised"] in collapsed],
        "collapse_contiguous": collapsed == list(range(collapsed[0], collapsed[-1] + 1)),
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
        "seconds": 1.0,
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class PrefixScanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("prefix-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict) -> "object":
        path = self.directory / "prefix_scan.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_the_sweep(self) -> None:
        report = reproduce_mod.reproduce(self.write(prefix_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("prefix_scan.02.most_common_share", ids)
        self.assertIn("prefix_scan.04.linear_layers", ids)
        self.assertIn("prefix_scan.first_collapse", ids)

    def test_share_that_disagrees_with_its_counts_fails(self) -> None:
        # Tamper with a prefix that did *not* collapse, so the collapsed set is unchanged and
        # the only disagreement left is the share itself.
        bundle = prefix_bundle()
        bundle["conditions"][1]["most_common_share"] = 0.5
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["prefix_scan.03.most_common_share"])

    def test_counts_that_do_not_sum_to_the_question_count_are_rejected(self) -> None:
        bundle = prefix_bundle()
        bundle["conditions"][1]["histogram"]["A"] = 5
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_float32_layers_that_do_not_add_up_are_rejected(self) -> None:
        bundle = prefix_bundle()
        bundle["conditions"][1]["float32_layers"] = 99
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_tensor_count_that_forgets_two_weights_fails(self) -> None:
        bundle = prefix_bundle()
        bundle["conditions"][0]["linear_layers"] = 12
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["prefix_scan.02.linear_layers"])

    def test_first_collapse_that_skips_the_early_prefix_fails(self) -> None:
        bundle = prefix_bundle()
        bundle["first_collapse"] = 4
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["prefix_scan.first_collapse"])

    def test_duplicate_prefix_length_is_rejected(self) -> None:
        bundle = prefix_bundle()
        duplicate = copy.deepcopy(bundle["conditions"][1])
        bundle["conditions"].append(duplicate)
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_sweep_with_no_collapse_is_rejected(self) -> None:
        bundle = prefix_bundle()
        for block in bundle["conditions"]:
            block["histogram"] = {"A": 4, "B": 3, "C": 2, "D": 1}
            block["most_common"] = "A"
            block["most_common_share"] = 0.4
        bundle["first_collapse"] = 2
        bundle["collapse_lengths"] = [2, 4]
        bundle["collapse_letters"] = ["A", "D"]
        bundle["collapse_contiguous"] = False
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
