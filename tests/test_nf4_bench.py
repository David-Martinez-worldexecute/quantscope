"""Guard the NF4 bundle: a real 4-bit container must cost fewer bytes and keep its answers.

The published sentence is comparative -- a real format stores a fifth of the bytes and still
answers most of the same letters -- so the checker recomputes the byte total from the per-dtype
tensors, the agreement count from the two answer lists, and refuses a bundle where the packed
model is not actually smaller.
"""
from __future__ import annotations

import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

COUNT = 10
FP32_ANSWERS = ["A", "B", "A", "D", "C", "A", "B", "B", "A", "D"]
NF4_ANSWERS = ["A", "B", "D", "D", "C", "A", "A", "B", "A", "C"]


def nf4_bundle() -> dict:
    agree = sum(1 for before, after in zip(FP32_ANSWERS, NF4_ANSWERS) if before == after)
    return {
        "schema": reproduce_mod.NF4_SCHEMA,
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": "synthetic",
        "protocol": {
            "questions": COUNT,
            "max_tokens": 96,
            "device": "cuda",
            "scoring": "last-position single-token A/B/C/D logits",
            "gpu": "synthetic",
            "torch": "0.0",
            "cuda": "0.0",
            "container": "bitsandbytes NF4",
            "reference": "the same checkpoint in float32",
        },
        "float32": {
            "correct": 4,
            "count": COUNT,
            "histogram": {"A": 4, "B": 3, "C": 1, "D": 2},
            "most_common": "A",
            "most_common_share": 0.4,
            "stored_bytes": 1_000_000_000,
            "stored_by_dtype": {"float32": 1_000_000_000},
            "gpu_memory_after_load": 1_000_500_000,
            "prefill_ms_median": 50.0,
            "decode_tokens_per_second": 20.0,
        },
        "nf4": {
            "correct": 4,
            "count": COUNT,
            "histogram": {"A": 4, "B": 2, "C": 2, "D": 2},
            "most_common": "A",
            "most_common_share": 0.4,
            "answers_identical_to_float32": agree,
            "mean_abs_logit_drift": 1.5,
            "answers": list(NF4_ANSWERS),
            "float32_answers": list(FP32_ANSWERS),
            "gold": ["A", "B", "C", "D", "C", "A", "B", "B", "A", "C"],
            "stored_bytes": 250_000_000,
            "stored_by_dtype": {"bfloat16": 150_000_000, "uint8": 100_000_000},
            "gpu_memory_after_load": 260_000_000,
            "prefill_ms_median": 80.0,
            "decode_tokens_per_second": 15.0,
            "bitsandbytes": "0.50.2",
        },
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
        "seconds": 1.0,
    }


def failing_ids(report: dict) -> list[str]:
    return [check["id"] for check in report["checks"] if not check["ok"]]


class Nf4BenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = support.new_dir("nf4-")
        self.addCleanup(shutil.rmtree, self.directory, True)

    def write(self, bundle: dict) -> "object":
        path = self.directory / "nf4_bench.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_both_stages(self) -> None:
        report = reproduce_mod.reproduce(self.write(nf4_bundle()))
        self.assertEqual(report["status"], "pass", failing_ids(report))
        ids = {check["id"] for check in report["checks"]}
        self.assertIn("nf4_bench.float32.stored_bytes", ids)
        self.assertIn("nf4_bench.nf4.stored_bytes", ids)
        self.assertIn("nf4_bench.nf4.answers_identical_to_float32", ids)

    def test_byte_total_that_disagrees_with_its_tensors_fails(self) -> None:
        bundle = nf4_bundle()
        bundle["nf4"]["stored_bytes"] = 260_000_000
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report), ["nf4_bench.nf4.stored_bytes"])

    def test_agreement_count_that_does_not_match_the_answers_fails(self) -> None:
        bundle = nf4_bundle()
        bundle["nf4"]["answers_identical_to_float32"] = 9
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(failing_ids(report),
                         ["nf4_bench.nf4.answers_identical_to_float32"])

    def test_answer_lists_that_do_not_cover_the_questions_are_rejected(self) -> None:
        bundle = nf4_bundle()
        bundle["nf4"]["answers"] = bundle["nf4"]["answers"][:5]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_a_container_that_stores_more_than_float32_is_rejected(self) -> None:
        bundle = nf4_bundle()
        bundle["nf4"]["stored_bytes"] = 2_000_000_000
        bundle["nf4"]["stored_by_dtype"] = {"bfloat16": 1_500_000_000, "uint8": 500_000_000}
        bundle["nf4"]["gpu_memory_after_load"] = 2_100_000_000
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_histogram_that_does_not_sum_to_the_count_is_rejected(self) -> None:
        bundle = nf4_bundle()
        bundle["nf4"]["histogram"]["A"] = 9
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_loading_less_memory_than_the_weights_is_rejected(self) -> None:
        bundle = nf4_bundle()
        bundle["float32"]["gpu_memory_after_load"] = 1_000
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))


if __name__ == "__main__":
    unittest.main()
