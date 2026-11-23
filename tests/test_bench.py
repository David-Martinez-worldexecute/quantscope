"""Guard the cost bundle: the bench schema must be checked, not trusted.

Timings cannot be recomputed on another machine, so `reproduce` checks the
identities that live inside the file instead: the stored rate against the stored
seconds, the median inside its own min and max, and the ordering of payload sizes
that is true of the formats themselves. Each test here bends exactly one of those.
"""
from __future__ import annotations

import copy
import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

SIZES = {
    "baseline": 600_000_000,
    "int8": 160_000_000,
    "int4": 85_000_000,
    "ternary": 50_000_000,
}


def bench_bundle() -> dict:
    modes = {}
    for mode, size in SIZES.items():
        seconds = 0.8
        modes[mode] = {
            "quantized_bytes": size,
            "replaced_linear_layers": 0 if mode == "baseline" else 196,
            "load_seconds": 3.0,
            "prefill_ms_median": 120.0,
            "prefill_ms_min": 110.0,
            "prefill_ms_max": 130.0,
            "prefill_repeats": 5,
            "decode_seconds": seconds,
            "max_new_tokens": 32,
            "decode_tokens_per_second": round(32 / seconds, 6),
            "decode_repeats": 3,
            "peak_rss_mib": 3000.0,
        }
    return {
        "schema": reproduce_mod.BENCH_SCHEMA,
        "generated_utc": "2026-11-23T00:00:00Z",
        "model": {"name": "Qwen3-0.6B", "disk_weight_bytes": 1_500_000_000},
        "environment": {"torch": "2.6.0", "platform": "cpu", "threads": 4},
        "protocol": {
            "device": "cpu",
            "prompt_tokens": 128,
            "max_new_tokens": 32,
            "prefill_repeats": 5,
            "decode_repeats": 3,
            "group_size": 64,
        },
        "modes": modes,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


class BenchBundleTest(unittest.TestCase):
    def write(self, bundle: dict) -> str:
        directory = support.new_dir("bench-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = directory / "bench_test.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_every_mode(self) -> None:
        report = reproduce_mod.reproduce(self.write(bench_bundle()))
        self.assertEqual(report["status"], "pass", report["checks"])
        covered = {check["id"] for check in report["checks"]}
        self.assertEqual(len(covered), 8)
        for mode in SIZES:
            self.assertIn(f"bench.{mode}.decode_tokens_per_second", covered)
            self.assertIn(f"bench.{mode}.prefill_ms_median", covered)

    def test_rate_that_disagrees_with_its_seconds_fails(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle["modes"]["int4"]["decode_tokens_per_second"] += 1.0
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failing, ["bench.int4.decode_tokens_per_second"])

    def test_median_outside_its_own_bounds_fails(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle["modes"]["ternary"]["prefill_ms_median"] = 500.0
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failing, ["bench.ternary.prefill_ms_median"])

    def test_sizes_out_of_order_are_rejected(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle["modes"]["int4"]["quantized_bytes"] = SIZES["ternary"]
        with self.assertRaises(reproduce_mod.ReproduceError) as caught:
            reproduce_mod.reproduce(self.write(bundle))
        self.assertIn("ordered by width", str(caught.exception))

    def test_missing_mode_is_rejected(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle["modes"].pop("int8")
        with self.assertRaises(reproduce_mod.ReproduceError) as caught:
            reproduce_mod.reproduce(self.write(bundle))
        self.assertIn("expected modes", str(caught.exception))

    def test_zero_seconds_is_rejected_not_divided(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle["modes"]["int8"]["decode_seconds"] = 0.0
        with self.assertRaises(reproduce_mod.ReproduceError) as caught:
            reproduce_mod.reproduce(self.write(bundle))
        self.assertIn("decode_seconds", str(caught.exception))

    def test_missing_limit_list_is_rejected(self) -> None:
        bundle = copy.deepcopy(bench_bundle())
        bundle.pop("limits")
        with self.assertRaises(reproduce_mod.ReproduceError) as caught:
            reproduce_mod.reproduce(self.write(bundle))
        self.assertIn("limits", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
