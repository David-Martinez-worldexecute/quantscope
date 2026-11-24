"""Guard the packed bundle: the memory claim has to be checked, not trusted.

`reproduce` cannot re-weigh the model on another machine, so it checks what the file
must say about itself: the stored decode rate against the stored seconds, the prefill
median inside its own range, a payload order that is true of the formats, a positive
peak RSS, and the isolation flag without which the RSS numbers are not comparable.
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
    "int4": 85_000_000,
    "ternary": 50_000_000,
}


def packed_bundle() -> dict:
    modes = {}
    for mode, size in SIZES.items():
        seconds = 1.25
        modes[mode] = {
            "mode": mode,
            "payload_bytes": size,
            "replaced_linear_layers": 0 if mode == "baseline" else 196,
            "load_seconds": 4.0,
            "prefill_ms_median": 140.0,
            "prefill_ms_min": 130.0,
            "prefill_ms_max": 160.0,
            "decode_seconds": seconds,
            "decode_tokens_per_second": round(32 / seconds, 6),
            "peak_rss_mib": 2800.0 if mode == "baseline" else 1500.0,
        }
    return {
        "schema": reproduce_mod.PACKED_SCHEMA,
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": {"name": "Qwen3-0.6B"},
        "protocol": {
            "isolation": "one child process per mode",
            "group_size": 64,
            "prompt_tokens": 128,
            "max_new_tokens": 32,
            "device": "cpu",
            "dtype": "float32",
        },
        "modes": modes,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


class PackedBundleTest(unittest.TestCase):
    def write(self, bundle: dict) -> str:
        directory = support.new_dir("packed-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = directory / "packed_test.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_clean_bundle_passes_and_covers_every_mode(self) -> None:
        report = reproduce_mod.reproduce(self.write(packed_bundle()))
        self.assertEqual(report["status"], "pass", report["checks"])
        covered = {check["id"] for check in report["checks"]}
        self.assertEqual(len(covered), 6)
        for mode in SIZES:
            self.assertIn(f"packed.{mode}.decode_tokens_per_second", covered)
            self.assertIn(f"packed.{mode}.prefill_ms_median", covered)

    def test_rate_that_disagrees_with_its_seconds_fails(self) -> None:
        bundle = packed_bundle()
        bundle["modes"]["int4"]["decode_tokens_per_second"] += 1.0
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(report["status"], "fail")
        failed = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failed, ["packed.int4.decode_tokens_per_second"])

    def test_median_outside_its_range_fails(self) -> None:
        bundle = packed_bundle()
        bundle["modes"]["ternary"]["prefill_ms_median"] = 500.0
        report = reproduce_mod.reproduce(self.write(bundle))
        self.assertEqual(report["status"], "fail")
        failed = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failed, ["packed.ternary.prefill_ms_median"])

    def test_payload_order_that_contradicts_the_format_is_rejected(self) -> None:
        bundle = packed_bundle()
        bundle["modes"]["int4"]["payload_bytes"] = bundle["modes"]["ternary"]["payload_bytes"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_missing_isolation_flag_is_rejected(self) -> None:
        bundle = packed_bundle()
        del bundle["protocol"]["isolation"]
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_missing_peak_rss_is_rejected(self) -> None:
        bundle = packed_bundle()
        bundle["modes"]["int4"]["peak_rss_mib"] = None
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))

    def test_unknown_mode_is_rejected(self) -> None:
        bundle = packed_bundle()
        bundle["modes"]["int8"] = copy.deepcopy(bundle["modes"]["int4"])
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write(bundle))
