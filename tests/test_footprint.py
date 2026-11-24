"""Guard the footprint bundle: the stored-weight claim has to be checkable.

`packed_qwen3_0.6b.json` measures a process that loads float32 and then packs, so its
resident set never falls. This bundle measures the other thing: stream the checkpoint,
keep only the packed codes, and report what is held. The checks here are the two
identities that can be recomputed from the stored bytes plus the ordering that is the
claim itself, because a reversed payload order would make the cost page say the
opposite of what was measured.
"""
from __future__ import annotations

import copy
import json
import shutil
import unittest

import support

from quantscope import reproduce as reproduce_mod

HELD = {
    "baseline": 600_000_000,
    "int4": 85_000_000,
    "ternary": 50_000_000,
}
SETTLED = {
    "baseline": 2900.0,
    "int4": 760.0,
    "ternary": 620.0,
}


def footprint_bundle() -> dict:
    modes = {}
    for mode, held in HELD.items():
        modes[mode] = {
            "mode": mode,
            "matrices": 196,
            "held_tensors": 196 if mode == "baseline" else 392,
            "held_bytes": held,
            "held_mib": round(held / (1024 * 1024), 3),
            "load_seconds": 1.5,
            "rss_peak_mib": SETTLED[mode] + 100.0,
            "rss_settled_mib": SETTLED[mode],
        }
    return {
        "schema": reproduce_mod.FOOTPRINT_SCHEMA,
        "generated_utc": "2026-11-24T00:00:00Z",
        "model": {"name": "Qwen3-0.6B"},
        "protocol": {"group_size": 64, "device": "cpu", "dtype": "float32"},
        "modes": modes,
        "limits": ["synthetic bundle: this file exists to exercise the checker"],
    }


class FootprintBundleTest(unittest.TestCase):
    def write(self, bundle: dict) -> str:
        directory = support.new_dir("footprint-")
        self.addCleanup(shutil.rmtree, directory, True)
        path = f"{directory}/packed_footprint.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(bundle, handle)
        return path

    def report(self, bundle: dict) -> dict:
        return reproduce_mod.reproduce(self.write(bundle))

    def test_clean_bundle_passes_and_covers_every_mode(self) -> None:
        report = self.report(footprint_bundle())
        self.assertEqual(report["status"], "pass", report)
        ids = {check["id"] for check in report["checks"]}
        for mode in HELD:
            self.assertIn(f"footprint.{mode}.held_mib", ids)

    def test_held_mib_that_disagrees_with_its_bytes_fails(self) -> None:
        bundle = footprint_bundle()
        bundle["modes"]["int4"]["held_mib"] += 3.0
        report = self.report(bundle)
        self.assertEqual(report["status"], "fail", report)
        failures = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertEqual(failures, ["footprint.int4.held_mib"])

    def test_held_bytes_in_the_wrong_order_are_rejected(self) -> None:
        bundle = footprint_bundle()
        bundle["modes"]["int4"]["held_bytes"] = bundle["modes"]["ternary"]["held_bytes"] - 1
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_settled_rss_that_does_not_follow_the_bytes_is_rejected(self) -> None:
        bundle = footprint_bundle()
        bundle["modes"]["ternary"]["rss_settled_mib"] = 4000.0
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_tensor_count_that_forgets_the_scales_is_rejected(self) -> None:
        bundle = footprint_bundle()
        bundle["modes"]["ternary"]["held_tensors"] = 196
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_missing_settled_rss_is_rejected(self) -> None:
        bundle = footprint_bundle()
        bundle["modes"]["int4"]["rss_settled_mib"] = None
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)

    def test_unknown_mode_is_rejected(self) -> None:
        bundle = footprint_bundle()
        extra = copy.deepcopy(bundle["modes"]["int4"])
        extra["mode"] = "int8"
        bundle["modes"]["int8"] = extra
        with self.assertRaises(reproduce_mod.ReproduceError):
            self.report(bundle)


if __name__ == "__main__":
    unittest.main()
