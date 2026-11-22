"""Guard the reproducer: the `reproduce` command must fail on a bent number.

`quantscope reproduce` claims that every headline percentage in the public docs can be
recomputed from the numbers stored next to it. A checker that always passes would make
that claim meaningless, so each test here bends exactly one number and asserts the
reproducer notices.

The tests read the repository's real evidence bundle to prove the shipped file passes,
then work on a mutated copy for the failure cases.
"""
from __future__ import annotations

import copy
import json
import unittest

import support

from quantscope import reproduce as reproduce_mod


def load_repository_bundle() -> dict:
    with open(reproduce_mod.DEFAULT_EVIDENCE, encoding="utf-8") as handle:
        return json.load(handle)


class ReproduceRepositoryTest(unittest.TestCase):
    def test_repository_bundle_reproduces_cleanly(self) -> None:
        report = reproduce_mod.reproduce()
        self.assertEqual(report["status"], "pass", report.get("failures"))
        self.assertEqual(report["counts"]["failed"], 0)
        # 12 precision-matrix cells + 4 summaries + 15 block means + 21 headline numbers.
        self.assertGreaterEqual(report["counts"]["checked"], 40)
        self.assertEqual(len(report["evidence_sha256"]), 64)

    def test_every_headline_number_is_covered(self) -> None:
        bundle = load_repository_bundle()
        report = reproduce_mod.reproduce()
        covered = {check["id"] for check in report["checks"]}
        for key in bundle["headline_pp"]:
            self.assertIn(f"headline_pp.{key}", covered)


class ReproduceTamperTest(unittest.TestCase):
    def write_bundle(self, bundle: dict) -> str:
        directory = support.new_dir("reproduce-")
        self.addCleanup(__import__("shutil").rmtree, directory, True)
        path = directory / "ff_evidence.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def test_one_wrong_headline_number_fails(self) -> None:
        bundle = load_repository_bundle()
        bundle = copy.deepcopy(bundle)
        key = "fp32_to_ternary"
        bundle["headline_pp"][key] = round(bundle["headline_pp"][key] + 0.5, 6)
        report = reproduce_mod.reproduce(self.write_bundle(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertIn(f"headline_pp.{key}", failing)

    def test_extra_headline_key_is_rejected(self) -> None:
        bundle = copy.deepcopy(load_repository_bundle())
        bundle["headline_pp"]["invented_number"] = 1.0
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write_bundle(bundle))

    def test_missing_headline_key_is_rejected(self) -> None:
        bundle = copy.deepcopy(load_repository_bundle())
        bundle["headline_pp"].pop("blockwise_recovery")
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write_bundle(bundle))

    def test_summary_that_disagrees_with_its_cells_fails(self) -> None:
        bundle = copy.deepcopy(load_repository_bundle())
        bundle["precision_matrix"]["summary"]["fp32"]["mean_accuracy"] += 0.001
        report = reproduce_mod.reproduce(self.write_bundle(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertTrue(any("fp32" in item for item in failing), failing)

    def test_cell_accuracy_that_disagrees_with_its_counts_fails(self) -> None:
        bundle = copy.deepcopy(load_repository_bundle())
        cell = bundle["precision_matrix"]["cells"][0]
        cell["accuracy"] = round(cell["accuracy"] - 0.01, 6)
        report = reproduce_mod.reproduce(self.write_bundle(bundle))
        self.assertEqual(report["status"], "fail")


class RealModelBundleTest(unittest.TestCase):
    """The 171-question bundle: every printed ratio has to come from stored integers."""

    def write_bundle(self, bundle: dict) -> str:
        directory = support.new_dir("real-model-")
        self.addCleanup(__import__("shutil").rmtree, directory, True)
        path = directory / "real_model_test.json"
        path.write_text(json.dumps(bundle), encoding="utf-8")
        return path

    def load_bundle(self) -> dict:
        path = reproduce_mod.EVIDENCE_DIR / "real_model_qwen3_0.6b.json"
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_bundle_is_present_and_covers_every_mode(self) -> None:
        report = reproduce_mod.reproduce()
        covered = {check["id"] for check in report["checks"]}
        for mode in reproduce_mod.REAL_MODEL_MODES:
            self.assertIn(f"real_model.{mode}.accuracy", covered)
            self.assertIn(f"real_model.{mode}.correct", covered)

    def test_tampered_accuracy_fails(self) -> None:
        bundle = self.load_bundle()
        bundle["modes"]["int8"]["accuracy"] = 0.5
        report = reproduce_mod.reproduce(self.write_bundle(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertIn("real_model.int8.accuracy", failing)

    def test_moving_a_flip_between_columns_fails_the_baseline_identity(self) -> None:
        # Keep the four cells summing to 171 (so only the identities can catch it) but
        # move one lost question into the gained column.
        bundle = self.load_bundle()
        pairs = bundle["modes"]["int4"]["pairs_vs_baseline"]
        pairs["n10"] += 1
        pairs["n01"] -= 1
        report = reproduce_mod.reproduce(self.write_bundle(bundle))
        self.assertEqual(report["status"], "fail")
        failing = [check["id"] for check in report["checks"] if not check["ok"]]
        self.assertIn("real_model.int4.accuracy_from_baseline", failing)

    def test_unknown_schema_is_rejected(self) -> None:
        with self.assertRaises(reproduce_mod.ReproduceError):
            reproduce_mod.reproduce(self.write_bundle({"schema": "something-else-v1"}))


class ReproduceCliTest(unittest.TestCase):
    def test_cli_exit_code_is_zero_on_the_repository_bundle(self) -> None:
        from quantscope import cli

        self.assertEqual(cli.main(["reproduce"]), 0)


if __name__ == "__main__":
    unittest.main()
