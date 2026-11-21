"""Guard the guard: the evidence bundle checker must fail on broken bundles.

`scripts/check_public.py` refuses to publish a repo whose evidence bundle is
inconsistent with the docs. That refusal is only worth anything if it actually
fires, so this file builds the smallest bundle the checker accepts and then
breaks one property at a time.

Each test breaks exactly one invariant, because a checker that rejects
everything is as useless as one that accepts everything.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_public.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_public_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned_mnist_spans() -> list[list[int]]:
    # 784 pixel columns in one span, then each of the 10 label columns alone.
    return [[0, 784]] + [[784 + i, 785 + i] for i in range(10)]


def unaligned_mnist_spans() -> list[list[int]]:
    # Same group count, but 794 does not divide by 11, so the boundaries land mid-pixel-block.
    edges = [round(i * 794 / 11) for i in range(12)]
    return [[edges[i], edges[i + 1]] for i in range(11)]


def aligned_cifar_spans() -> list[list[int]]:
    return [[0, 3072]] + [[3072 + i, 3073 + i] for i in range(10)]


def unaligned_cifar_spans() -> list[list[int]]:
    edges = [round(i * 3082 / 11) for i in range(12)]
    return [[edges[i], edges[i + 1]] for i in range(11)]


def cifar_payload() -> dict:
    """One CIFAR-10 run block; the bundle carries the 20k and the full-set run."""
    return {
        "group_definitions": {
            "aligned_g11": aligned_cifar_spans(),
            "unaligned_g11": unaligned_cifar_spans(),
        },
        "variants": {
            "fp32": {"quantisers": []},
            "ternary_row": {"quantisers": ["ternary_linear"]},
            "aligned_g11": {"quantisers": ["grouped_ste"]},
            "unaligned_g11": {"quantisers": ["grouped_ste"]},
        },
    }


def valid_bundle() -> dict:
    return {
        "precision_matrix": {
            "summary": {
                "fp32": {"mean_accuracy": 0.9414},
                "ternary": {"mean_accuracy": 0.8952},
                "first_fp32": {"mean_accuracy": 0.9357},
                "second_fp32": {"mean_accuracy": 0.9399},
            }
        },
        "mechanism": {
            "blockwise_in_training": {"accuracy_mean": 0.9249},
            "two_bit_layer0": {"accuracy_mean": 0.9014},
            "two_bit_both": {"accuracy_mean": 0.9056},
            "label_at_second_layer": {"accuracy_mean": 0.7992},
            "reproduction": {
                "rows": [
                    {"seed": seed, "same_prediction": 10000, "same_correct": 10000, "count": 10000}
                    for seed in (0, 1, 2)
                ]
            },
        },
        "group_scale": {
            "variants": {
                "aligned_g11": {"groups": 11, "spans": aligned_mnist_spans()},
                "unaligned_g11": {"groups": 11, "spans": unaligned_mnist_spans()},
                "aligned_g14": {"groups": 14, "spans": []},
            }
        },
        "cifar": cifar_payload(),
        "cifar_full": cifar_payload(),
        "group_two_layers": {
            "runs": {
                "20261121T115449Z": {"status": "complete", "layer1_input_dim": 500, "variants": {"l1_g10": {}}},
                "20261121T112844Z": {"status": "running", "layer1_input_dim": 500, "variants": {"l1_g50": {}}},
            }
        },
        "headline_pp": {
            "fp32_to_ternary": -4.62,
            "blockwise_recovery": 2.963333,
            "blockwise_share_of_gap": 0.641414,
            "cifar_aligned_share_of_gap": 0.991365,
            "cifar_full_aligned_share_of_gap": 0.973475,
        },
    }


class EvidenceBundleCheckerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.checker = load_checker()

    def issues(self, payload: dict) -> list[str]:
        return self.checker.evidence_bundle_issues(payload)

    def test_accepts_the_minimal_valid_bundle(self) -> None:
        self.assertEqual(self.issues(valid_bundle()), [])

    def test_rejects_missing_cifar_variant(self) -> None:
        payload = valid_bundle()
        del payload["cifar"]["variants"]["unaligned_g11"]
        self.assertTrue(any("cifar.variants.unaligned_g11" in issue for issue in self.issues(payload)))

    def test_rejects_a_bundle_that_drops_the_full_cifar_run(self) -> None:
        # The 50,000-example run is the check that the ordering is not an artefact
        # of the 20,000-example prefix, so a bundle without it must not pass.
        payload = valid_bundle()
        del payload["cifar_full"]
        self.assertTrue(any("cifar_full.variants" in issue for issue in self.issues(payload)))

    def test_rejects_a_full_cifar_run_that_secretly_aligns_the_labels(self) -> None:
        payload = valid_bundle()
        payload["cifar_full"]["group_definitions"]["unaligned_g11"] = aligned_cifar_spans()
        self.assertTrue(any("cifar_full" in issue for issue in self.issues(payload)))

    def test_rejects_a_cifar_unaligned_run_that_secretly_aligns_the_labels(self) -> None:
        # The whole comparison rests on the unaligned run NOT isolating the label block.
        payload = valid_bundle()
        payload["cifar"]["group_definitions"]["unaligned_g11"] = aligned_cifar_spans()
        self.assertTrue(any("cifar unaligned_g11 has a boundary" in issue for issue in self.issues(payload)))

    def test_rejects_a_cifar_aligned_run_without_the_label_boundary(self) -> None:
        payload = valid_bundle()
        payload["cifar"]["group_definitions"]["aligned_g11"] = unaligned_cifar_spans()
        self.assertTrue(any("cifar aligned_g11 has no boundary" in issue for issue in self.issues(payload)))

    def test_rejects_a_two_layer_run_whose_layer_one_is_not_500_wide(self) -> None:
        payload = valid_bundle()
        payload["group_two_layers"]["runs"]["20261121T115449Z"]["layer1_input_dim"] = 510
        self.assertTrue(any("500-wide layer 1" in issue for issue in self.issues(payload)))

    def test_rejects_an_unfinished_two_layer_run_with_no_variants(self) -> None:
        payload = valid_bundle()
        payload["group_two_layers"]["runs"]["20261121T112844Z"]["variants"] = {}
        self.assertTrue(any("unfinished and empty" in issue for issue in self.issues(payload)))

    def test_rejects_a_bundle_without_the_cifar_headline(self) -> None:
        payload = valid_bundle()
        del payload["headline_pp"]["cifar_aligned_share_of_gap"]
        self.assertTrue(any("cifar_aligned_share_of_gap" in issue for issue in self.issues(payload)))

    def test_rejects_a_grouped_cifar_variant_that_does_not_name_its_quantiser(self) -> None:
        payload = valid_bundle()
        payload["cifar"]["variants"]["aligned_g11"]["quantisers"] = []
        self.assertTrue(any("does not name its quantiser" in issue for issue in self.issues(payload)))

    def test_the_published_bundle_in_the_repo_passes(self) -> None:
        # A checker that passes its own fixture but fails the shipped bundle would be theatre.
        bundle_path = ROOT / "docs" / "evidence" / "ff_evidence.json"
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.assertEqual(self.issues(payload), [])

    def test_breaking_the_fixture_does_not_mutate_the_original(self) -> None:
        payload = valid_bundle()
        broken = copy.deepcopy(payload)
        broken["headline_pp"].clear()
        self.assertEqual(self.issues(payload), [])
        self.assertTrue(self.issues(broken))


if __name__ == "__main__":
    unittest.main()
