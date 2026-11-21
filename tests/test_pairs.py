import unittest

from support import DATA, temp_dir  # noqa: F401  (also puts the repository root on sys.path)

from quantscope import model, protocol
from quantscope.run import load_questions, pair_counts, run as run_pipeline


class PairCountTest(unittest.TestCase):
    def test_all_correct(self):
        self.assertEqual(
            pair_counts([True, True], [True, True]),
            {"n00": 0, "n01": 0, "n10": 0, "n11": 2},
        )

    def test_baseline_right_candidate_wrong_counts_as_n10(self):
        self.assertEqual(pair_counts([True, False], [False, False])["n10"], 1)

    def test_baseline_wrong_candidate_right_counts_as_n01(self):
        self.assertEqual(pair_counts([False, True], [True, True])["n01"], 1)

    def test_both_wrong_counts_as_n00(self):
        self.assertEqual(pair_counts([False, False], [False, False])["n00"], 2)


class RunPairingTest(unittest.TestCase):
    N = 8

    def _build_model(self, directory):
        questions = load_questions(DATA)[: self.N]
        prompts = [protocol.build_prompt(row["question"], row["options"]) for row in questions]
        weights, bias = model.fit(
            prompts, [row["answer"] for row in questions], seed=0, epochs=80
        )
        return model.save_model(directory, weights, bias, model.DEFAULT_BUCKETS, 0)

    def test_pairs_sum_to_the_number_of_questions(self):
        with temp_dir() as tmp:
            model_dir = self._build_model(tmp / "model")
            manifest, _ = run_pipeline(
                model_dir, DATA, self.N, "fp32,int4", 64, tmp / "run"
            )
            for mode in manifest["quant"]["modes"]:
                entry = manifest["summary"][mode]
                if mode == "fp32":
                    # The baseline is not paired against itself, so it has no
                    # pairs block by design; only the other formats get one.
                    self.assertNotIn("pairs", entry, "baseline has no pairing")
                    continue
                self.assertEqual(sum(entry["pairs"].values()), self.N, mode)

    def test_delta_is_the_accuracy_difference_in_percentage_points(self):
        with temp_dir() as tmp:
            model_dir = self._build_model(tmp / "model")
            manifest, _ = run_pipeline(
                model_dir, DATA, self.N, "fp32,int4", 64, tmp / "run"
            )
            baseline = manifest["summary"]["fp32"]["accuracy"]
            other = manifest["summary"]["int4"]
            expected = 100.0 * (other["accuracy"] - baseline)
            self.assertAlmostEqual(other["delta_pp"], expected, places=6)

    def test_non_baseline_modes_record_their_reference(self):
        with temp_dir() as tmp:
            model_dir = self._build_model(tmp / "model")
            manifest, _ = run_pipeline(
                model_dir, DATA, self.N, "fp32,int4,ternary", 64, tmp / "run"
            )
            self.assertNotIn("vs", manifest["summary"]["fp32"])
            self.assertEqual(manifest["summary"]["int4"]["vs"], "fp32")
            self.assertEqual(manifest["summary"]["ternary"]["vs"], "fp32")


if __name__ == "__main__":
    unittest.main()
