import hashlib
import unittest

import numpy as np

from support import temp_dir

from quantscope import model, quant

QUESTIONS = ["What is 2+2?", "Which planet is closest to the Sun?", "How many bits are in a byte?"]
ANSWERS = ["B", "C", "B"]
OPTIONS = ["first", "second", "third", "fourth"]


def _prompts():
    from quantscope import protocol

    return [protocol.build_prompt(q, OPTIONS) for q in QUESTIONS]


class FeatureTest(unittest.TestCase):
    def test_features_are_unit_length(self):
        from quantscope import protocol

        features = model.featurize(protocol.build_prompt(QUESTIONS[0], OPTIONS))
        self.assertEqual(features.shape, (model.DEFAULT_BUCKETS,))
        self.assertAlmostEqual(float(np.linalg.norm(features)), 1.0, places=6)

    def test_same_text_gives_same_features(self):
        first = model.featurize("hello world")
        second = model.featurize("hello world")
        self.assertTrue(np.array_equal(first, second))


class FitTest(unittest.TestCase):
    def test_same_seed_gives_identical_weights(self):
        first, _ = model.fit(_prompts(), ANSWERS, seed=7, epochs=30)
        second, _ = model.fit(_prompts(), ANSWERS, seed=7, epochs=30)
        self.assertTrue(np.array_equal(first, second))

    def test_different_seed_gives_different_weights(self):
        first, _ = model.fit(_prompts(), ANSWERS, seed=7, epochs=30)
        second, _ = model.fit(_prompts(), ANSWERS, seed=8, epochs=30)
        self.assertFalse(np.array_equal(first, second))

    def test_fit_does_not_mutate_the_inputs(self):
        prompts = _prompts()
        snapshot = list(prompts)
        model.fit(prompts, ANSWERS, seed=0, epochs=5)
        self.assertEqual(prompts, snapshot)


class StorageTest(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with temp_dir() as tmp:
            weights, bias = model.fit(_prompts(), ANSWERS, seed=0, epochs=20)
            directory = model.save_model(tmp / "model", weights, bias, model.DEFAULT_BUCKETS, 0)
            loaded = model.load_model(directory)
            self.assertTrue(np.array_equal(loaded["weights"], weights))
            self.assertTrue(np.array_equal(loaded["bias"], bias))
            self.assertEqual(loaded["config"]["kind"], model.TOY_KIND)

    def test_file_manifest_is_sorted_and_hashes_the_bytes(self):
        with temp_dir() as tmp:
            weights, bias = model.fit(_prompts(), ANSWERS, seed=0, epochs=10)
            directory = model.save_model(tmp / "model", weights, bias, model.DEFAULT_BUCKETS, 0)
            entries = model.file_manifest(directory)
            names = [entry["path"] for entry in entries]
            self.assertEqual(names, sorted(names))
            for entry in entries:
                data = (directory / entry["path"]).read_bytes()
                self.assertEqual(entry["bytes"], len(data))
                self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())


class QuantizationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.weights = rng.normal(0.0, 0.2, size=(4, 64)).astype(np.float32)

    def test_quantized_weights_keep_shape_and_dtype(self):
        for mode in quant.MODES:
            out = quant.quantize(self.weights, mode, group_size=16)
            self.assertEqual(out.shape, self.weights.shape, mode)
            self.assertEqual(out.dtype, np.float32, mode)

    def test_fp32_is_the_identity(self):
        self.assertTrue(np.array_equal(quant.quantize(self.weights, "fp32"), self.weights))

    def test_int4_stays_within_half_a_step(self):
        out = quant.quantize(self.weights, "int4", group_size=16)
        step = (np.abs(self.weights).max() / 7.0)
        self.assertLessEqual(float(np.abs(out - self.weights).max()), step / 2.0 + 1e-6)
        self.assertGreater(float(np.mean(np.abs(out - self.weights) > 0)), 0.5)

    def test_ternary_has_at_most_three_values(self):
        out = quant.quantize(self.weights, "ternary")
        self.assertLessEqual(len(np.unique(out)), 3)
        self.assertGreater(float(np.mean(np.abs(out - self.weights) > 0)), 0.5)

    def test_describe_records_the_formula(self):
        described = quant.describe("int4", 64)
        self.assertEqual(described["mode"], "int4")
        self.assertEqual(described["group_size"], 64)
        self.assertIn("[-7, 7]", described["formula"])


if __name__ == "__main__":
    unittest.main()
