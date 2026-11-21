import hashlib
import json
import shutil
import unittest
from pathlib import Path

from support import DATA, new_dir

from quantscope import model, protocol
from quantscope.run import load_questions, run as run_pipeline
from quantscope.verify import verify


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ManifestTest(unittest.TestCase):
    N = 8

    @classmethod
    def setUpClass(cls):
        cls.root = new_dir("manifest-")
        questions = load_questions(DATA)[: cls.N]
        prompts = [protocol.build_prompt(row["question"], row["options"]) for row in questions]
        weights, bias = model.fit(
            prompts, [row["answer"] for row in questions], seed=0, epochs=80
        )
        cls.model_dir = model.save_model(
            cls.root / "model", weights, bias, model.DEFAULT_BUCKETS, 0
        )
        cls.manifest, cls.run_dir = run_pipeline(
            cls.model_dir, DATA, cls.N, "fp32,int4,ternary", 64, cls.root / "run"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_schema_and_protocol_identity(self):
        self.assertEqual(self.manifest["schema"], "quantscope-run-v2")
        self.assertEqual(self.manifest["protocol_id"], protocol.PROTOCOL_ID)
        self.assertEqual(self.manifest["cli_version"], "0.1.0")

    def test_complete_flag_is_true(self):
        self.assertIs(self.manifest["complete"], True)

    def test_every_mode_covers_every_question(self):
        for mode in self.manifest["quant"]["modes"]:
            self.assertEqual(len(self.manifest["items"][mode]), self.N, mode)

    def test_model_files_are_recorded_with_hashes(self):
        files = self.manifest["model"]["files"]
        self.assertTrue(files)
        for entry in files:
            self.assertEqual(len(entry["sha256"]), 64, entry["path"])
            self.assertGreater(entry["bytes"], 0, entry["path"])

    def test_recorded_data_hash_matches_the_question_file(self):
        self.assertEqual(self.manifest["data"]["sha256"], sha256_of(DATA))
        self.assertEqual(self.manifest["data"]["n"], self.N)

    def test_environment_is_recorded(self):
        env = self.manifest["env"]
        self.assertEqual(env["device"], "cpu")
        self.assertTrue(env["python"].startswith("3."))
        self.assertTrue(env["numpy"])

    def test_run_directory_holds_manifest_and_per_mode_items(self):
        self.assertTrue((self.run_dir / "manifest.json").is_file())
        for mode in self.manifest["quant"]["modes"]:
            self.assertTrue((self.run_dir / f"items_{mode}.jsonl").is_file(), mode)

    def test_verifier_accepts_an_untouched_run(self):
        result = verify(self.run_dir, self.model_dir)
        self.assertEqual(result["status"], "pass", result["issues"])
        self.assertEqual(result["issues"], [])

    def _copy_run(self, name):
        target = self.root / name
        shutil.copytree(self.run_dir, target)
        return target

    def test_verifier_reports_a_flipped_prediction(self):
        target = self._copy_run("tampered-pred")
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        item = manifest["items"]["int4"][0]
        item["pred"] = "A" if item["pred"] != "A" else "B"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = verify(target, self.model_dir)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["issues"])

    def test_verifier_reports_a_missing_complete_flag(self):
        target = self._copy_run("missing-complete")
        path = target / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["complete"] = False
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        result = verify(target, self.model_dir)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["issues"])


if __name__ == "__main__":
    unittest.main()
