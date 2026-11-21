import unittest

from support import ROOT  # noqa: F401  (puts the repository root on sys.path)

from quantscope import protocol


class BuildPromptTest(unittest.TestCase):
    def test_template_has_one_question_and_four_labelled_options(self):
        text = protocol.build_prompt("What is 2+2?", ["3", "4", "5", "6"])
        self.assertIn("Question: What is 2+2?", text)
        for label, body in zip(protocol.LABELS, ["3", "4", "5", "6"]):
            self.assertIn(f"{label}. {body}", text)
        self.assertTrue(text.endswith("Answer:"))
        self.assertEqual(text.count("Answer:"), 1)

    def test_prompt_uses_real_newlines_not_escapes(self):
        text = protocol.build_prompt("q", ["a", "b", "c", "d"])
        self.assertNotIn(chr(92) + "n", text)
        self.assertEqual(len(text.splitlines()), 10)

    def test_mapping_form_matches_list_form(self):
        as_list = protocol.build_prompt("q", ["a", "b", "c", "d"])
        as_map = protocol.build_prompt("q", {"A": "a", "B": "b", "C": "c", "D": "d"})
        self.assertEqual(as_list, as_map)

    def test_no_few_shot_examples_and_no_thinking_marker(self):
        text = protocol.build_prompt("q", ["a", "b", "c", "d"])
        self.assertNotIn("Example", text)
        self.assertNotIn("Thinking", text)


class ValidateQuestionTest(unittest.TestCase):
    GOOD = {"id": "q1", "question": "2+2?", "options": ["3", "4", "5", "6"], "answer": "B"}

    def test_accepts_good_record(self):
        self.assertEqual(protocol.validate_question(dict(self.GOOD)), self.GOOD)

    def test_rejects_missing_key(self):
        broken = dict(self.GOOD)
        del broken["answer"]
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_question(broken)

    def test_rejects_unknown_answer_label(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_question(dict(self.GOOD, answer="E"))

    def test_rejects_three_options(self):
        with self.assertRaises(protocol.ProtocolError) as ctx:
            protocol.validate_question(dict(self.GOOD, options=["3", "4", "5"]))
        self.assertIn("4 entries", str(ctx.exception))

    def test_rejects_empty_option(self):
        with self.assertRaises(protocol.ProtocolError) as ctx:
            protocol.validate_question(dict(self.GOOD, options=["3", " ", "5", "6"]))
        self.assertIn("empty", str(ctx.exception))


class SampleCountTest(unittest.TestCase):
    def test_accepts_exact_count(self):
        protocol.require_sample_count(list(range(10)), 10)

    def test_refuses_to_shrink_the_denominator(self):
        with self.assertRaises(protocol.ProtocolError) as ctx:
            protocol.require_sample_count(list(range(5)), 10)
        message = str(ctx.exception)
        self.assertIn("requested 10", message)
        self.assertIn("only 5", message)

    def test_refuses_zero(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.require_sample_count(list(range(5)), 0)


class SourceTest(unittest.TestCase):
    REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

    def test_absent_source_is_allowed(self):
        self.assertIsNone(protocol.validate_source(None, None))

    def test_full_revision_is_accepted(self):
        self.assertEqual(
            protocol.validate_source("Qwen/Qwen3-0.6B", self.REVISION),
            {"repo": "Qwen/Qwen3-0.6B", "revision": self.REVISION},
        )

    def test_short_hash_is_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_source("Qwen/Qwen3-0.6B", self.REVISION[:7])

    def test_branch_name_is_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_source("Qwen/Qwen3-0.6B", "main")

    def test_repo_without_revision_is_rejected(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.validate_source("Qwen/Qwen3-0.6B", None)


class ArgmaxTest(unittest.TestCase):
    def test_picks_the_largest_logit(self):
        self.assertEqual(protocol.argmax_label([0.1, 2.0, -1.0, 0.5]), "B")

    def test_rejects_wrong_shape(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.argmax_label([0.1, 0.2])

    def test_rejects_non_finite_value(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.argmax_label([0.1, float("nan"), 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
