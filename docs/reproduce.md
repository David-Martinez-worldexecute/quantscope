# Reproducing the numbers

Two kinds of numbers live in this repo, and they are reproduced in two different
ways. Neither one needs a GPU, and neither one downloads a model.

## 1. The QuantScope pipeline (this repo, toy model, about a second)

The demo trains a tiny four-way classifier on the 20 sample questions, then runs
the same questions through three weight formats. It is a pipeline test, not a
benchmark: training and evaluation use the same 20 questions, so 100% means
"the plumbing works", not "quantization is free".

```bash
python -m quantscope.cli demo --out runs/demo
python -m quantscope.cli verify --run runs/demo/run
python -m quantscope.cli report --run runs/demo/run
```

`verify` re-derives every prediction and every summary number from the stored
logits and compares them with what the manifest claims. Exit code 0 is the pass
signal; the printed status is for humans.

To use your own questions, write a JSONL file with one object per line:

```json
{"id": "q1", "question": "2 + 2 = ?", "options": ["3", "4", "5", "6"], "answer": "B"}
```

and point `--data` at it. `--n` defaults to the number of lines; asking for more
samples than the file contains is an error, never a silent smaller denominator.

For a real Hugging Face checkpoint there is one more step, which is *not* part
of v0.1.0: the model loader only understands the toy format. The roadmap section
of the README says so plainly rather than shipping a half-tested path.

## 2. The Forward-Forward ablation (the linked study, minutes on a laptop CPU)

The evidence file is `docs/evidence/ff_controlled_evidence.json`: four weight
configurations, three seeds each, 10,000 test images per cell, 120,000 per-item
predictions re-checked by an independent script. Its `summary` block is the only
source for the table in the README and in `forward-forward-ternary.md`.

The run was produced with Python 3.12.10 and torch 2.6.0+cu124 on an RTX 4060
Laptop (8 GB), CPU path also valid but slower. The training script lives with the
rest of the study rather than here, because it needs MNIST and a two-layer
Forward-Forward network that has nothing to do with this package; what this repo
keeps is the evidence and the claim, not the training harness.

If you re-run it and get different digits, that is a finding, not a formatting
problem: the four configurations share their initial weights per seed, and the
evidence file records the initial-state hashes so a mismatch can be attributed
to the environment rather than argued about.

## 3. The real-model evaluation (Qwen3-0.6B, CPU, one run)

`docs/real-model-qwen3-0.6b.md` is the receipt for the real-model table in the
README: 171 four-option questions, three weight formats, CPU float32, the
protocol this repo documents. It lists the SHA-256 of the manifest, of the three
per-item prediction files and of the prepared question file, so a copy of the
evidence can be checked without trusting this page.

Two things are deliberately *not* in this repository:

- **The question text.** The 171 items come from a research-licensed evaluation
  set. Redistributing them is not covered by that licence, so the repo ships the
  hash and the counts, and anyone who has the set can prepare it the same way and
  compare the hash before trusting the numbers.
- **The model weights.** 1.5 GB of BF16 is a download, not a git artifact. The
  manifest records the SHA-256 of all nine files it loaded, so a locally obtained
  copy can be verified file by file.

The evaluation itself was produced by the lab harness that shares this protocol,
not by this package's `run` command, which loads only the toy format in v0.1.0.
Say so when citing the numbers: "same protocol, different implementation" is a
weaker claim than "this package produced it", and it is the true one.
