# Reproducing the numbers

Two kinds of numbers live in this repo, and they are reproduced in two different
ways. Neither one needs a GPU, and neither one downloads a model.

## 0. Every published number, one command, no downloads

```bash
python -m quantscope.cli reproduce            # or: reproduce --evidence <path> --out report.json
```

This reads `docs/evidence/` and recomputes 75 numbers: the per-seed mean accuracy of
each precision mode, the per-block `accuracy_mean` of every ablation, and all 21 `headline_pp`
percentages the README and the study pages quote. Each check compares the stored value with the value
recomputed from the counts sitting next to it; the last line prints how many numbers were checked and
how many mismatched, and the exit code is 0 only when nothing mismatched.

What it cannot do: it does not retrain, does not re-run inference, and does not read a model. It proves
the published numbers are *internally* consistent with their own per-item counts and with each other —
that is the claim the pages make. Re-running the study end to end is section 2 below.

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

### 2b. The mechanism runs (why layer 1 is the expensive one)

`docs/evidence/ff_evidence.json` covers the follow-ups described in
`forward-forward-ternary.md`: the per-layer quantisation diagnostics, the same
scale split moved into the training loop, two bit-width variants, the
label-at-second-layer run, the depth sweep, and the question-by-question
reproduction check against the earlier post-hoc patch. It is generated on the
lab side from the run manifests — every number and every paired count is
recomputed from those files, not transcribed — and it deliberately contains no
local paths, only labels, byte sizes and SHA-256 hashes of its sources. A depth
sweep appears under `depth` when that run has finished; if the key is absent, the
sweep has not been included yet, and the document says so rather than implying it.
The same bundle carries the second-dataset replication twice: `cifar` is the
run on the first 20,000 training images and `cifar_full` repeats it on all
50,000, both with four variants and three seeds, so the pair answers whether the
ordering survives more data. The layer-1 granularity runs sit under
`group_two_layers` (two runs: one finished, one stopped early and shipped with
its own status note, because a run that did not finish must not look like a
finished one). `docs/cifar-10-two-training-sets.md` reads both CIFAR-10 runs in
one place; its numbers are recomputed from `docs/evidence/cifar_compare.json`,
which is the cross-run comparison written by the lab-side script, copied
byte-for-byte.

### 2c. The 0.8B load and forward measurement (real weights, one CPU run)

`docs/real-model-qwen3.5-0.8b.md` reports one load and one prefill of the public
`Qwen/Qwen3.5-0.8B` snapshot: 1.75 GB of weights, both files re-hashed before loading, 95.64 s to
materialize, a peak working set of 0.38 → 3.42 → 3.49 GiB, and a 64-token prefill of 0.38 s
(median of three) on CPU in bfloat16.

The repository ships no weights. The numbers come from a lab-side script that reads the downloaded
shard, verifies the two SHA-256 values, records the process peak working set through
`psapi.GetProcessMemoryInfo` and writes `qwen35_forward.json`; the published file
`docs/evidence/qwen35_forward.json` is a byte copy of that output with the local paths and the
token preview removed. `scripts/check_public.py` re-checks its internal consistency: two weight
files with 64-hex hashes, file sizes that sum to the recorded total, finite logits whose second
dimension equals the prompt length, a median that is the middle of the three timings, and peak
memory that grows monotonically through load and forward.

What it cannot show: GPU memory, generation throughput, cache growth, or answer quality.

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
