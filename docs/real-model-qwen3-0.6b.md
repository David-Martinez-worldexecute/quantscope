# Real model: Qwen3-0.6B, 171 multiple-choice questions

This page records one real evaluation so a reader can check the numbers in the README instead of
trusting them. It is deliberately narrow: one model, one subset of questions, one run, CPU only.

## What was measured

- Model: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, 9 files,
  1,519,207,673 bytes in total. `model.safetensors` is BF16, 311 tensors, 150,330,0328 bytes,
  SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Question set: 171 records from a 4-option multiple-choice set, prepared once and hashed.
  SHA-256 of the prepared file: `253da6725dcb30916b6cdd4b7704ce46d751693c0e11c4548a1f4ea27421c8d8`.
  The questions themselves are not redistributed here; see `reproduce.md` for why.
- Protocol: `quantscope-choice-logits-v2` — one chat-template prompt per question, zero examples,
  reasoning disabled, score = the single-token logit of `A`, `B`, `C` or `D` at the last position,
  prediction = argmax. No sampling, no free-form generation.
- Environment: Python 3.12.10, torch 2.15.0+cpu, transformers 5.23.0, device `cpu`, dtype
  `float32`, group size 64, INT4 bits 4, `lm_head` excluded from quantization.

## Results

| weight format | correct | accuracy | delta vs FP32 | FP32 right → wrong | FP32 wrong → right |
| --- | --- | --- | --- | --- | --- |
| FP32 | 56 / 171 | 32.7485% | — | — | — |
| INT4 RTN, group 64 | 54 / 171 | 31.5789% | −1.1696 pp | 28 | 26 |
| ternary (1.58-bit) | 42 / 171 | 24.5614% | −8.1871 pp | 49 | 7 |

The pairing counts are the part a single accuracy number hides: INT4 moves answers in both
directions almost equally (28 lost, 26 gained), which is what a small, mostly-neutral
perturbation looks like. Ternary moves 49 correct answers to wrong while recovering 7, which is
collapsing, not neutral noise.

## Evidence

All of this lives in the lab's evidence store, which is not part of this repository:

| file | bytes | SHA-256 |
| --- | --- | --- |
| `manifest.json` | 18,842 | `eb5c31273c3cfc6e1da17af39c0ec429449ff233eed22c721c3800691f24d8de` |
| `baseline.jsonl` | 36,408 | `9fcc7c6e068a4d0d78274d793477a59b43ba1013309b324979b07cddb8aa06be` |
| `int4.jsonl` | 36,448 | `1904406ec6b42205a58d2b09faec461f2f5368870f7a91dbb8b8f0b227693c1f` |
| `ternary.jsonl` | 36,910 | `2ce3d59d3c777bc2b3da867e4f564d65ffa3bebca574018d3534e2418fbe2b33` |

The manifest records, per mode, every item's id, gold answer, prediction, correctness and the four
logits, plus the SHA-256 of the model files it loaded and of the question file.

## Independent re-check

The run was re-checked by a separate script that never imports the runner. It re-derives each
prediction as the argmax of the recorded logits, re-counts correct answers and accuracy,
re-computes both pairing tables and the deltas, and re-hashes the nine model files against the
hashes in the manifest.

Outcome: `status=pass`, `count=171`, `mismatches=[]`, `model_files_verified=9`. Recomputed
correct counts 56 / 54 / 42 and both pairing tables match the manifest exactly.

## What this does not say

- It is **one run**. There are no seeds here, so the ± std in the Forward-Forward study has no
  counterpart on this page. Do not attach error bars that were never measured.
- 171 items is small: at 32.75% baseline accuracy, the 95% interval around that accuracy is roughly
  ±7 percentage points. The −1.17 pp INT4 delta is inside that noise; the −8.19 pp ternary delta is
  not, but one run still cannot separate it from item selection.
- It is not a benchmark ranking and not comparable to published leaderboard numbers, which use
  different prompts, extraction rules and subset sizes.
- It says nothing about speed, memory or throughput; only about which option the model picks.
- These numbers came from the lab harness that shares this protocol. This package's `run` command
  implements the same protocol on the documented toy format; loading Hugging Face checkpoints is
  the next API change, not a property of v0.1.0.
