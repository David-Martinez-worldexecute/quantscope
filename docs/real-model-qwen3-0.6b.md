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
| FP32 | 56 / 171 | 32.7485% | — | — |
| INT8 dynamic (PyTorch) | 45 / 171 | 26.3158% | −6.4327 pp | 27 | 16 |
| INT4 RTN, group 64 | 54 / 171 | 31.5789% | −1.1696 pp | 28 | 26 |
| ternary (1.58-bit) | 42 / 171 | 24.5614% | −8.1871 pp | 49 | 7 |

What the numbers do support is narrower and still useful. PyTorch's own dynamic INT8 — the
quantiser a reader would reach for first — moved this set by −6.43 pp, INT4 round-to-nearest
by −1.17 pp, and ternary by −8.19 pp. That ordering is the useful part: it says how much of
the number is the method and how much is simply "low bits cost something here". The flips
are not symmetric either — ternary turned 49 correct answers wrong while recovering 35.

## Evidence

All of this lives in the lab's evidence store, which is not part of this repository:

| file | bytes | SHA-256 |
| --- | --- | --- |
| `manifest.json` | 26,771 | `f01467c118cdba3ed70048b7f70098d164970b1218b732f7de905684a9ca8407` |
| `baseline.jsonl` | 36,412 | `8f734c5f719809142d1393cb0bc408014c32a5d2d8b93c3b9e5dd1b14b4c3968` |
| `int8.jsonl` | 36,368 | `c6803068a12ab0fcc48fbbbbee70282f0810bdc58456d0f3d28108818cc0142d` |
| `int4.jsonl` | 36,432 | `1ca5f4179e4a5db1c38b001780da84d853850213a5f5c2598d01a773877dd0e8` |
| `ternary.jsonl` | 36,927 | `5a6e8ea338e355b685c7326538d76528ec6c51b1b079ae6fdd00e775c30ce7ae` |

The manifest records, per mode, every item's id, gold answer, prediction, correctness and the four
logits, plus the SHA-256 of the model files it loaded and of the question file.

Those five files stay in the lab's store. What this repository carries is their digest:
`docs/evidence/real_model_qwen3_0.6b.json` (`quantscope-real-model-v1`), 6,609 bytes, holding —
for all four modes — the correct count, the accuracy, the pairing table against FP32 and the nine
model-file hashes. `python -m quantscope.cli reproduce` recomputes every one of those numbers from
the integers stored in that same file: `accuracy == correct/count`, the four pairing cells sum to
171, `delta_pp == 100*(gained-lost)/171`, and each mode's accuracy equals FP32's count minus the
questions it lost plus the ones it gained. Editing any single number in the bundle turns the
command red.

## Independent re-check

The run was re-checked by a separate script that never imports the runner. It re-derives each
prediction as the argmax of the recorded logits, re-counts correct answers and accuracy,
re-computes both pairing tables and the deltas, and re-hashes the nine model files against the
hashes in the manifest.

Outcome: `status=pass`, `count=171`, `mismatches=[]`, `model_files_verified=9`. Recomputed
correct counts 56 / 45 / 54 / 42 and all pairing tables match the manifest exactly. The mode
list is read from the manifest, so a new mode cannot be skipped silently.

## What this does not say

- The INT8 row is `torch.ao.quantization.quantize_dynamic` (per-channel symmetric int8
weights, activation scales computed at run time), i.e. the framework default. It is **not**
GGUF `Q4_K_M` and **not** bitsandbytes NF4: those formats were not available in this
environment, so no claim is made about them, and nothing here measures speed or memory.
- It is **one run**. There are no seeds on this page, so the ± standard deviation reported in
the Forward-Forward study has no counterpart here; do not attach error bars that were never
measured.
- 171 items is small: at 32.75% baseline accuracy the 95% interval around that accuracy is
roughly ±7 percentage points. The −1.17 pp INT4 delta and the −6.43 pp INT8 delta both sit
inside that noise; only the −8.19 pp ternary delta is outside it, and even that cannot be
separated from item selection on a single run.
- It is not a benchmark ranking and is not comparable to published leaderboard numbers,
which use different prompts, extraction rules and subset sizes.
- It says nothing about speed, memory or throughput; only about which option the model
picks.
- These numbers came from the lab harness that shares this protocol. This package's `run`
command implements the same protocol on the documented toy format; loading Hugging Face
checkpoints is the next API change, not a property of v0.1.0.
