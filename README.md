# QuantScope

**Measure what a quantization mode costs in accuracy — then prove it later.**

QuantScope runs the same multiple-choice questions through several weight formats of the same
model, records every prediction together with the four choice logits, and writes a manifest that
a separate `verify` command can re-check against the recorded bytes. It is a measurement tool,
not a leaderboard: it reports pairwise flips and a delta in percentage points, and refuses to
guess when the sample count does not match what was asked for.

## Install

```bash
pip install quantscope
```

Python 3.10+. The core needs `numpy` only and runs offline on CPU.

## Quickstart

One command builds a tiny deterministic model, evaluates three weight modes and prints a table:

```bash
quantscope demo --out runs/demo
```

Then run the same pipeline against any model directory in the documented toy format, verify the
run, and rebuild a self-contained HTML report:

```bash
quantscope run --model runs/demo/model --data data/sample_questions.jsonl \
    --n 20 --modes fp32,int4,ternary --out runs/example
quantscope verify --run runs/example
quantscope report --run runs/example
```

`verify` re-derives every prediction from the stored logits and re-checks the summary
arithmetic, the file hashes and the `complete` flag. It prints `PASS` or lists the mismatches,
and its exit code is the verdict.

## Results

The tool is the deliverable; the numbers below come from a separate controlled study that ships
in `docs/forward-forward-ternary.md`, with per-item evidence in
`docs/evidence/ff_controlled_evidence.json`.

| weight format | MNIST test accuracy (mean ± std over 3 seeds) |
| --- | --- |
| FP32 | 94.14% ± 0.18 |
| all-ternary | 89.52% ± 0.48 |
| first layer FP32, second ternary | 93.57% ± 0.19 |
| first ternary, second layer FP32 | 90.43% ± 0.27 |

Same initialization per seed (verified by state hash), 12 training units, 120,000 per-item
predictions, independently re-audited. The study is about a small forward-forward network on
MNIST, not about a language model.

### Real model: Qwen3-0.6B, 171 multiple-choice questions

Measured once, on CPU, in float32, with the same prompt protocol this repository documents
(one template, A/B/C/D single-token logits, no generated reasoning):

| weight format | correct | accuracy | delta vs FP32 | flips (FP32 right→wrong / wrong→right) |
| --- | --- | --- | --- | --- |
| FP32 | 56 / 171 | 32.75% | — | — |
| INT4 RTN, group 64 | 54 / 171 | 31.58% | −1.17 pp | 28 / 26 |
| ternary (1.58-bit) | 42 / 171 | 24.56% | −8.19 pp | 49 / 35 |

Read it as one run on one 171-item set, not as a benchmark ranking: there is no seed variance
here, the interval around a 171-item accuracy is wide, and the base model answers only a third of
these questions correctly in the first place. What the numbers do support is narrower and still
useful: INT4 round-to-nearest cost about one point on this set, while ternary cost about eight,
and the flips are not symmetric — ternary turned 49 correct answers wrong while recovering 7.

These numbers were produced by the laboratory harness that shares this protocol, not by this
package's `run` command: v0.1.0 loads only the documented toy format. The per-item evidence,
the manifest hash and the independent re-check script live in `docs/real-model-qwen3-0.6b.md`.


## Reproduce

See `docs/reproduce.md`: two commands, the pinned environment, and the checks each one performs.
The repository ships no model weights and no third-party benchmark items; `data/sample_questions.jsonl`
contains 20 questions written for this repository.

## Limits

- Accuracy on a multiple-choice set is the only thing measured. Throughput, latency, memory and
  perplexity are out of scope.
- The `demo` model is a synthetic toy used to demonstrate the pipeline. Its numbers say nothing
  about real language models.
- v0.1.0 loads the documented toy format. Loading Hugging Face checkpoints is the next release and
  will be recorded in `docs/reproduce.md` with its own evidence, not announced here in advance.
- A run is only comparable to another run with the same protocol id, the same question file hash
  and the same sample count; the manifest records all three so a mismatch is visible.

## License

MIT. See `LICENSE`.
