# What this project is, and what I think it means

This page is the argument behind the repository, in my own voice. The tables in
[`forward-forward-ternary.md`](forward-forward-ternary.md) are the measurements;
this is why I made them, what I think they mean, and where I think the reasoning
would break.

## Why I built it

Quantization posts usually end with a percentage. "1.58-bit weights, 89.5% on
MNIST." The number is real, but it arrives stripped of everything needed to
disagree with it: which images, which seed, which layer, what the baseline was,
whether the quantized layers actually ran.

I built QuantScope because I wanted to answer a smaller and harder question:
*what does a weight format cost, on this machine, with this data, and can someone
else check me without trusting me?* That second clause is the whole design. A run
writes down every prediction and every logit. A separate program re-derives the
accuracy from those files. A third program recomputes every published number from
the evidence bundle. Each layer can fail independently, and when it does, it says
so.

So the repository is not "here is a result". It is "here is a result, here is the
record it came from, and here is the command that will call me a liar if the two
drift apart."

## The shape of it

```text
run.py  ──►  manifest + predictions (sha256 of data, model, protocol)
                     │
                     ├─► verify   : re-derive accuracy, argmax, hashes
                     ├─► reproduce: recompute all 75 published numbers
                     └─► figures  : three SVGs, rendered from the evidence
```

Three properties I cared about, in order:

1. **Offline.** The checks need `numpy`, nothing else. No model weights, no
   network. A stranger clones the repo and runs `python -m quantscope.cli reproduce`
   in seconds.
2. **Boring on purpose.** No dashboards, no tracking, no "leaderboard". A JSON
   file, a few assertions, a non-zero exit code.
3. **Guarded by CI, not by my memory.** GitHub Actions runs the unit tests, the
   reproduction, the figure byte-comparison, and — in a separate job, so a torch
   install cannot take the evidence checks down with it — a one-minute
   Forward-Forward experiment with a direction assertion.

## What I measured

### The cost of ternarizing is not uniform

![MNIST ternary cost](img/fig-mnist-ternary-cost.svg)

Two-layer MLP, full MNIST, three seeds. FP32 reaches 94.14%; the same network
with ternary weights reaches 89.52%. That 4.62 pp is the honest headline.

But look at what the drop is made of. Only the *scale grouping* was changed:
instead of one scale per output row over the whole 794-column input, the 784 pixel
columns share one scale and each of the ten one-hot label columns gets its own.
Accuracy goes to **92.49%**, recovering **2.96 pp — 64.1% of the entire cost** —
with no extra bits, no extra parameters, no retraining schedule. A second,
independent aligned run measured 92.43% (+2.91 pp), so this is not one lucky seed.

### Bit width is the weaker lever

Also from the evidence bundle: making the first layer 2-bit instead of ternary
recovers **0.62 pp**; making both layers 2-bit recovers **1.03 pp**. Compare that
with the 2.96 pp from moving a scale boundary. On this network, *where the scale
is computed* buys roughly five times what *how many levels* buys.

![Label isolation recovery](img/fig-label-isolation-recovery.svg)

That is the finding I would defend. It is also the one I would expect to fail on
a model whose input is homogeneous — which is exactly why the next step was a
dataset where pixels and labels are not the only two kinds of input.

### The ordering survives more data, and widens

![CIFAR two training sets](img/fig-cifar-two-training-sets.svg)

On CIFAR-10, the same two-layer architecture, 20,000 training images: FP32
36.73%, per-row ternary 24.76%, aligned 36.62% (recovering 99.1% of an 11.97 pp
cost), mis-aligned 25.75%. With all 50,000 images the cost grows to **16.34 pp**
and the aligned run still recovers **97.3%** of it, while the mis-aligned run
recovers only 4.57 pp. The separation between aligned and mis-aligned grouping
goes from 10.88 pp to 11.33 pp.

More data did not shrink the problem. It made the *difference between two ways of
placing the same scale* more visible. That is the opposite of what "more data
fixes everything" would predict, and it is why I bothered to run the second
training set at all.

### A small real-model check

Qwen3-0.6B, 171 multiple-choice questions, one run per format, CPU:

| weight format | correct | accuracy | delta vs FP32 | FP32 right → wrong | FP32 wrong → right |
| --- | --- | --- | --- | --- | --- |
| FP32 | 56 / 171 | 32.75% | — | — | — |
| INT8 dynamic (PyTorch) | 45 / 171 | 26.32% | −6.43 pp | 27 | 16 |
| INT4 RTN, group 64 | 54 / 171 | 31.58% | −1.17 pp | 28 | 26 |
| ternary, group 64 | 42 / 171 | 24.56% | −8.19 pp | 49 | 35 |

Two things are true at once, and I want both on the record:

- **Ternary costs a lot here**: −8.19 pp, with 49 answers broken and 35 rescued.
- **The INT8-vs-INT4 ordering is not a result.** With 171 questions the 95%
  interval is roughly ±7 pp; the 5.41 pp gap between them sits inside it. What the
  table supports is "the dynamic-activation path introduced more error than
  weight-only rounding did on this machine", not "INT8 is worse than INT4".

## What I actually think about AI

Four claims, each tied to something above.

**1. Quantization is mostly an interface problem, not a precision problem.**
The label columns are ten columns out of 794 — about 1.3% of the input width — and
they wreck a quantiser that assumes the row is homogeneous. The fix was not a
better rounding rule; it was telling the quantiser what the input *is*. My
generalization: most "the model broke" stories in AI are interface stories —
assumptions about the shape of an input leaking into a component that was never
told about the shape. Precision gets blamed because it is easy to change.

**2. Locality is a property of the wiring, not of a layer.** Forward-Forward is
sold as "local learning", and I wanted to see how local. Moving label injection
from layer 0 to layer 1 drops accuracy to 79.92% (−9.60 pp, worse than plain
ternary), and layer-0 features alone classify at 60.90% by nearest centroid. So
layer 0 is simultaneously the bottleneck *and* the only place supervision enters.
Calling that "local" is formally true and practically misleading: the layer is
local in its gradients and global in its job.

**3. A percentage hides the thing you want to know.** "−1.17 pp" could be 5 answers
flipped out of 400, or 28 broken and 26 rescued — same number, different worlds.
Paired counts (`right→wrong`, `wrong→right`) and a sample size are the minimum
honest unit of reporting. This is not pedantry; the INT8/INT4 comparison above is
uninterpretable without it.

**4. Claims about machine intelligence need accounting, not adjectives.** What
struck me most is not that a 1.58-bit network reaches 89.5% — it is how much of
the "1.58-bit" story is really a story about *where the scale goes*. When a
dramatic-sounding property dissolves into a bookkeeping choice, that is usually
the sign of an honest measurement, not a disappointing one. I would rather publish
a boring mechanism with a ledger than an exciting number with a paragraph.

## What this does not say

- Two-layer MLPs, no convolutions, one dataset family. A transformer result would
  need its own run, not an extrapolation from this one.
- Three seeds. The 4.62 pp cost is stable across them; the third decimal place is
  not.
- The real-model table is a single run per format, on 171 questions. It is a
  sanity check, not a benchmark entry.
- INT8 here is `torch.ao.quantization.quantize_dynamic`. It is neither GGUF
  `Q4_K_M` nor bitsandbytes NF4, and nothing in this repo measures speed or memory.
- `reproduce` proves internal consistency: the published numbers agree with the
  evidence bundle and with the arithmetic in the source. It does not re-run the
  experiment. Re-running is what `experiments/ff_mnist` and the documented commands
  are for.

## Check me in three minutes

```bash
pip install quantscope
python -m quantscope.cli reproduce        # 75 numbers, from the evidence bytes
python -m quantscope.cli demo --out runs/demo
python -m quantscope.cli verify --run runs/demo/run
```

For the Forward-Forward result itself:

```bash
cd experiments/ff_mnist
python run.py --group aligned --train 4000 --test 1000 --epochs 5 --download --out runs/aligned.json
python run.py --group row     --train 4000 --test 1000 --epochs 5 --out runs/row.json
python assert_direction.py --aligned runs/aligned.json --row runs/row.json
```

That is the same pair of commands CI runs. Locally it takes about ten seconds
after the 11 MB MNIST download; the numbers will not equal the table above,
because 4,000 images is a smoke test and the table is not. What it does check is
that the mechanism still points the same way.

## Where I would go next

- A convolutional or transformer model with a heterogeneous input, to test claim 1
  outside this architecture.
- Per-group scale *learning* instead of a hand-placed boundary: if the grouping is
  what matters, a learned grouping should beat my eleven hand-cut spans.
- A proper multi-seed real-model run, so the INT8/INT4 comparison either survives
  or dies with a stated interval.

If you find a number here that does not reproduce, that is the most useful thing
you can tell me: `reproduce` exits non-zero, and the manifest has the checksums to
argue with.
