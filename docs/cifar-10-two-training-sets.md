# Two training sets, one ordering

The mechanism claim in [`forward-forward-ternary.md`](forward-forward-ternary.md) is that the
first layer's ternary cost comes from one shared per-row scale mixing a loud label block
into the pixel weights. A natural objection is that it is an artifact of a small training
set: 20,000 CIFAR-10 images is a slice, and a slice can flatter a fragile effect.

So the same four variants were trained twice from scratch, once on the first 20,000 images
of the official training order and once on all 50,000, everything else held fixed. This page
is the cross-run comparison. It is a second dataset for the mechanism question, not a
CIFAR-10 benchmark.

## Setup (identical in both runs)

- Two-layer forward-forward network, 500 hidden units per layer, no convolution.
- Input: 3072 pixel values plus a 10-way one-hot label, so 3082 numbers per example.
- 5 epochs per layer, batch 128, learning rate 0.001, theta 1.0.
- Normalisation: mean [0.4914, 0.4822, 0.4465], std [0.247, 0.2435, 0.2616].
- Seeds [0, 1, 2]; the test set is the same 10,000 images in both runs.
- Only layer 0 is quantised; layer 1 keeps the per-row ternary scale from the main study.

## Accuracy on the same 10,000 test images

Mean and sample standard deviation over 3 seeds.

| variant | 20,000 images | 50,000 images | lift |
| --- | --- | --- | --- |
| FP32 | 36.73 ± 0.84 | 41.25 ± 0.28 | +4.53 |
| plain ternary (one scale per row) | 24.76 ± 1.72 | 24.92 ± 1.90 | +0.16 |
| ternary, 11 groups aligned to the input layout | 36.62 ± 0.46 | 40.82 ± 0.44 | +4.20 |
| ternary, 11 equal-width groups | 25.75 ± 1.52 | 29.49 ± 1.77 | +3.74 |

Per-seed correct counts out of 10,000 (seeds 0, 1, 2):

| variant | 20,000 images | 50,000 images |
| --- | --- | --- |
| FP32 | 3585 / 3753 / 3680 | 4099 / 4122 / 4155 |
| plain ternary (one scale per row) | 2617 / 2285 / 2526 | 2647 / 2548 / 2280 |
| ternary, 11 groups aligned to the input layout | 3712 / 3620 / 3655 | 4052 / 4062 / 4132 |
| ternary, 11 equal-width groups | 2646 / 2400 / 2678 | 3124 / 2951 / 2771 |

## What the extra data did to the gap

| quantity | 20,000 images | 50,000 images |
| --- | --- | --- |
| ternary cost vs FP32 | -11.97 pp | -16.34 pp |
| gap closed by aligned groups | +11.86 pp | +15.90 pp |
| share of the gap closed | 99.14 % | 97.35 % |
| gap closed by equal-width groups | +0.99 pp | +4.57 pp |
| aligned minus equal-width | 10.88 pp | 11.33 pp |
| aligned groups vs FP32 | -0.10 pp | -0.43 pp |

The ordering is identical in both runs, and the gap is larger, not smaller. With 2.5×
the training data the plain ternary cost grows from 11.97 to 16.34 points; aligned groups still
recover almost all of it, while equal-width groups recover a small fraction. More data
makes the first layer *more* expensive to quantise, and makes the label-block explanation
*easier* to see, because the two grouped variants separate further.

## Paired per-seed deltas inside each run

Each number is a paired difference over the 10,000 test items: the same seed, the same
items, only the weight format changed.

| run | variant | vs FP32 (seeds 0/1/2) | vs plain ternary (seeds 0/1/2) |
| --- | --- | --- | --- |
| 20,000 images | plain ternary (one scale per row) | -9.68 / -14.68 / -11.54 | this *is* the reference |
| 20,000 images | ternary, 11 groups aligned to the input layout | +1.27 / -1.33 / -0.25 | +10.95 / +13.35 / +11.29 |
| 20,000 images | ternary, 11 equal-width groups | -9.39 / -13.53 / -10.02 | +0.29 / +1.15 / +1.52 |
| 50,000 images | plain ternary (one scale per row) | -14.52 / -15.74 / -18.75 | this *is* the reference |
| 50,000 images | ternary, 11 groups aligned to the input layout | -0.47 / -0.60 / -0.23 | +14.05 / +15.14 / +18.52 |
| 50,000 images | ternary, 11 equal-width groups | -9.75 / -11.71 / -13.84 | +4.77 / +4.03 / +4.91 |

## The two runs answer different questions on different items

The two runs are independent trainings, so pairing them per seed is *not* a quality
comparison: a model trained on 20,000 images and a model trained on 50,000 disagree on
many individual items simply because the data changed. Those cross-run paired deltas are
large in both directions and are evidence of churn, not of improvement.

| variant | cross-run paired delta (seeds 0/1/2) | paired items |
| --- | --- | --- |
| FP32 | -21.91 / -15.90 / -21.31 | 2346 / 2321 / 2229 |
| plain ternary (one scale per row) | -1.12 / -15.32 / +12.97 | 2672 / 1717 / 1896 |
| ternary, 11 groups aligned to the input layout | -17.00 / -20.58 / -23.01 | 2000 / 2148 / 2073 |
| ternary, 11 equal-width groups | -18.60 / -22.74 / -4.92 | 2570 / 2423 / 1891 |

Across all 12 cross-run pairs the delta spans -23.01 to +12.97 points. Nothing here should be read
as one training set beating the other; the runs were made to test whether the mechanism
survives a change of slice, not to rank training-set sizes.

## What this page does not say

- Both runs use the same 10,000-image test set, so the two accuracies are compared on identical examples.
- The subset run uses the first 20,000 training images of the official order; the full run uses all 50,000. Any other difference between the runs is held fixed.
- Same architecture, same 5 epochs per layer, same 3 seeds; this compares slice sizes, not schedules.
- One architecture, one quantizer, one label encoding. The absolute accuracies are low
  (a two-layer forward-forward network on CIFAR-10 is weak), so read the deltas, not the levels.
- Three seeds. The ± values are sample standard deviations across seeds, not confidence
  intervals over the test set.

## Evidence

| file | what it holds | sha256 |
| --- | --- | --- |
| `docs/evidence/ff_evidence.json` | both run blocks (`cifar`, `cifar_full`), recomputed | see the bundle header |
| `docs/evidence/cifar_compare.json` | this two-run comparison, recomputed per item | `02600be368110ac1…` |

Both runs were re-audited item by item before this page was written: the stored jsonl was
re-hashed, every prediction re-derived from its `gold` label, every paired count and every
summary recomputed. The auditor is part of the laboratory repository, not this package.
