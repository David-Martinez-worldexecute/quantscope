# What ternary weights cost two layers of Forward-Forward

A four-way ablation on MNIST: same network, same data, same seed, only the weight
format changes. The interesting part is not "ternary is worse" — it is *which
layer* pays.

## Setup

Two-layer Forward-Forward network. Each layer is trained for 5 epochs with its
own local objective (positive samples get a higher goodness than negative ones,
threshold θ = 1.0), batch 128, Adam at lr = 0.001. Input is 794 wide: 784 pixel
features plus a 10-dimensional one-hot label, normalised with mean 0.1307 and
standard deviation 0.3081. Hidden width 500. MNIST 60,000 train / 10,000 test.

Four weight configurations, each run under seeds 0, 1 and 2:

| configuration | layer 1 | layer 2 |
|---|---|---|
| `fp32` | float32 | float32 |
| `ternary` | ternary | ternary |
| `first_fp32` | float32 | ternary |
| `second_fp32` | ternary | float32 |

Within a seed, all four configurations start from the **same initial weights**.
That is not a claim in prose: the run recorded `initial_state_sha256` and
`initial_rng_sha256` per cell, and the three values that belong to one seed are
identical across the four formats. So a difference between two rows below is a
difference in weight format, not in initialisation luck.

## Accuracy, 10,000 test images per cell

| configuration | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| `fp32` | 9,406 | 9,402 | 9,435 | **94.14 ± 0.18** |
| `ternary` | 8,994 | 8,900 | 8,963 | **89.52 ± 0.48** |
| `first_fp32` | 9,341 | 9,353 | 9,378 | **93.57 ± 0.19** |
| `second_fp32` | 9,066 | 9,014 | 9,049 | **90.43 ± 0.27** |

Keeping the **first** layer in float32 recovers most of what ternary costs:
94.14 → 89.52 is a 4.62 point drop, and `first_fp32` sits 0.57 points below the
all-float baseline, i.e. about 88% of the gap recovered. Keeping the **second**
layer in float32 instead recovers about 20% (`second_fp32` is 3.71 points below
baseline). On this setup the first layer is the expensive one.

## Where the errors actually move

Accuracy deltas hide the direction of change, so each cell was paired with its
own `fp32` counterpart question by question. `n01` counts questions the baseline
got wrong and the variant got right; `n10` is the reverse.

| variant | seed | n01 | n10 | Δ accuracy (pp) |
|---|---|---|---|---|
| `ternary` | 0 | 107 | 519 | −4.12 |
| `ternary` | 1 | 86 | 588 | −5.02 |
| `ternary` | 2 | 66 | 538 | −4.72 |
| `first_fp32` | 0 | 42 | 107 | −0.65 |
| `first_fp32` | 1 | 20 | 69 | −0.49 |
| `first_fp32` | 2 | 30 | 87 | −0.57 |
| `second_fp32` | 0 | 112 | 452 | −3.40 |
| `second_fp32` | 1 | 99 | 487 | −3.88 |
| `second_fp32` | 2 | 85 | 471 | −3.86 |

Totals: `ternary` 259 fixed / 1,645 broken, `first_fp32` 92 / 263,
`second_fp32` 296 / 1,410. Ternary is not mostly swapping answers — it is
mostly losing answers it used to get, and keeping the first layer in float32
cuts that loss by roughly an order of magnitude.

## What this does not say

* **One dataset, one architecture, one input encoding.** MNIST with a label
  appended to the input is a forgiving setup; the numbers should not be
  extrapolated to transformers, to larger widths, or to other quantizers.
* **Three seeds are three paired runs, not thirty thousand independent samples.**
  The ± values are sample standard deviations across seeds; the pairing is what
  makes the deltas meaningful.
* **The mechanism is a hypothesis.** "The first layer sees raw pixels and the
  one-hot label, so quantisation there destroys information the second layer
  cannot rebuild" fits the data, but this experiment does not isolate it. A
  width sweep or an activation-level probe would.
* **Accuracy only.** No throughput, no memory, no energy. Ternary weights here
  are float32 tensors holding three distinct values; nothing about this measures
  the speed benefit of real low-bit kernels.

The per-item evidence behind every number above is in
`docs/evidence/ff_controlled_evidence.json`; `docs/reproduce.md` says how it was
produced and re-checked.
