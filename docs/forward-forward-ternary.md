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

## Where the first-layer cost comes from

Three follow-up runs try to explain the 4.62 points, and one tries to fix them.
All three use the same architecture, data and training budget as above; the
seeds are the same paired seeds.

**The scale is shared, and the label columns are loud.** Ternary weights use one
scale per output row, `mean(|w|)` over all 794 input columns. Ten of those
columns carry the one-hot label: 1.26% of the layer-1 weights, but their mean
column norm is 30.53 against 0.95 for a pixel column — about 32×. Averaging over
the row therefore sets a threshold that suits the label columns and is far too
coarse for the pixels. In the float32 model, row-wise ternarisation of layer 1
sets **49.9%** of its weights to zero (layer 2: 36.8%), with a weight relative
error of 0.94 and an output relative error of 0.57 (layer 2: 0.59 / 0.48).

**Give the label block its own scale and most of the gap closes.** Computing the
scale separately for the 784 pixel columns and the 10 label columns *inside the
training loop* (straight-through, so the float32 master weights still learn)
reaches **92.49 ± 0.15** — 2.97 points above all-ternary, i.e. **64% of the
4.62-point gap**. It is still 1.09 points below `first_fp32`, so the rest is not
a scale-placement problem.

**Bit width is not the bottleneck.** Replacing the three levels
{-1, 0, +1}·scale with four, {-1.5, -0.5, +0.5, +1.5}·scale — no zero level, so
the zeroed half of layer 1 comes back — reaches 90.14 ± 0.31 for the first layer
only (**+0.62** points) and 90.56 ± 0.27 for both layers (**+1.03**). Doubling
the code points buys a sixth of what moving the scale buys.

**Moving the label out of layer 1 is worse, not better.** Feeding the one-hot
label to layer 2 instead, and calibrating layer 1 without labels (its only
option once the labels are gone), drops to **79.92 ± 0.44**, 9.60 points below
all-ternary. This run changes two things at once — label position and the layer-1
objective — so it is directional evidence, not a clean single-variable result.
The reading it supports: the label columns are not only the thing that hurts
quantisation, they are also where the supervision enters.

**More layers do not help.** Repeating the comparison at two, three and four
trained layers — same data, same five epochs per layer, same two paired seeds —
the float32 network gains a little from depth (94.04 → 94.45 → 94.81) while the
all-ternary network stays flat (89.47 → 89.82 → 89.80). The gap therefore does
not shrink: 4.57 → 4.63 → 5.01 points, with paired per-seed deltas of
−4.12 / −5.02, −4.11 / −5.15 and −4.44 / −5.57. Extra depth buys nothing here,
which points at a cost charged per quantised layer rather than at a capacity
shortage.

**What the grouping buys is alignment, not count.** Eleven groups recover most of
the loss only if the boundaries fall on the label block: 784 pixels as one group
plus each of the ten label columns on its own reaches 92.43 ± 0.13, while eleven
equal-width groups of the same 794 columns — boundaries at 72, 144, … that cut
through the pixel block and never separate the labels — reach 91.03 ± 0.24. Same
group count, **+1.40** points from alignment alone. Splitting the pixel side
further (four 196-column blocks, 14 groups) adds 0.13 points, so pixel
granularity is not what matters either. This is the single-variable check the
earlier blockwise run could not provide, and it puts the recovery on the label
block being kept out of the pixel scale.

**The same comparison on a harder dataset makes the point sharper.** Swapping
MNIST for CIFAR-10 (3,072 pixels, 20,000 training images, otherwise the same two
layers, 500 hidden units, five epochs per layer and three seeds) the plain
per-row ternary first layer costs 11.97 points: float32 reaches 36.73 ± 0.31,
ternary 24.76 ± 0.86. Keeping each label column out of the pixel scale — one
3,072-pixel group plus ten single-column groups — puts it back at 36.62 ± 0.23,
within 0.11 points of float32, with paired per-seed deltas against float32 of
+1.27 / −1.33 / −0.25 and against plain ternary of +10.95 / +13.35 / +11.29.
Eleven equal-width groups again stay near ternary at 25.75 ± 0.76 (+0.29 / +1.15
/ +1.52 against ternary). So the alignment effect is not an MNIST artefact, and
here it recovers 99% of the gap. The absolute numbers are low — a 2-layer
Forward-Forward network on CIFAR-10 is a weak model — so this is a mechanism
result, not a claim that the setup works well on CIFAR.

**More training data widens the gap instead of closing it.** The same four
variants re-run on the full 50,000-image training set (same architecture, same
five epochs per layer, same three seeds, same 10,000 test images; the only change
is the training subset) give float32 41.25 ± 0.28,
plain ternary 24.92 ± 1.90, aligned
40.82 ± 0.44 and equal-width
29.49 ± 1.77. The extra data lifts float32
by 4.53 points and aligned ternary almost as much (+4.20)
but leaves plain ternary where it was (+0.16), so the ternary
cost grows from 11.97 to 16.34 points while the
label-column alignment still recovers 97% of it (+15.90 points,
ending 0.43 points short of float32). Equal-width groups do improve,
from 25.75 to 29.49, but recover only +4.57
points, so the aligned-unaligned separation widens from 10.88 to 11.33
points. The ordering of the four variants is identical on both training sets.

**Layer 1 does not repeat the story.** Layer 1 never sees the label columns: the
10 one-hot columns are concatenated at the input of layer 0 only, and layer 1
receives the 500 activations of layer 0. Grouping layer 1 into 10 or 25 blocks
on top of an aligned layer 0 gives 92.50 and 92.44 against 92.43 for aligned
layer 0 alone — about +0.07 points, inside seed noise. Grouping layer 1 only,
leaving layer 0 on a single per-row scale, gives 89.49, indistinguishable from
plain ternary at 89.52. The recovery therefore belongs to the label block at the
input of the first layer and does not transfer to a layer that has no label
columns to isolate.

Taken together: the first layer is expensive because a shared per-row scale
mixes a 32× louder label block into the pixel weights, and because that label
block is the only place the layer gets its signal. Fixing the scale recovers
about two thirds of the gap without touching the labels; removing the labels
costs far more than the gap.



* **One dataset, one architecture, one input encoding.** MNIST with a label
  appended to the input is a forgiving setup; the numbers should not be
  extrapolated to transformers, to larger widths, or to other quantizers.
* **Three seeds are three paired runs, not thirty thousand independent samples.**
  The ± values are sample standard deviations across seeds; the pairing is what
  makes the deltas meaningful.
* **The mechanism is only partly isolated.** The next section tests it with
  four follow-up runs and a depth sweep; together they explain about two thirds
  of the first-layer cost, show that neither bit width nor more layers closes
  the gap, and leave the residual ~1.1 points open.
* **Accuracy only.** No throughput, no memory, no energy. Ternary weights here
  are float32 tensors holding three distinct values; nothing about this measures
  the speed benefit of real low-bit kernels.
* **The CIFAR-10 numbers are two runs, not one trained longer.** One uses a
  20,000-image prefix of the official training order and one uses all 50,000;
  they are independent trainings, so the questions the two models answer
  differently are answer churn, not a quality comparison. Both are a second
  dataset for the mechanism question, not a CIFAR-10 benchmark, with the same
  three seeds and the same five epochs per layer.

The per-item evidence behind the four-way ablation is in
`docs/evidence/ff_controlled_evidence.json`; the mechanism runs above, their
paired counts and the depth sweep are in `docs/evidence/ff_evidence.json`.
`docs/reproduce.md` says how both were produced and re-checked.
