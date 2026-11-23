# What each weight format costs — in bytes, seconds and accuracy

Accuracy is half the question. A format that keeps 99% of the answers but does not fit on
the machine, or makes it four times slower, is not a trade-off you can take. This page is
the other half, measured on the same run as [the accuracy page](real-model-qwen3-0.6b.md).

## The short version

The four rows below are the same run as the accuracy page: one Qwen3-0.6B (float32, cpu, 4
threads), 128-token prompt, 32 greedy tokens.

Payload falls from 2274 MiB to 725 MiB (68.1% smaller) for the three-value format, but on
this CPU only INT8 dynamic (PyTorch) is measurably faster at prefill.

Decode is the same story: INT8 dynamic (PyTorch) reaches 8.58 tok/s against 5.19 tok/s for
FP32.

## Payload and speed

| format | payload | vs FP32 | load + quantise | prefill (median) | decode | replaced layers |
| --- | --- | --- | --- | --- | --- | --- |
| FP32 (unquantised) | 2274 MiB | 1.00× | 5.64 s | 1104 ms | 5.19 tok/s | 0 |
| INT8 dynamic (PyTorch) | 1016 MiB | 0.45× | 10.33 s | 561 ms | 8.58 tok/s | 196 |
| INT4 RTN, groups of 64 | 830 MiB | 0.37× | 1.74 s | 1136 ms | 4.93 tok/s | 196 |
| ternary RTN, groups of 64 | 725 MiB | 0.32× | 1.78 s | 1122 ms | 5.01 tok/s | 196 |

Payload is an estimate computed from the layer shapes, not the size of a file on disk: `1`
byte per weight for INT8, `0.5` for INT4, `0.25` for ternary, each plus a 4-byte scale per
64-value group, and 4 bytes for the weights that are never quantised (`lm_head` and the
embeddings). The bundle's own `limits` say this in the machine-readable form, and
`quantscope reproduce` re-derives the numbers from those limits.

## Accuracy, from the same run

| format | correct | accuracy | 95% interval |
| --- | --- | --- | --- |
| FP32 (unquantised) | 56 / 171 | 32.75% | 25.7–39.8% |
| INT8 dynamic (PyTorch) | 45 / 171 | 26.32% | 19.9–33.3% |
| INT4 RTN, groups of 64 | 54 / 171 | 31.58% | 24.6–38.6% |
| ternary RTN, groups of 64 | 42 / 171 | 24.56% | 18.1–31.0% |

Intervals are bootstrapped from the per-question 0/1 outcomes, seed 12345, 4000 resamples.
They overlap heavily: 171 questions cannot separate three formats that sit within a few
points of each other. That is the honest reading, and it is why the page reports intervals
next to the point estimates.

## Why the small formats are not faster here

INT4 and ternary in this repository are *simulated*: the weights are rounded to 4 or 2 bits
and immediately multiplied back into float32, so the arithmetic is still float32 and the
model still occupies float32 memory. The payload column is what those formats *would* cost
if stored that way; the time columns are what this implementation actually costs. Only the
INT8 row runs a real quantised kernel (PyTorch's dynamic quantisation), which is why it is
the one row that beats FP32 on prefill.

So the comparison is deliberately unfair in one direction and honest about it: the low-bit
rows pay the accuracy price of quantisation without collecting the speed prize. A real
deployment would use a packed kernel; this repository measures the accuracy question, and
the payload question analytically, and says which is which.

## What this is not

- Not a GPU result. One CPU, threads fixed in the bundle's `environment` block.
- Not a throughput benchmark. 32 decoded tokens per pass, three passes, fastest kept.
- Not a memory measurement of the quantised model. Peak RSS includes the float32 weights that this implementation keeps in place.

## Reproduce it

```bash
python -m quantscope.cli reproduce        # re-derives every number on this page
python scripts/make_figures.py --check    # the figure is byte-identical to the bundle
```

The page itself is generated from the two bundles by a lab-side script, so a number here that disagrees with `docs/evidence/` cannot survive a rebuild.
