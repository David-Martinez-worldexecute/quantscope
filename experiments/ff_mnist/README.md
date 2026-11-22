# A one-command reproduction of the Forward-Forward result

Everything in this folder exists so that a stranger — and CI — can run the
experiment behind `docs/forward-forward-ternary.md` instead of taking the table on
trust.

```bash
python run.py --group aligned --train 4000 --test 1000 --epochs 5 --download --out runs/aligned.json
python run.py --group row     --train 4000 --test 1000 --epochs 5 --download --out runs/row.json
python assert_direction.py --aligned runs/aligned.json --row runs/row.json
```

`--download` fetches the four official MNIST idx files (~11 MB gzipped) into
`runs/mnist/` once; after that the two runs above take about five seconds each on
a laptop CPU. It tries two public mirrors in turn and streams in 1 MB chunks, so a
slow link still prints progress instead of looking frozen — the 47 MB training
images are the slow part, not the training.
Dependencies: `torch` and `numpy`. No torchvision, no PIL, no dataset library —
`data.py` is four `struct.unpack` calls and a deterministic subsample.

## What is being compared

The network is a two-layer Forward-Forward net. Its input is
`[784 pixels | 10 one-hot label columns]`, and the ten label columns carry much
larger weight norms than the pixel columns. Weights are ternarised with a
straight-through estimator, and the two runs differ in **one** place — where the
scale is computed:

| `--group` | scale for the first layer | groups |
| --- | --- | --- |
| `row` | one scale per output row, over the whole 794-column input | 1 |
| `aligned` | one scale per output row per input group: 784 pixel columns as one group, each label column its own | 11 |

Everything else — data order, seed, learning rate, epochs, widths — is identical,
so the difference between the two runs is attributable to the grouping. That is
the whole claim; there is no tuning on either side.

`run.py` writes a manifest with the settings, the sampled data's checksums, the
accuracy, the per-class counts, the per-layer losses, and how many quantised
weights landed exactly on zero. The manifest is the artefact: the CSV of a paper
table is not evidence, a record with checksums is.

## What the CI smoke does and does not prove

The smoke job runs the two commands above and then `assert_direction.py`, which
checks that settings match, that both runs beat chance, and that `aligned` is not
worse than `row` beyond 1 pp.

It does **not** check the magnitude of the gap, and it does not claim these
numbers equal the published ones. A 4,000-image, five-epoch-per-layer run is a
smoke test; the table in `docs/forward-forward-ternary.md` comes from the full
training set with three seeds. Keeping those two statements apart is the point.

## Files

| file | what it is |
| --- | --- |
| `data.py` | reads MNIST idx files (plain or `.gz`), deterministic subsample |
| `quant.py` | ternary quantisation with row-wise or input-aligned scale groups |
| `ff.py` | the two-layer Forward-Forward net, layer-wise training, goodness classifier |
| `run.py` | one configuration → one manifest |
| `assert_direction.py` | compares two manifests; exits non-zero if the direction is lost |
