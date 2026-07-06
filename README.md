# ACN — Adaptive Consensus Networks

A Thousand-Brains-style sparse-column architecture: an image is split into
overlapping patches, each patch gets its own mini network (a "column"), and a
**sparse subset of columns activates per input**. The active columns run an ADMM
consensus loop, converge to one shared answer, and vote. The rest stay silent.

## What is ACN?

Instead of one big network swallowing the whole image, **ACN chops the image into
overlapping patches and gives each patch its own tiny network** (a "mini network"
or "column"). No single column sees enough to classify the image alone — they
have to **talk**.

The talking is an ADMM consensus loop: each active column proposes a local answer
(`x`), negotiates a shared agreement with neighbors (`z`), and remembers how much
it had to compromise (`u`). After K rounds, the active columns' answers line up
and are fused into one prediction (only active columns vote — the
Thousand-Brains sparse vote).

The adaptive part is **which columns fire** — a **learned gate**. The encoder
outputs a per-column relevance logit `s_i`; the gate is `sigmoid(s_i)`. The
active count is **discovered by training** under a gentle sparsity penalty —
a "1" recruits few columns, an "8" recruits many. Brain-like ~10% active.

The relevance head is bias-init high so gates start near 1 (all columns active);
the sparsity penalty then prunes the useless columns DOWN over training, instead
of starting at ~0.5 and sliding to 0 (the gate-collapse failure mode fixed in
LOG_2026-07-05).

A secondary Physarum-style conductance dynamics on the wires (`D_ij`) grows links
where neighbors keep disagreeing and prunes them where they agree, masked by
column activity so silent columns carry no flow.

Each column and every link exposes inspectable state (`x`, `z`, `u`, `D`,
disagreement flux `Q`, the gate `active`), so you can watch *which columns fired,
how hard they coordinated, and which wires survived* — per input, per digit class.

**In one sentence:** many small patch-networks negotiate a shared answer through
ADMM, a sparse subset activates per input, and they vote — so the model is a
diverse ensemble that softly agrees, robust under distribution shift.

## Quick start

```bash
pip install -e .
pytest -q
python train.py                       # full training (50 epochs) + consensus GIF
python train.py train.epochs=3        # quick tweak via dotted overrides
```

Training writes checkpoints + a consensus-agreement GIF to `results/runs/model_result/`.

The GIF shows, per sample: the input digit | the **Mini networks** grid (all
active cells colored by the fused global decision; inactive cells black) | the
**Mini networks decision** bar chart (the soft fused scorecard with a +/- axis,
winner highlighted) | a digit color legend. The per-row label shows only the
ground-truth digit — compare the winning color to the legend to judge
correct/wrong.

## Inspecting one sample

```bash
python scripts/inspect_one_sample.py   # dump raw per-column scorecards for sample 6741
```

Loads the checkpoint, runs one forward pass, and prints the per-column raw
logits, the gate, the fused scorecard, and the per-column argmax colors — so you
can see exactly how the fused argmax wins. Edit the script to point at other
samples.

## Package layout

```
acn/
  config.py          # dataclasses + presets (poc test config, model_result)
  decomposition.py   # overlapping patch extraction (batched)
  topology.py        # spatial-neighbor edges + conductance init
  consensus.py       # primal/consensus/dual + Physarum flux conductance (numpy ref + torch)
  networks.py        # encoder (+relevance head), decoder, restriction maps, sparse_fuse, column_gate
  model.py           # AdaptiveConsensusNetwork nn.Module (gate -> consensus -> sparse fuse)
  train.py           # training loop, loss (CE + local + wire-sparse + gate-sparse), logging
  inspect.py         # serialize x,z,u,D,Q,active + summary stats
  visualize.py       # consensus GIF (grid + fused-decision bars)
tests/               # parity, gradcheck, shapes, math unit tests
scripts/
  viz_consensus.py        # generate the consensus GIF from a checkpoint
  inspect_one_sample.py   # dump raw per-column scorecards for one sample
```

See `BLUEPRINT.md` for the full architecture and `LOG_2026-07-05.md` for the
sparse-column migration and the learned-gate collapse/fix.
