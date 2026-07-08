# ACN — Adaptive Consensus Networks

A Thousand-Brains-style sparse-column architecture: an image is split into
overlapping patches, each patch gets its own mini network (a "column"), and a
**sparse subset of columns activates per input**. The active columns run an
ADMM consensus loop, converge to one shared answer, and vote.

## What is ACN?

Instead of one big network swallowing the whole image, **ACN chops the image into
overlapping multi-scale patches and gives each patch its own tiny network** (a
"mini network" or "column"). No single column sees enough to classify the image
alone — they have to **talk**.

The talking is a **unified hierarchy loop**: the bottom layer (74 patch-columns)
and the top layer (8 abstract columns) step together for 20 rounds. Each round:
1. Bottom columns do one ADMM consensus step (all-pairs — active columns talk to
   each other wherever they are).
2. The top layer reads the bottom's current belief and does its own consensus step.
3. A **predictive-coding exchange** sends predictions down and errors up — both
   layers adjust each other (biologically plausible, like cortical feedback loops).

The **verdict** is a **cooperative confidence-weighted vote**: both layers
contribute, weighted by how confident each is on that input. A clean digit → the
top's gestalt leads. A noisy tie → the bottom's per-patch detail leads.

## Quick start

```bash
pip install -e .
pytest -q
python train.py mnist                      # full training (50 epochs) + GIFs
python train.py mnist train.epochs=3       # quick tweak via dotted overrides
```

Training writes checkpoints + metrics to `results/runs/acn_result/`.

## Visualizing

```bash
python scripts/viz_spotlight.py            # generates spotlight.gif + robustness_spotlight.gif
```

The GIFs show a **2×5 decision confidence grid**: each cell is a digit (0-9),
blank if the model's confidence is < 50%, colored with brightness = confidence.
Watch the decision form as the consensus rounds animate.

## Package layout

```
acn/
  config.py          # dataclasses + presets (mnist, poc)
  decomposition.py   # overlapping multi-scale patch extraction
  topology.py        # all-pairs edges + conductance init
  consensus.py       # ADMM primal/consensus/dual + motor + hierarchy loop
  networks.py        # encoder (+positional), decoder, motor, predictive maps, gate
  model.py           # AdaptiveConsensusNetwork nn.Module
  train.py           # BPTT training loop, loss, logging
  inspect.py         # serialize state + summary stats
  visualize.py       # 2×5 confidence grid GIFs
tests/
  test_smoke.py      # forward, backprop, shapes, presets
scripts/
  viz_spotlight.py        # generate the confidence GIFs from a checkpoint
  eval_robustness.py      # evaluate under corruptions
```

## Architecture (145,714 parameters)

- **Bottom layer**: 74 multi-scale columns (4×4, 6×6, 8×8 patches), 8 active per
  input (hard top-k gate), latent dim 32, all-pairs topology.
- **Top layer**: 8 abstract columns, 4 active per input, latent dim 16, all-pairs.
- **Hierarchy**: 20 rounds, BPTT with detach_after=8, predictive-coding exchange
  (decode_down + encode_up, separate learned maps).
- **Readout**: cooperative confidence-weighted vote (logit-margin weights).
- **Motor system**: 2D path integration + efference copy.
- **Warm start**: feedforward sweep (each column's own primal, no primer/broadcast).

## Results

- **MNIST test accuracy: ~95%** with 145K params (LeNet-scale, sparse-column
  consensus hierarchy instead of conv stack).
