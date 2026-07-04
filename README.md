# ACN — Adaptive Consensus Networks

A collection of mini networks where each sees only a local patch of an image, and
communication links between them grow and prune dynamically based on how much they
need to coordinate to produce the correct global prediction.

## What is ACN?

Instead of one big network swallowing the whole image, **ACN chops the image into
overlapping patches and gives each patch its own tiny network** (a "mini network"
or "agent"). No single mini network sees enough to classify the image alone — they
have to **talk**.

The talking is an ADMM consensus loop: each mini network proposes a local answer
(`x`), negotiates a shared agreement with neighbors (`z`), and remembers how much
it had to compromise (`u`). After a few rounds of this, all the local answers line
up and are fused into one prediction.

The novel part is the **wires, not the nodes**. Every link between two mini
networks has a *conductance* `D_ij` that grows where the two endpoints keep
disagreeing and prunes where they already agree — a rule inspired by the slime
mold *Physarum* (it thickens tubes that carry flow and abandons empty ones). So
the communication graph is not fixed: **it adapts during the forward pass**,
becoming sparse where coordination isn't needed and dense where it is. This is
the "Adaptive" in ACN.

Each mini network and every link exposes inspectable state (`x`, `z`, `u`, `D`,
disagreement flux `Q`), so you can watch *which patches coordinated, how hard,
and which wires survived* — per input, per digit class.

**In one sentence:** many small patch-networks negotiate a shared answer through
ADMM, and the links they use to negotiate grow and prune themselves like a slime
mold, so the communication graph is learned by the dynamics of coordination
itself.

## Quick start

```bash
pip install -e .
pytest -q
python -m acn.train --config poc      # MNIST resized to 8x8, small-scale POC
python -m acn.train --config mnist   # native 28x28 MNIST -> ~97% + auto consensus GIF
```

Training writes checkpoints + a consensus-agreement GIF to `results/runs/<name>/`.

## Package layout

```
acn/
  config.py          # dataclasses driving every run
  decomposition.py   # overlapping patch extraction (batched)
  topology.py        # spatial-neighbor edges + conductance init
  consensus.py       # primal/consensus/dual + Physarum conductance (numpy ref + torch)
  networks.py        # encoder, decoder, diagonal restriction maps, fusion
  model.py           # AdaptiveConsensusNetwork nn.Module
  train.py           # training loop, loss, logging
  inspect.py         # serialize x,z,u,D,Q + summary stats
  visualize.py       # conductance heatmaps, evolution, consensus GIF
tests/               # parity, gradcheck, shapes, math unit tests
scripts/
  viz_consensus.py   # generate the consensus-agreement GIF from a checkpoint
```

See `PLAN.md` and `UPGRADED-PLAN.md` for the build plan and the path to a
paper-worthy contribution.
