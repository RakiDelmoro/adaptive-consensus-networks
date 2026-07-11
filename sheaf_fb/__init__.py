"""Sheaf-Forward-Backward network with Equilibrium Propagation.

A multi-agent neural network where each agent is a neuron-like unit that sees
only a small part of the input (a 4x4 MNIST patch). Agents coordinate through
**learned sheaf communication channels** (per-edge restriction maps ``F_ij``
that project each agent's private ``d``-dim state onto a shared ``c``-dim
channel with a neighbor) to reach a global answer.

The network settles to an equilibrium via **Forward-Backward splitting** (a
gradient step on the local objective followed by a proximal step toward sheaf
consensus with neighbors) and is trained with **Equilibrium Propagation** — two
settles (free + nudged), the difference of equilibria is the local learning
signal. No backpropagation through time, no unrolling, no storing of
intermediate rounds. Every parameter update is local to an agent / edge.

This is the Forward-Backward + EP counterpart to the sibling ``acn`` package
(Sheaf-ADMM + BPTT): one state variable per agent instead of three, and
~15x less activation memory per training sample.
"""

from .config import ExperimentConfig, get_preset
from .model import SheafFBModel
from .train import train

__all__ = ["ExperimentConfig", "get_preset", "SheafFBModel", "train"]

__version__ = "0.1.0"
