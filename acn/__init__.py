"""Adaptive Consensus Networks — predictive-coding consensus, EP-trained.

The design (one sentence): columns each predict their latent from their local
patch, neighbors predict each other (consensus via prediction error), and a
shared per-column decoder reads each column's latent into a digit prediction —
the label is broadcast to all columns and the global verdict is the average of
the per-column predictions. All settle to minimize one prediction-error energy.
Trained with centered EP (free / nudged+ / nudged- settles, one contrast
backward). No BPTT.

This is the EP-native form of "agents agreeing on an answer": the settled state
is a minimum of one scalar energy (a sum of squared prediction errors), not a
saddle of an augmented Lagrangian. EP is mathematically native to it. The
per-column readout mirrors Sheaf-ADMM's per-agent classification head (decode
each agent, then average) so the per-column consensus visualization is the
trained, in-distribution readout.
"""

__version__ = "0.0.1"
