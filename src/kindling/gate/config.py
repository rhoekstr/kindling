"""Configuration for the gating network."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GatingConfig:
    """Gating network training + inference knobs.

    Attributes
    ----------
    enabled:
        Master switch. Off by default; flip to True to train the gate
        at fit time and use its weights instead of the Bayesian blend's
        posterior mean at recommend time.
    hidden_dim:
        Hidden layer size. 16 is plenty for the 8-ish context features
        and 11 signals we currently have.
    n_epochs:
        Training epochs.
    batch_size:
        Mini-batch size for BPR SGD.
    negatives_per_positive:
        Number of uniformly-sampled negatives per positive pair.
    learning_rate:
        Plain SGD learning rate. Gate is small so no need for Adam.
    weight_decay:
        L2 regularization on gate weights.
    seed:
        RNG seed for reproducibility.
    min_users:
        Skip gate training when the dataset has fewer entities than this.
    min_positives:
        Skip gate training when too few positive pairs to sample from.
    """

    enabled: bool = False
    hidden_dim: int = 16
    n_epochs: int = 20
    batch_size: int = 512
    negatives_per_positive: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    seed: int = 0
    min_users: int = 100
    min_positives: int = 500
