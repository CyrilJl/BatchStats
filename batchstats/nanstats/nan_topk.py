"""Extreme-value selection ignoring NaNs per position."""

from ..stats.topk import BatchTopK


class BatchNanTopK(BatchTopK):
    """Top-k selection ignoring NaNs independently at every position."""

    _ignore_nan = True
