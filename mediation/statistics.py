"""Summaries over prompt pairs: confidence intervals, standard errors, ranks, rank agreement."""

from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr


class MeanWithInterval(NamedTuple):
    """Mean over prompt pairs and the half-width of its 95% confidence interval."""

    mean: np.ndarray
    half_width: np.ndarray


def mean_with_confidence_interval(
    values: np.ndarray, axis: int = 0
) -> MeanWithInterval:
    """Mean over prompt pairs with a normal-approximation 95% interval.

    The half-width is ``1.96 * SD / sqrt(N)`` with the sample standard deviation
    (``ddof=1``) over the ``N`` prompt pairs along ``axis``.
    """
    count = values.shape[axis]
    half_width = 1.96 * values.std(axis=axis, ddof=1) / np.sqrt(count)
    return MeanWithInterval(values.mean(axis=axis), half_width)


def standard_error(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Standard error of the mean, ``SD / sqrt(N)`` with the population standard deviation
    (``ddof=0``), as the shaded bands of Figure 2 use."""
    return values.std(axis=axis, ddof=0) / np.sqrt(values.shape[axis])


def signed_ranks(values: np.ndarray) -> np.ndarray:
    """Rank of every entry among all entries, 1 for the largest signed value.

    Ranks are over the signed values, not their magnitudes, so a large negative value ranks
    last. Equal values are ranked in order of appearance.

    :return: integer array of the shape of ``values``.
    """
    ranks = np.empty(values.size, dtype=int)
    ranks[np.argsort(-values.ravel(), kind="stable")] = np.arange(1, values.size + 1)
    return ranks.reshape(values.shape)


def rank_agreement(nie: np.ndarray, pie: np.ndarray) -> float:
    """Rank agreement rho: Spearman correlation between the heads' mean NIE and mean PIE.

    :param nie: per-pair NIE of every head, shape (pairs, layers, heads).
    :param pie: per-pair PIE of every head, same shape.
    :return: Spearman correlation of the two (layers x heads) maps of per-head means over
        pairs, taken over all heads with average ranks for ties.
    """
    return float(
        spearmanr(nie.mean(axis=0).ravel(), pie.mean(axis=0).ravel()).statistic
    )
