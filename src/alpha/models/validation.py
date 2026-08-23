"""Validation machinery for strategies fitted to overlapping financial data.

Standard k-fold cross-validation reports badly inflated scores on this problem
for two compounding reasons:

**Group leakage.** One token contributes several decision rows that share a
single price path. Split at random and near-duplicate rows land on both sides of
the fold, so the model is graded partly on tokens it has already memorised.
Splitting by *pool* fixes this.

**Temporal leakage.** A label spans the horizon after its decision time. A
training row whose horizon overlaps a test row's decision time encodes
information from the test period. Lopez de Prado's remedy is *purging* —
dropping training rows whose label window overlaps the test window — and
*embargoing* a further gap after it, since market state is autocorrelated across
the boundary.

This module also implements two honesty checks. :func:`deflated_sharpe` adjusts
an observed Sharpe ratio for the number of configurations tried, because the
best of many random strategies looks good by construction.
:func:`probability_of_backtest_overfitting` estimates how often the
in-sample-best configuration underperforms the median out of sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np


@dataclass
class PurgedGroupTimeSplit:
    """Walk-forward splits that are grouped by entity and purged in time.

    Each fold trains on everything before a test window (minus a purge and
    embargo band) and tests on that window. Groups never straddle a boundary.
    """

    n_splits: int = 5
    # Label horizon in the same units as ``times``. Training rows whose label
    # window reaches into the test window are purged.
    purge: float = 45 * 60
    # Extra gap after the test window before training resumes.
    embargo: float = 15 * 60

    def split(
        self, times: Sequence[float], groups: Sequence[str]
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        times_arr = np.asarray(times, dtype=float)
        groups_arr = np.asarray(groups)
        n = len(times_arr)
        if n == 0:
            return
        order = np.argsort(times_arr, kind="stable")
        sorted_times = times_arr[order]

        # Test windows are contiguous in time, covering the latter portion of
        # the sample so that every fold trains only on the past.
        start = int(n * 0.4)
        if start >= n - 1:
            start = max(0, n - self.n_splits - 1)
        bounds = np.linspace(start, n, self.n_splits + 1).astype(int)

        for i in range(self.n_splits):
            lo, hi = bounds[i], bounds[i + 1]
            if hi - lo < 1:
                continue
            test_idx = order[lo:hi]
            t_start, t_end = sorted_times[lo], sorted_times[hi - 1]

            # A group that appears in test may not appear in train at all.
            test_groups = set(groups_arr[test_idx].tolist())

            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            # Purge: drop training rows whose label window overlaps the test
            # window, and rows inside the embargo band after it.
            overlaps = (times_arr + self.purge >= t_start) & (times_arr <= t_end + self.embargo)
            train_mask &= ~overlaps
            # Group isolation: no token may appear on both sides.
            train_mask &= ~np.isin(groups_arr, list(test_groups))

            train_idx = np.flatnonzero(train_mask)
            if len(train_idx) and len(test_idx):
                yield train_idx, test_idx


def deflated_sharpe(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    variance_of_trial_sharpes: float | None = None,
) -> float:
    """Probability that the true Sharpe ratio exceeds zero, after deflation.

    Testing many configurations guarantees that the best one looks good. The
    deflated Sharpe ratio compares the observed value against the expected
    maximum of ``n_trials`` draws from a null distribution, then converts the
    excess into a probability. A value below ~0.95 means the result is not
    distinguishable from selection noise.
    """
    if n_observations < 2 or n_trials < 1:
        return 0.0
    from scipy.stats import norm  # local import keeps scipy optional at import time

    var = variance_of_trial_sharpes if variance_of_trial_sharpes is not None else 1.0
    sigma = math.sqrt(max(var, 1e-12))
    # Expected maximum of n_trials standard normals (Bailey & Lopez de Prado).
    euler = 0.5772156649015329
    if n_trials > 1:
        z1 = norm.ppf(1.0 - 1.0 / n_trials)
        z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = (1.0 - euler) * z1 + euler * z2
    else:
        expected_max = 0.0
    threshold = sigma * expected_max

    numerator = (observed_sharpe - threshold) * math.sqrt(n_observations - 1)
    denominator = math.sqrt(
        max(1e-12, 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
    )
    return float(norm.cdf(numerator / denominator))


def probability_of_backtest_overfitting(
    performance: np.ndarray, n_partitions: int = 8
) -> float:
    """Combinatorially-symmetric estimate of overfitting probability.

    ``performance`` is a (observations × configurations) matrix of per-period
    performance. The sample is cut into ``n_partitions`` blocks; for every way of
    splitting blocks into equal in-sample and out-of-sample halves, we pick the
    configuration that won in-sample and record its out-of-sample rank. PBO is
    the fraction of splits where that winner lands in the bottom half — i.e. how
    often the selection procedure itself is the source of the result.
    """
    perf = np.asarray(performance, dtype=float)
    if perf.ndim != 2 or perf.shape[1] < 2:
        return 0.0
    n_obs, n_cfg = perf.shape
    n_partitions = min(n_partitions, n_obs)
    if n_partitions < 2:
        return 0.0
    blocks = np.array_split(np.arange(n_obs), n_partitions)
    half = n_partitions // 2
    if half < 1:
        return 0.0

    logits: list[float] = []
    for combo in combinations(range(n_partitions), half):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_partitions) if i not in combo])
        if not len(is_idx) or not len(oos_idx):
            continue
        is_perf = perf[is_idx].mean(axis=0)
        oos_perf = perf[oos_idx].mean(axis=0)
        best = int(np.argmax(is_perf))
        # Relative rank of the in-sample winner, out of sample.
        rank = float((oos_perf < oos_perf[best]).sum()) / n_cfg
        rank = min(max(rank, 1.0 / (n_cfg + 1)), 1.0 - 1.0 / (n_cfg + 1))
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        return 0.0
    return float(np.mean([1.0 if x <= 0 else 0.0 for x in logits]))
