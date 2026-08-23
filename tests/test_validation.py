"""Tests for the validation machinery that keeps results honest."""
from __future__ import annotations

import numpy as np
import pytest

from alpha.models.validation import (
    PurgedGroupTimeSplit,
    deflated_sharpe,
    probability_of_backtest_overfitting,
)


def test_groups_never_appear_on_both_sides_of_a_fold():
    """Rows from one token share a price path and must not straddle a split."""
    groups = np.repeat([f"p{i}" for i in range(40)], 5)
    times = np.sort(np.random.default_rng(0).uniform(0, 86_400, 200))
    for train, test in PurgedGroupTimeSplit(n_splits=4).split(times, groups):
        assert not set(groups[train]) & set(groups[test])


def test_training_rows_overlapping_the_test_window_are_purged():
    times = np.arange(0, 1000, dtype=float) * 60
    groups = np.array([f"g{i}" for i in range(1000)])
    purge = 45 * 60
    for train, test in PurgedGroupTimeSplit(n_splits=3, purge=purge, embargo=0).split(times, groups):
        test_start, test_end = times[test].min(), times[test].max()
        # Training legitimately spans both sides of the test window, so the
        # condition must hold for each row individually: either its label
        # window closes before the test opens, or it begins after the test ends.
        before = times[train] + purge < test_start
        after = times[train] > test_end
        assert np.all(before | after)


def test_split_yields_nothing_for_empty_input():
    assert list(PurgedGroupTimeSplit().split([], [])) == []


def test_deflated_sharpe_falls_as_more_configurations_are_tried():
    a = deflated_sharpe(1.5, n_trials=1, n_observations=250)
    b = deflated_sharpe(1.5, n_trials=500, n_observations=250)
    assert a > b


def test_deflated_sharpe_is_a_probability():
    for trials in (1, 10, 1000):
        value = deflated_sharpe(2.0, trials, 250)
        assert 0.0 <= value <= 1.0


def test_pbo_is_near_one_half_for_pure_noise():
    noise = np.random.default_rng(3).normal(size=(200, 16))
    assert 0.3 < probability_of_backtest_overfitting(noise) < 0.7


def test_pbo_is_low_when_one_configuration_is_genuinely_better():
    rng = np.random.default_rng(3)
    perf = rng.normal(size=(200, 16))
    perf[:, 5] += 0.5
    assert probability_of_backtest_overfitting(perf) < 0.2


def test_pbo_handles_degenerate_input():
    assert probability_of_backtest_overfitting(np.zeros((10, 1))) == 0.0
