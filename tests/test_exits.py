"""Tests for path-dependent exit rules."""
from __future__ import annotations

import random

import pytest

from alpha.risk.exits import (
    DumpDetectorConfig,
    ExitMonitor,
    detect_dump,
    detect_liquidity_exit,
    robust_sigma,
)


def volatile_path(n=60, swing=0.08, seed=7):
    rnd = random.Random(seed)
    path = [1.0]
    for _ in range(n):
        path.append(path[-1] * (1 + rnd.uniform(-swing, swing)))
    return path


def test_ordinary_volatility_does_not_trigger():
    assert not detect_dump(volatile_path()).triggered


def test_a_large_dump_triggers():
    path = volatile_path()
    path.append(path[-1] * 0.45)
    signal = detect_dump(path)
    assert signal.triggered
    assert signal.z_score < -4


def test_a_moderate_dip_does_not_trigger():
    path = volatile_path()
    path.append(path[-1] * 0.90)
    assert not detect_dump(path).triggered


def test_quiet_token_is_protected_by_the_sigma_floor():
    """Without a floor, a near-zero MAD makes every tick look like a shock."""
    quiet = [1.0 + 0.0001 * i for i in range(60)] + [0.97]
    assert not detect_dump(quiet).triggered


def test_upward_shocks_never_trigger():
    path = volatile_path()
    path.append(path[-1] * 4.0)
    assert not detect_dump(path).triggered


def test_short_history_does_not_trigger():
    assert not detect_dump([1.0, 0.2]).triggered


def test_mad_is_far_more_outlier_resistant_than_standard_deviation():
    """The reason MAD is used here rather than a sigma-based threshold.

    A memecoin's return series is dominated by the very shocks we are trying to
    detect, so a standard-deviation threshold widens exactly when it needs to
    stay tight. The test compares how much each estimator moves when a single
    extreme value is added.
    """
    import statistics

    clean = [0.01, -0.01, 0.02, -0.02] * 10
    contaminated = clean + [5.0]

    mad_growth = robust_sigma(contaminated) / robust_sigma(clean)
    std_growth = statistics.pstdev(contaminated) / statistics.pstdev(clean)

    # One outlier inflates the standard deviation by well over an order of
    # magnitude; MAD barely moves.
    assert std_growth > 20
    assert mad_growth < 2
    assert mad_growth < std_growth / 10


def test_liquidity_withdrawal_is_detected_from_the_peak():
    assert detect_liquidity_exit([50_000, 52_000, 48_000, 25_000]).triggered


def test_stable_liquidity_does_not_trigger():
    assert not detect_liquidity_exit([50_000, 52_000, 48_000, 51_000]).triggered


def test_monitor_prefers_liquidity_over_price():
    """A draining pool is more urgent than a price move."""
    monitor = ExitMonitor()
    for price in volatile_path():
        monitor.update(price, 40_000)
    monitor.update(0.4, 10_000)
    assert "liquidity" in monitor.check().reason


def test_monitor_bounds_its_history():
    monitor = ExitMonitor(max_history=50)
    for i in range(500):
        monitor.update(1.0 + i * 1e-6, 1000)
    assert len(monitor.prices) <= 50
