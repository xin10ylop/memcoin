"""Tests for point-in-time feature construction.

The lookahead tests are the most important in the suite: a feature that can see
the future produces a backtest that cannot be trusted, and the failure is silent.
"""
from __future__ import annotations

import pytest

from alpha.features.build import FEATURE_NAMES, build_features
from tests.conftest import make_snapshot


def test_feature_names_are_stable_and_sorted():
    assert FEATURE_NAMES == sorted(FEATURE_NAMES)
    assert len(FEATURE_NAMES) > 40
    assert "log_liquidity" in FEATURE_NAMES
    assert "buy_ratio_m5" in FEATURE_NAMES


def test_features_never_see_beyond_the_decision_index():
    """The defining invariant: appending future data must not change the past."""
    history = [make_snapshot(age=i, price=1e-6, liq=10_000) for i in range(1, 6)]
    baseline = build_features(history, index=2)

    # A wildly different future must leave the index-2 features untouched.
    extended = history + [
        make_snapshot(age=99, price=1.0, liq=10_000_000, buys=9999, sells=0)
    ]
    after = build_features(extended, index=2)
    assert baseline.values == after.values


def test_features_do_change_when_the_past_changes():
    """Guards against the invariance test passing vacuously."""
    a = [make_snapshot(age=i, liq=10_000) for i in range(1, 4)]
    b = [make_snapshot(age=i, liq=10_000 * i) for i in range(1, 4)]
    assert build_features(a).values != build_features(b).values


def test_negative_index_resolves_to_the_last_snapshot():
    history = [make_snapshot(age=i) for i in range(1, 5)]
    assert build_features(history, -1).values == build_features(history, 3).values


def test_all_features_are_finite_on_degenerate_input():
    """Zero liquidity, zero price and zero volume must not produce NaN or inf."""
    degenerate = make_snapshot(price=0.0, liq=0.0, buys=0, sells=0)
    values = build_features([degenerate, degenerate]).values
    assert values, "expected features to be produced"
    for name, value in values.items():
        assert value == value, f"{name} is NaN"
        assert abs(value) != float("inf"), f"{name} is infinite"


def test_buy_ratio_is_neutral_without_trades():
    values = build_features([make_snapshot(buys=0, sells=0)] * 2).values
    assert values["buy_ratio_m5"] == pytest.approx(0.5)


def test_buy_ratio_reflects_direction():
    buyers = build_features([make_snapshot(buys=90, sells=10)] * 2).values
    sellers = build_features([make_snapshot(buys=10, sells=90)] * 2).values
    assert buyers["buy_ratio_m5"] > 0.8
    assert sellers["buy_ratio_m5"] < 0.2


def test_trajectory_tracks_liquidity_drawdown():
    history = [
        make_snapshot(age=1, liq=10_000),
        make_snapshot(age=2, liq=50_000),
        make_snapshot(age=3, liq=25_000),
    ]
    values = build_features(history).values
    # Peak was 50k, now 25k -> 50% drawdown from the observed high.
    assert values["liq_drawdown"] == pytest.approx(0.5, abs=0.01)


def test_empty_history_is_rejected():
    with pytest.raises(ValueError):
        build_features([])


def test_out_of_range_index_is_rejected():
    with pytest.raises(IndexError):
        build_features([make_snapshot()], index=5)
