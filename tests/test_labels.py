"""Tests for triple-barrier labelling."""
from __future__ import annotations

import pytest

from alpha.features.label import Barrier, LabelConfig, label_trade, summarise
from tests.conftest import FakeCandle

BASE = 1_700_000_000


def rising(steps=10, rate=0.3):
    return [FakeCandle(BASE + 60 * i, 1 + rate * (i - 1), 1 + rate * i, 1 + rate * (i - 1) * 0.99, 1 + rate * i)
            for i in range(1, steps + 1)]


def falling(steps=10, rate=0.1):
    return [FakeCandle(BASE + 60 * i, 1 - rate * (i - 1), 1 - rate * (i - 1), 1 - rate * i, 1 - rate * i)
            for i in range(1, steps + 1)]


def test_take_profit_barrier():
    label = label_trade("P", rising(), BASE)
    assert label.barrier is Barrier.TAKE_PROFIT
    assert label.is_win
    assert label.net_return > 1.0


def test_stop_loss_barrier():
    label = label_trade("P", falling(), BASE)
    assert label.barrier is Barrier.STOP_LOSS
    assert not label.is_win
    assert label.net_return < -0.4


def test_time_barrier_when_price_goes_nowhere():
    flat = [FakeCandle(BASE + 60 * i, 1.0, 1.02, 0.99, 1.0) for i in range(1, 80)]
    label = label_trade("P", flat, BASE, LabelConfig(horizon_min=30))
    assert label.barrier is Barrier.TIME
    assert label.minutes_held == pytest.approx(30, abs=1)


def test_absence_of_future_data_is_a_total_loss_not_a_dropped_row():
    """The core anti-survivorship-bias behaviour."""
    label = label_trade("P", [], BASE)
    assert label.barrier is Barrier.NO_DATA
    assert label.net_return == -1.0
    assert not label.is_win


def test_candles_only_before_decision_also_count_as_no_data():
    past_only = [FakeCandle(BASE - 60 * i, 1, 1, 1, 1) for i in range(1, 5)]
    assert label_trade("P", past_only, BASE).barrier is Barrier.NO_DATA


def test_single_candle_spanning_both_barriers_resolves_pessimistically():
    """We cannot know which barrier was touched first, so assume the adverse one."""
    both = [FakeCandle(BASE + 60, 1.0, 5.0, 0.1, 0.2)]
    assert label_trade("P", both, BASE).barrier is Barrier.STOP_LOSS


def test_entry_uses_the_first_candle_after_the_decision():
    candles = [
        FakeCandle(BASE - 60, 9.0, 9.0, 9.0, 9.0),   # before: must be ignored
        FakeCandle(BASE + 60, 2.0, 2.1, 1.9, 2.0),   # entry here
    ]
    assert label_trade("P", candles, BASE).entry_price == pytest.approx(2.0)


def test_costs_reduce_the_realised_return():
    cfg_free = LabelConfig(round_trip_cost=0.0)
    cfg_costly = LabelConfig(round_trip_cost=0.10)
    assert (label_trade("P", rising(), BASE, cfg_free).net_return
            > label_trade("P", rising(), BASE, cfg_costly).net_return)


def test_summarise_reports_coherent_aggregates():
    labels = [label_trade("P", rising(), BASE) for _ in range(3)]
    labels += [label_trade("P", falling(), BASE) for _ in range(7)]
    summary = summarise(labels)
    assert summary["n"] == 10
    assert summary["hit_rate"] == pytest.approx(0.3)
    assert summary["payoff_ratio"] > 1


def test_summarise_handles_empty_input():
    assert summarise([])["n"] == 0
