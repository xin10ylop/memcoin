"""Tests for the signal layer — the system's actual output."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from alpha.signals import ExitPlan, Signal, SignalEmitter, build_exit_plan


def make_signal(**kw):
    defaults = dict(
        mint="MINT", symbol="TKN", pool="POOL", dex="pump-fun",
        score=0.40, survival_probability=0.75, reference_price=1e-6,
        suggested_usd=100.0, liquidity_usd=30_000.0, exit_plan=build_exit_plan(),
    )
    defaults.update(kw)
    return Signal(**defaults)


def test_signal_expires():
    """A stale signal is invalid, not merely worse — the edge itself decays."""
    signal = make_signal(valid_for_seconds=60)
    assert signal.is_valid()
    assert not signal.is_valid(datetime.now(timezone.utc) + timedelta(seconds=61))


def test_seconds_remaining_never_negative():
    signal = make_signal(valid_for_seconds=10)
    later = datetime.now(timezone.utc) + timedelta(seconds=999)
    assert signal.seconds_remaining(later) == 0.0


def test_exit_plan_produces_concrete_prices():
    """A bot needs levels, not percentages."""
    plan = ExitPlan(take_profit_pct=2.0, stop_loss_pct=0.5, trailing_stop_pct=0.3, max_hold_minutes=60)
    levels = plan.describe(entry_price=100.0)
    assert levels["take_profit_price"] == pytest.approx(300.0)
    assert levels["stop_loss_price"] == pytest.approx(50.0)
    assert levels["max_hold_minutes"] == 60


def test_trailing_arms_above_entry():
    plan = build_exit_plan()
    levels = plan.describe(entry_price=100.0)
    assert levels["trailing_arms_above"] > 100.0


def test_default_target_is_wide():
    """Grid search showed expectancy improves monotonically with a wider target."""
    assert build_exit_plan().take_profit_pct >= 1.5


def test_serialisation_includes_exit_levels():
    data = make_signal().to_dict()
    assert "exit_levels" in data
    assert "expires_at" in data
    assert data["exit_levels"]["take_profit_price"] > data["exit_levels"]["stop_loss_price"]


def test_description_mentions_both_sides_of_the_trade():
    text = make_signal().describe()
    assert "BUY" in text and "EXIT" in text
    assert "dump" in text and "liquidity" in text


def test_emitter_writes_and_reads_back(tmp_path):
    emitter = SignalEmitter(tmp_path / "sig.jsonl", echo=False)
    assert emitter.emit(make_signal(mint="A")) is True
    assert emitter.emit(make_signal(mint="B")) is True
    rows = emitter.recent()
    assert len(rows) == 2
    assert rows[0]["mint"] == "B"   # newest first


def test_emitter_suppresses_low_scores(tmp_path):
    emitter = SignalEmitter(tmp_path / "sig.jsonl", echo=False, min_score=0.30)
    assert emitter.emit(make_signal(score=0.10)) is False
    assert emitter.suppressed == 1
    assert emitter.emitted == 0


def test_emitter_can_filter_to_valid_signals(tmp_path):
    emitter = SignalEmitter(tmp_path / "sig.jsonl", echo=False)
    emitter.emit(make_signal(mint="OLD", valid_for_seconds=-10))
    emitter.emit(make_signal(mint="NEW", valid_for_seconds=600))
    valid = emitter.recent(valid_only=True)
    assert [r["mint"] for r in valid] == ["NEW"]


def test_missing_feed_reads_as_empty(tmp_path):
    assert SignalEmitter(tmp_path / "none.jsonl", echo=False).recent() == []
