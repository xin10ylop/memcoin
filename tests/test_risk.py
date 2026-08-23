"""Tests for portfolio limits and circuit breakers."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from alpha.risk.portfolio import Portfolio, PortfolioConfig, Position, RiskState


def make_position(pool="P", entry=1.0, tokens=100.0, cost=100.0, minutes_ago=0.0, **kw):
    return Position(
        pool=pool, mint="M", symbol="SYM", entry_price=entry, tokens=tokens, cost_usd=cost,
        opened_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago), **kw
    )


def test_stop_loss_triggers_below_threshold():
    pos = make_position(stop_loss=0.45)
    assert pos.exit_signal(0.50) == "stop_loss"
    assert pos.exit_signal(0.95) is None


def test_take_profit_triggers_above_threshold():
    pos = make_position(take_profit=1.5)
    assert pos.exit_signal(2.6) == "take_profit"


def test_stop_is_evaluated_before_target():
    """A price that satisfies both must resolve to the conservative outcome."""
    pos = make_position(take_profit=0.1, stop_loss=0.05)
    pos.mark(2.0)
    assert pos.exit_signal(0.5) == "stop_loss"


def test_trailing_stop_only_arms_after_a_real_gain():
    pos = make_position(trailing_stop=0.35, stop_loss=0.9, take_profit=99.0)
    pos.mark(1.10)                      # only +10%: trailing must stay disarmed
    assert pos.exit_signal(0.80) is None
    pos.mark(2.00)                      # +100%: now armed
    assert pos.exit_signal(1.20) == "trailing_stop"


def test_max_hold_forces_an_exit():
    pos = make_position(minutes_ago=60, max_hold_min=45, take_profit=99, stop_loss=0.99)
    assert pos.exit_signal(1.0) == "max_hold"


def test_cannot_open_the_same_pool_twice():
    p = Portfolio()
    p.open(make_position(pool="A"), 100)
    allowed, why = p.can_open("A", 100)
    assert not allowed and "already holding" in why


def test_max_open_positions_is_enforced():
    p = Portfolio(PortfolioConfig(max_open_positions=2))
    p.open(make_position(pool="A"), 100)
    p.open(make_position(pool="B"), 100)
    allowed, why = p.can_open("C", 100)
    assert not allowed and "max open positions" in why


def test_total_exposure_cap_is_enforced():
    p = Portfolio(PortfolioConfig(starting_equity_usd=1_000, max_total_exposure_pct=0.10))
    p.open(make_position(pool="A", cost=90), 90)
    allowed, why = p.can_open("B", 90)
    assert not allowed and "max exposure" in why


def test_drawdown_breaker_halts_trading():
    p = Portfolio(PortfolioConfig(starting_equity_usd=1_000, drawdown_halt_pct=0.20))
    p.cash_usd = 700           # −30% from peak
    assert p.risk_state() is RiskState.HALTED
    allowed, why = p.can_open("A", 10)
    assert not allowed and "halted" in why


def test_consecutive_losses_throttle_size():
    p = Portfolio(PortfolioConfig(consecutive_losses_throttle=3, throttle_size_multiplier=0.5))
    for i in range(3):
        p.open(make_position(pool=f"P{i}", cost=100), 100)
        p.close(f"P{i}", 50, "stop_loss")
    assert p.risk_state() is RiskState.THROTTLED
    assert p.size_multiplier() == 0.5


def test_a_win_resets_the_loss_streak():
    p = Portfolio()
    p.open(make_position(pool="A", cost=100), 100)
    p.close("A", 10, "stop_loss")
    assert p.consecutive_losses == 1
    p.open(make_position(pool="B", cost=100), 100)
    p.close("B", 300, "take_profit")
    assert p.consecutive_losses == 0


def test_reentry_cooldown_blocks_immediate_repurchase():
    p = Portfolio(PortfolioConfig(reentry_cooldown_min=120))
    p.open(make_position(pool="A", cost=100), 100)
    p.close("A", 50, "stop_loss")
    allowed, why = p.can_open("A", 100)
    assert not allowed and "cooldown" in why


def test_closing_returns_proceeds_to_cash():
    p = Portfolio(PortfolioConfig(starting_equity_usd=1_000))
    p.open(make_position(pool="A", cost=100), 100)
    assert p.cash_usd == pytest.approx(900)
    p.close("A", 250, "take_profit")
    assert p.cash_usd == pytest.approx(1_150)


def test_stats_report_hit_rate_and_profit_factor():
    p = Portfolio()
    p.open(make_position(pool="A", cost=100), 100); p.close("A", 300, "take_profit")
    p.open(make_position(pool="B", cost=100), 100); p.close("B", 50, "stop_loss")
    stats = p.stats()
    assert stats["trades"] == 2
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(200 / 50)
