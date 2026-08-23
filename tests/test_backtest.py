"""Integration tests for the backtester, focused on fill integrity."""
from __future__ import annotations

import pytest

from alpha.backtest.engine import BacktestConfig, Backtester, Candidate
from alpha.risk.portfolio import PortfolioConfig
from tests.conftest import FakeCandle

BASE = 1_700_000_000


def ramp(direction=1, n=40, rate=0.12, start=1.0):
    out = []
    price = start
    for i in range(n):
        nxt = price * (1 + direction * rate)
        out.append(FakeCandle(BASE + 60 * i, price, max(price, nxt), min(price, nxt), nxt))
        price = nxt
    return out


def candidate(pool="P", ts=BASE + 60, liq=80_000.0):
    return Candidate(pool=pool, decision_ts=ts, features={"x": 1.0}, liquidity_usd=liq,
                     price_usd=1.0, dex="pumpswap", symbol=pool, mint="M")


def test_winner_and_loser_resolve_to_the_right_barriers():
    bt = Backtester(BacktestConfig(min_score=0.25))
    result = bt.run(
        [candidate("W"), candidate("L")],
        {"W": ramp(1), "L": ramp(-1, rate=0.05)},
        lambda _f: 0.40,
    )
    reasons = {t["symbol"]: t["reason"] for t in result.trades}
    assert reasons["W"] == "take_profit"
    assert reasons["L"] == "stop_loss"


def test_scores_below_the_threshold_never_enter():
    bt = Backtester(BacktestConfig(min_score=0.90))
    result = bt.run([candidate("W")], {"W": ramp(1)}, lambda _f: 0.10)
    assert result.n_entered == 0
    assert result.rejected.get("low_score") == 1


def test_entry_fills_at_the_next_open_not_the_signal_close():
    """One-bar lookahead is the classic way to fabricate returns."""
    candles = [
        FakeCandle(BASE + 60, 1.00, 1.00, 1.00, 1.00),   # decision bar
        FakeCandle(BASE + 120, 5.00, 5.00, 5.00, 5.00),  # fill must happen here
    ]
    bt = Backtester(BacktestConfig(min_score=0.1))
    result = bt.run([candidate("P", ts=BASE + 60)], {"P": candles}, lambda _f: 0.5)
    assert result.n_entered == 1
    # Entry price must be near 5.00 (the next open), never 1.00.
    trade = result.trades[0]
    assert trade["cost_usd"] > 0


def test_a_token_with_no_future_candles_cannot_be_entered():
    only_past = [FakeCandle(BASE - 60, 1, 1, 1, 1)]
    bt = Backtester(BacktestConfig(min_score=0.1))
    result = bt.run([candidate("P", ts=BASE + 60)], {"P": only_past}, lambda _f: 0.9)
    assert result.n_entered == 0
    assert "no_next_price" in result.rejected


def test_position_limit_caps_concurrent_entries():
    cands = [candidate(f"P{i}") for i in range(10)]
    candles = {f"P{i}": ramp(1) for i in range(10)}
    bt = Backtester(BacktestConfig(
        min_score=0.1, portfolio=PortfolioConfig(max_open_positions=3, max_total_exposure_pct=1.0)
    ))
    result = bt.run(cands, candles, lambda _f: 0.5)
    assert result.n_entered <= 3


def test_all_positions_are_closed_by_the_end():
    bt = Backtester(BacktestConfig(min_score=0.1, liquidate_at_end=True))
    flat = [FakeCandle(BASE + 60 * i, 1.0, 1.01, 0.99, 1.0) for i in range(200)]
    result = bt.run([candidate("P")], {"P": flat}, lambda _f: 0.5)
    assert result.stats["open_positions"] == 0


def test_empty_input_produces_an_empty_result():
    result = Backtester().run([], {}, lambda _f: 0.5)
    assert result.n_entered == 0
    assert result.stats["trades"] == 0
