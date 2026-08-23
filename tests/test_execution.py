"""Tests for AMM cost modelling and position sizing."""
from __future__ import annotations

import math

import pytest

from alpha.execution.costs import CostModel, FillSide
from alpha.risk.sizing import PositionSizer, SizingConfig, kelly_fraction, optimal_order_usd


def test_price_impact_grows_with_order_size():
    m = CostModel()
    small = m.price_impact(100, 100_000, 0.0025)
    large = m.price_impact(5_000, 100_000, 0.0025)
    assert large > small > 0


def test_price_impact_shrinks_with_pool_depth():
    m = CostModel()
    assert m.price_impact(500, 10_000, 0.0025) > m.price_impact(500, 1_000_000, 0.0025)


def test_price_impact_matches_the_constant_product_formula():
    """impact = 1/(1-f) * (1 + A(1-f)/(L/2)) - 1"""
    m = CostModel()
    amount, liquidity, fee = 500.0, 60_000.0, 0.0025
    expected = (1 / (1 - fee)) * (1 + amount * (1 - fee) / (liquidity / 2)) - 1
    assert m.price_impact(amount, liquidity, fee) == pytest.approx(expected, rel=1e-9)


def test_orders_exceeding_the_depth_limit_are_rejected():
    m = CostModel(max_pool_fraction=0.02)
    fill = m.simulate(FillSide.BUY, 5_000, 1.0, 100_000, dex="pumpswap")
    assert fill.rejected and "pool depth" in fill.reason


def test_zero_liquidity_is_rejected():
    assert CostModel().simulate(FillSide.BUY, 100, 1.0, 0.0).rejected


def test_buys_fill_above_spot_and_sells_below():
    m = CostModel()
    buy = m.simulate(FillSide.BUY, 200, 1.0, 200_000, dex="pumpswap")
    sell = m.simulate(FillSide.SELL, 200, 1.0, 200_000, dex="pumpswap")
    assert buy.effective_price > 1.0
    assert sell.effective_price < 1.0


def test_round_trip_cost_is_u_shaped_in_size():
    """Fixed fees dominate small orders; price impact dominates large ones."""
    m = CostModel()
    liquidity = 200_000.0
    tiny = m.round_trip_cost_pct(5, liquidity, "pumpswap")
    middle = m.round_trip_cost_pct(optimal_order_usd(liquidity, m.network_fee_usd(first_buy=True)),
                                   liquidity, "pumpswap")
    big = m.round_trip_cost_pct(3_500, liquidity, "pumpswap")
    assert middle < tiny
    assert middle < big


def test_optimal_order_size_matches_the_closed_form():
    fee, liquidity = 0.64, 100_000.0
    assert optimal_order_usd(liquidity, fee) == pytest.approx(math.sqrt(fee * liquidity / 2))


def test_optimal_order_scales_with_sqrt_of_liquidity():
    fee = 0.64
    assert optimal_order_usd(400_000, fee) == pytest.approx(2 * optimal_order_usd(100_000, fee), rel=1e-9)


def test_venue_fees_differ_by_dex():
    m = CostModel()
    assert m.venue_fee("pump-fun") > m.venue_fee("pumpswap")
    assert m.venue_fee("unknown-dex") == m.venue_fee("_default")


@pytest.mark.parametrize(
    "p,b,expected_sign",
    [(0.6, 1.0, 1), (0.4, 1.0, -1), (0.5, 1.0, 0)],
)
def test_kelly_sign_follows_edge(p, b, expected_sign):
    f = kelly_fraction(p, b, 1.0)
    if expected_sign > 0:
        assert f > 0
    else:
        assert f == 0.0  # no edge -> no bet


def test_kelly_rejects_degenerate_probabilities():
    assert kelly_fraction(0.0, 1.5) == 0.0
    assert kelly_fraction(1.0, 1.5) == 0.0


def test_sizing_rejects_when_there_is_no_edge():
    sizer = PositionSizer()
    assert not sizer.size(equity_usd=10_000, win_prob=0.10, liquidity_usd=50_000).approved


def test_sizing_is_capped_by_equity_share_not_by_confidence():
    """Even a near-certain signal may not exceed the per-position cap."""
    sizer = PositionSizer(SizingConfig(max_position_pct=0.02))
    decision = sizer.size(equity_usd=10_000, win_prob=0.99, liquidity_usd=5_000_000)
    assert decision.approved
    assert decision.usd <= 10_000 * 0.02 + 1e-6


def test_sizing_is_capped_by_pool_depth():
    sizer = PositionSizer(SizingConfig(max_pool_fraction=0.01))
    decision = sizer.size(equity_usd=1_000_000, win_prob=0.6, liquidity_usd=20_000)
    assert decision.usd <= 20_000 * 0.01 + 1e-6


def test_breakeven_win_rate_rises_with_costs():
    sizer = PositionSizer()
    assert sizer.breakeven_win_prob(0.10) > sizer.breakeven_win_prob(0.0)


def test_breakeven_matches_the_analytic_expression():
    cfg = SizingConfig(take_profit=1.5, stop_loss=0.45)
    sizer = PositionSizer(cfg)
    cost = 0.06
    expected = (cfg.stop_loss + cost) / (cfg.take_profit + cfg.stop_loss)
    assert sizer.breakeven_win_prob(cost) == pytest.approx(expected)
