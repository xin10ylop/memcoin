"""Tests for pump.fun bonding-curve mechanics."""
from __future__ import annotations

import pytest

from alpha.data.bondingcurve import (
    GRADUATION_SOL,
    GRADUATION_VSOL,
    MAX_PREGRAD_MULTIPLE,
    TOKEN_DECIMALS,
    TOTAL_SUPPLY,
    breakeven_price_multiple,
    buy_tokens_out,
    estimate_net_sol_from_marketcap,
    graduation_progress_features,
    launch_price_sol,
    sell_sol_out,
    state_from_net_sol,
)


def test_pregraduation_ceiling_is_14_7x():
    """A token that never graduates cannot have risen more than this."""
    assert MAX_PREGRAD_MULTIPLE == pytest.approx(14.696, abs=0.01)


def test_graduation_state_matches_published_constants():
    state = state_from_net_sol(GRADUATION_SOL)
    assert state.virtual_sol == pytest.approx(GRADUATION_VSOL, abs=0.001)
    assert state.virtual_tokens == pytest.approx(279_900_000, rel=0.001)
    assert state.has_graduated


def test_price_at_graduation_is_the_ceiling():
    assert state_from_net_sol(GRADUATION_SOL).multiple_from_launch == pytest.approx(
        MAX_PREGRAD_MULTIPLE, rel=1e-6
    )


def test_constant_product_is_preserved():
    k0 = state_from_net_sol(0).virtual_sol * state_from_net_sol(0).virtual_tokens
    for net in (1, 10, 50, 85):
        state = state_from_net_sol(net)
        assert state.virtual_sol * state.virtual_tokens == pytest.approx(k0, rel=1e-9)


def test_progress_is_monotonic_and_bounded():
    values = [state_from_net_sol(n).progress for n in (0, 10, 40, 85, 200)]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0


def test_price_rises_along_the_curve():
    prices = [state_from_net_sol(n).price_sol for n in (0, 5, 20, 60, 85)]
    assert prices == sorted(prices)


def test_buying_then_selling_immediately_loses_money():
    """Fees and curve slippage make a round trip strictly negative."""
    state = state_from_net_sol(30)
    tokens = buy_tokens_out(state, 1.0)
    assert sell_sol_out(state, tokens) < 1.0


def test_larger_buys_get_worse_average_prices():
    state = state_from_net_sol(30)
    small = buy_tokens_out(state, 0.1) / 0.1
    large = buy_tokens_out(state, 10.0) / 10.0
    assert large < small     # fewer tokens per SOL at size


def test_marketcap_inversion_round_trips():
    for net in (5.0, 40.0, GRADUATION_SOL):
        state = state_from_net_sol(net)
        market_cap = state.price_sol * (TOTAL_SUPPLY / TOKEN_DECIMALS)
        assert estimate_net_sol_from_marketcap(market_cap) == pytest.approx(net, rel=1e-6)


def test_breakeven_multiple_reaches_one_at_graduation():
    assert breakeven_price_multiple(GRADUATION_SOL) == pytest.approx(1.0, abs=1e-6)


def test_trade_count_efficiency_rewards_fewer_larger_trades():
    """The strongest published graduation predictor."""
    concentrated = graduation_progress_features(40, 50)["curve_sol_per_swap"]
    churned = graduation_progress_features(40, 5000)["curve_sol_per_swap"]
    assert concentrated > churned


def test_headroom_falls_to_zero_at_graduation():
    assert graduation_progress_features(GRADUATION_SOL, 100)["curve_headroom"] == pytest.approx(0.0, abs=1e-6)
    assert graduation_progress_features(1, 100)["curve_headroom"] > 0.9


def test_launch_price_is_positive_and_tiny():
    assert 0 < launch_price_sol() < 1e-6
