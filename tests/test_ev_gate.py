"""Tests for the expected-value gate on bonding-curve entries."""
from __future__ import annotations

import pytest

from alpha.data.bondingcurve import (
    GRADUATION_SOL,
    PLATFORM_GRADUATION_RATE,
    breakeven_graduation_probability,
    graduation_edge,
    post_migration_dead_liquidity,
)
from alpha.risk.ev_gate import EvGate, EvGateConfig


# --------------------------------------------------------------- curve maths

def test_breakeven_at_launch_is_6_8_percent():
    """p* = vSol^2/115.0054^2, and vSol starts at 30."""
    assert breakeven_graduation_probability(0.0) == pytest.approx(0.0680, abs=0.0005)


def test_breakeven_reaches_certainty_at_graduation():
    assert breakeven_graduation_probability(GRADUATION_SOL) == pytest.approx(1.0, abs=1e-6)


def test_breakeven_rises_monotonically_along_the_curve():
    values = [breakeven_graduation_probability(n) for n in (0, 10, 30, 50, 70, 85)]
    assert values == sorted(values)


def test_breakeven_is_quadratic_in_virtual_reserve():
    """Doubling the virtual reserve quadruples the required probability."""
    # vSol goes 30 -> 60 when net_sol goes 0 -> 30.
    assert breakeven_graduation_probability(30.0) == pytest.approx(
        4 * breakeven_graduation_probability(0.0), rel=1e-6
    )


def test_unconditional_hold_to_graduation_is_negative_ev_everywhere():
    """The central finding: no entry point on the curve is viable at base rate."""
    for net in (0, 5, 10, 20, 40, 60, 80):
        assert graduation_edge(net, PLATFORM_GRADUATION_RATE) < 0


def test_edge_turns_positive_once_probability_clears_breakeven():
    breakeven = breakeven_graduation_probability(0.0)
    assert graduation_edge(0.0, breakeven * 0.5) < 0
    assert graduation_edge(0.0, breakeven * 2.0) > 0
    assert graduation_edge(0.0, breakeven) == pytest.approx(0.0, abs=1e-9)


def test_post_migration_cohort_is_negative_sum():
    """~21% of migrated SOL cannot be extracted by holders collectively."""
    assert post_migration_dead_liquidity() == pytest.approx(0.207, abs=0.01)


# ------------------------------------------------------------------ the gate

def test_gate_rejects_unconditional_entries():
    gate = EvGate()
    for net in (0, 5, 20, 40):
        assert not gate.evaluate(net_sol=net).approved


def test_gate_approves_an_elite_deployer_early_on_the_curve():
    verdict = EvGate().evaluate(net_sol=5.0, p_graduate=0.40)
    assert verdict.approved
    assert verdict.margin > 1.5


def test_gate_requires_a_margin_not_merely_breakeven():
    """A probability that only just clears break-even is a coin flip plus costs."""
    breakeven = breakeven_graduation_probability(5.0)
    gate = EvGate(EvGateConfig(safety_margin=1.5))
    assert not gate.evaluate(net_sol=5.0, p_graduate=breakeven * 1.05).approved
    assert gate.evaluate(net_sol=5.0, p_graduate=breakeven * 1.6).approved


def test_gate_rejects_a_good_deployer_who_enters_late():
    """Payoff shrinks quadratically up the curve; a fixed edge stops being enough."""
    gate = EvGate()
    assert gate.evaluate(net_sol=5.0, p_graduate=0.40).approved
    assert not gate.evaluate(net_sol=50.0, p_graduate=0.40).approved


def test_gate_enforces_a_hard_ceiling_on_curve_position():
    gate = EvGate(EvGateConfig(max_net_sol=40.0))
    verdict = gate.evaluate(net_sol=60.0, p_graduate=0.99)
    assert not verdict.approved
    assert any("far up the curve" in r for r in verdict.reasons)


def test_gate_uses_the_deployer_lower_bound_not_the_point_estimate():
    """Sizing an all-or-nothing bet off a short lucky record is the failure mode."""
    class Score:
        launches, graduations = 3, 3
        posterior_rate, lower_bound = 0.95, 0.05

    verdict = EvGate().evaluate(net_sol=5.0, deployer_score=Score())
    assert verdict.estimated_p == pytest.approx(0.05)
    assert not verdict.approved


def test_gate_falls_back_to_the_base_rate_without_history():
    verdict = EvGate().evaluate(net_sol=5.0)
    assert verdict.estimated_p == pytest.approx(PLATFORM_GRADUATION_RATE)
    assert any("base rate" in r for r in verdict.reasons)


def test_exit_policy_defaults_to_leaving_before_graduation():
    assert "curve" in EvGate().exit_recommendation(80.0)


def test_exit_policy_can_be_overridden():
    gate = EvGate(EvGateConfig(allow_hold_through_migration=True))
    assert gate.exit_recommendation(80.0) == "hold_through_migration"


def test_verdict_exposes_numeric_features():
    features = EvGate().evaluate(net_sol=5.0, p_graduate=0.4).as_features()
    assert set(features) >= {"ev_breakeven_p", "ev_estimated_p", "ev_edge", "ev_margin"}
    assert all(isinstance(v, float) for v in features.values())
