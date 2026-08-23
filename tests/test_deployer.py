"""Tests for deployer reputation — the strongest pre-trade signal.

With a 0.63% platform base rate, the dominant failure mode is promoting a
deployer who got lucky over a handful of launches. These tests pin that down.
"""
from __future__ import annotations

import pytest

from alpha.features.deployer import (
    PLATFORM_GRADUATION_RATE,
    DeployerRegistry,
    score_deployer,
)


def test_unknown_deployer_gets_the_platform_prior():
    score = score_deployer("W", launches=0, graduations=0)
    assert score.tier == "unknown"
    assert score.posterior_rate == pytest.approx(PLATFORM_GRADUATION_RATE, rel=1e-9)


def test_small_perfect_record_is_not_promoted():
    """1-from-3 is a 33% raw rate and a 53x apparent lift. It is noise."""
    score = score_deployer("W", launches=3, graduations=1)
    assert score.raw_rate == pytest.approx(1 / 3)
    assert score.posterior_rate < 0.05          # shrunk hard toward the base rate
    assert score.tier not in ("elite", "promising")


def test_substantial_record_earns_elite():
    score = score_deployer("W", launches=20, graduations=8)
    assert score.tier == "elite"
    assert score.lift > 10
    assert score.lower_bound > PLATFORM_GRADUATION_RATE * 5


def test_posterior_is_monotonic_in_graduations():
    rates = [score_deployer("W", 20, g).posterior_rate for g in (0, 2, 5, 10)]
    assert rates == sorted(rates)


def test_more_evidence_tightens_the_lower_bound():
    """Same raw rate, more launches -> more confident, higher lower bound."""
    small = score_deployer("W", launches=10, graduations=4)
    large = score_deployer("W", launches=100, graduations=40)
    assert large.lower_bound > small.lower_bound
    assert large.raw_rate == pytest.approx(small.raw_rate)


def test_high_volume_no_graduations_is_a_factory():
    score = score_deployer("W", launches=50, graduations=0)
    assert score.is_factory
    assert score.posterior_rate < PLATFORM_GRADUATION_RATE


def test_a_few_failures_are_not_condemned():
    """Most deployers launch once and fail; that is the base rate, not a signal."""
    assert score_deployer("W", launches=2, graduations=0).tier == "neutral"


def test_features_are_numeric_and_bounded():
    features = score_deployer("W", 30, 20).as_features()
    assert all(isinstance(v, float) for v in features.values())
    assert features["dev_lift"] <= 200.0
    assert features["dev_is_elite"] == 1.0


def test_registry_reads_from_the_store(tmp_store):
    from alpha.data.pumpportal import Launch, Migration

    def launch(mint, dev):
        return Launch.from_event({
            "txType": "create", "mint": mint, "traderPublicKey": dev, "signature": "S",
            "solAmount": 1.0, "initialBuy": 1.0, "bondingCurveKey": "BC",
            "vSolInBondingCurve": 31.0, "vTokensInBondingCurve": 9e8,
            "marketCapSol": 30.0, "name": "n", "symbol": "s", "uri": "u", "pool": "pump",
        })

    for i in range(6):
        tmp_store.record_launch(launch(f"M{i}", "GOODDEV"))
    for i in range(3):
        tmp_store.record_migration(
            Migration.from_event({"txType": "migrate", "mint": f"M{i}", "signature": "S", "pool": "p"})
        )

    registry = DeployerRegistry(tmp_store)
    score = registry.score("GOODDEV")
    assert score.launches == 6
    assert score.graduations == 3
    assert score.lift > 5


def test_registry_falls_back_to_the_published_rate_until_we_have_enough_data(tmp_store):
    """Our own graduation rate is meaningless on a few hundred launches."""
    registry = DeployerRegistry(tmp_store)
    assert registry.observed_base_rate() == PLATFORM_GRADUATION_RATE


def test_registry_handles_an_empty_wallet(tmp_store):
    assert DeployerRegistry(tmp_store).score("").tier == "unknown"
