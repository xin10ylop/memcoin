"""Tests for the real-time launch scanner."""
from __future__ import annotations

import pytest

from alpha.data.pumpportal import Launch
from alpha.execution.launch_scanner import LaunchScanner, ScannerConfig


def launch(symbol="TST", dev_buy=2.0, dev="DEV", uri="https://meta", v_sol=None):
    return Launch.from_event({
        "txType": "create", "mint": f"MINT{symbol}", "traderPublicKey": dev,
        "signature": "SIG", "solAmount": dev_buy, "initialBuy": 1e6,
        "bondingCurveKey": "BC",
        "vSolInBondingCurve": 30.0 + dev_buy if v_sol is None else v_sol,
        "vTokensInBondingCurve": 9e8, "marketCapSol": 30.0,
        "name": symbol, "symbol": symbol, "uri": uri, "pool": "pump",
    })


class FakeScore:
    """Stand-in for a DeployerScore."""

    def __init__(self, launches=10, graduations=4, lower_bound=0.30):
        self.launches = launches
        self.graduations = graduations
        self.lower_bound = lower_bound
        self.posterior_rate = lower_bound


class FakeRegistry:
    def __init__(self, score):
        self._score = score

    def score(self, wallet):  # noqa: D102
        return self._score


def test_launch_with_no_deployer_buy_is_rejected():
    result = LaunchScanner().scan(launch(dev_buy=0.0))
    assert not result.approved
    assert "none of their own" in result.rejected_by


def test_oversized_deployer_allocation_is_rejected():
    result = LaunchScanner().scan(launch(dev_buy=30.0))
    assert not result.approved
    assert "too large" in result.rejected_by


def test_missing_metadata_is_rejected():
    result = LaunchScanner().scan(launch(uri=""))
    assert not result.approved
    assert "metadata" in result.rejected_by


def test_launch_already_up_the_curve_is_rejected():
    """A large same-block buy means the cheap part of the curve is gone."""
    result = LaunchScanner().scan(launch(dev_buy=1.0, v_sol=60.0))
    assert not result.approved
    assert "up the curve" in result.rejected_by


def test_unknown_deployer_is_rejected_on_expected_value():
    result = LaunchScanner().scan(launch())
    assert not result.approved
    assert "break-even" in result.rejected_by


def test_short_deployer_history_is_rejected_as_noise():
    """A 1-for-1 record is luck, not evidence."""
    scanner = LaunchScanner(deployer_registry=FakeRegistry(FakeScore(launches=1, graduations=1, lower_bound=0.99)))
    result = scanner.scan(launch())
    assert not result.approved
    assert "prior launches" in result.rejected_by


def test_elite_deployer_with_real_history_produces_a_signal():
    scanner = LaunchScanner(deployer_registry=FakeRegistry(FakeScore(launches=20, graduations=9, lower_bound=0.35)))
    result = scanner.scan(launch())
    assert result.approved
    assert result.verdict is not None and result.verdict.edge > 0


def test_signal_callback_fires_on_approval():
    seen = []
    scanner = LaunchScanner(
        deployer_registry=FakeRegistry(FakeScore(launches=20, graduations=9, lower_bound=0.35)),
        on_signal=seen.append,
    )
    scanner.scan(launch())
    assert len(seen) == 1


def test_rate_limit_caps_signals():
    scanner = LaunchScanner(
        deployer_registry=FakeRegistry(FakeScore(launches=20, graduations=9, lower_bound=0.35)),
        config=ScannerConfig(max_signals_per_hour=2),
    )
    results = [scanner.scan(launch(symbol=f"T{i}")) for i in range(5)]
    assert sum(1 for r in results if r.approved) == 2
    assert any("rate limit" in r.rejected_by for r in results)


def test_registry_failure_does_not_crash_the_scan():
    class Broken:
        def score(self, wallet):
            raise RuntimeError("registry down")

    result = LaunchScanner(deployer_registry=Broken()).scan(launch())
    assert not result.approved     # falls through to the EV gate and is refused


def test_signal_rate_sanity_check_flags_an_implausible_rate():
    """If the scanner starts approving broadly, something is broken."""
    scanner = LaunchScanner()
    scanner.stats.seen = 100
    scanner.stats.signalled = 40
    assert scanner.expected_signal_rate()["sane"] is False


def test_stats_track_rejection_reasons():
    scanner = LaunchScanner()
    scanner.scan(launch(dev_buy=0.0))
    scanner.scan(launch(uri=""))
    assert scanner.stats.seen == 2
    assert len(scanner.stats.rejections) == 2
