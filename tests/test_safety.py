"""Tests for safety screening verdict logic."""
from __future__ import annotations

from alpha.safety.rugcheck import Holder, RugCheckReport
from alpha.safety.screen import SafetyReport, SafetyScreener, SafetyThresholds, Severity, Verdict
from alpha.safety.solana_rpc import MintInfo


def clean_mint(**kw):
    defaults = dict(mint="M", ok=True, decimals=6, supply=1e9, mint_authority=None,
                    freeze_authority=None, is_initialized=True, program="Tokenkeg", extensions=[])
    defaults.update(kw)
    return MintInfo(**defaults)


def clean_report(**kw):
    defaults = dict(mint="M", ok=True, score=1, score_normalised=1, rugged=False, risks=[],
                    creator="C", creator_token_count=1, creator_balance_pct=0.5,
                    total_holders=200, total_market_liquidity=50_000,
                    top_holders=[Holder("a", "o1", 5.0, False), Holder("b", "o2", 3.0, False)],
                    raw={})
    defaults.update(kw)
    return RugCheckReport(**defaults)


def screen(mint_info, rugcheck, liquidity=50_000.0, thresholds=None):
    """Drive the screener's check logic without touching the network."""
    screener = SafetyScreener.__new__(SafetyScreener)
    screener.t = thresholds or SafetyThresholds()
    report = SafetyReport(mint="M", mint_info=mint_info, rugcheck=rugcheck)
    screener._check_authorities(report, mint_info)
    screener._check_extensions(report, mint_info)
    screener._check_rugcheck_verdict(report, rugcheck)
    screener._check_concentration(report, rugcheck)
    screener._check_creator(report, rugcheck)
    screener._check_liquidity(report, rugcheck, liquidity)
    screener._check_metadata(report, rugcheck, mint_info)
    return report


def test_clean_token_passes():
    assert screen(clean_mint(), clean_report()).verdict is Verdict.PASS


def test_live_mint_authority_is_a_structural_reject():
    report = screen(clean_mint(mint_authority="SOMEBODY"), clean_report())
    assert report.verdict is Verdict.REJECT
    assert any(c.name == "mint_authority_revoked" for c in report.blocking)


def test_live_freeze_authority_is_a_structural_reject():
    report = screen(clean_mint(freeze_authority="SOMEBODY"), clean_report())
    assert report.verdict is Verdict.REJECT


def test_dangerous_token2022_extension_is_rejected():
    report = screen(clean_mint(extensions=["permanentDelegate"]), clean_report())
    assert report.verdict is Verdict.REJECT
    assert any(c.name == "no_dangerous_extensions" for c in report.blocking)


def test_benign_token2022_extensions_are_allowed():
    report = screen(clean_mint(extensions=["metadataPointer", "tokenMetadata"]), clean_report())
    assert report.verdict is Verdict.PASS


def test_transfer_fee_above_the_limit_is_rejected():
    assert screen(clean_mint(transfer_fee_bps=500), clean_report()).verdict is Verdict.REJECT


def test_concentrated_single_holder_is_rejected():
    rc = clean_report(top_holders=[Holder("a", "o1", 60.0, False)])
    assert screen(clean_mint(), rc).verdict is Verdict.REJECT


def test_pool_accounts_are_excluded_from_concentration():
    """The AMM's own balance is not a holder risk."""
    rc = clean_report(
        top_holders=[Holder("amm", "poolowner", 95.0, False), Holder("b", "o2", 2.0, False)],
        raw={"knownAccounts": {"poolowner": {"name": "Pump Fun", "type": "AMM"}}},
    )
    assert screen(clean_mint(), rc).verdict is Verdict.PASS


def test_serial_deployer_is_rejected():
    rc = clean_report(creator_token_count=80)
    report = screen(clean_mint(), rc)
    assert report.verdict is Verdict.REJECT
    assert any(c.name == "creator_not_serial_deployer" for c in report.blocking)


def test_deployer_rug_history_is_structural():
    rc = clean_report(risks=[{"name": "Creator history of rugged tokens", "level": "danger"}])
    assert screen(clean_mint(), rc).verdict is Verdict.REJECT


def test_low_liquidity_is_immature_not_a_permanent_reject():
    """A young token is not a malicious one — it must remain retryable."""
    report = screen(clean_mint(), clean_report(), liquidity=100.0)
    assert report.verdict is Verdict.IMMATURE
    assert report.retryable
    assert not report.blocking


def test_low_holder_count_is_immature():
    report = screen(clean_mint(), clean_report(total_holders=3))
    assert report.verdict is Verdict.IMMATURE


def test_rugcheck_low_liquidity_danger_does_not_cause_a_permanent_reject():
    """RugCheck flags thin liquidity as 'danger'; for a new launch it is maturity."""
    rc = clean_report(risks=[{"name": "Low Liquidity", "level": "danger"}])
    assert screen(clean_mint(), rc).verdict is Verdict.PASS


def test_unrecognised_danger_risks_fail_closed():
    rc = clean_report(risks=[{"name": "Some Novel Attack", "level": "danger"}])
    assert screen(clean_mint(), rc).verdict is Verdict.REJECT


def test_unreadable_mint_fails_closed():
    report = screen(MintInfo(mint="M", ok=False), clean_report())
    assert report.verdict is Verdict.REJECT


def test_screening_error_is_a_reject():
    report = SafetyReport(mint="M", error="network down")
    assert report.verdict is Verdict.REJECT
    assert not report.passed


def test_risk_score_is_zero_for_a_clean_pass_and_100_for_a_reject():
    assert screen(clean_mint(), clean_report()).risk_score == 0.0
    assert screen(clean_mint(mint_authority="X"), clean_report()).risk_score == 100.0
