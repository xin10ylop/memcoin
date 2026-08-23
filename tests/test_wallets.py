"""Tests for wash-trading detection and smart-money scoring."""
from __future__ import annotations

import pytest

from alpha.features.wallets import detect_wash_trading, profile_wallets, score_wallets


def trade(wallet, kind, ts, volume=100.0):
    return {"wallet": wallet, "kind": kind, "ts": ts, "volume_usd": volume, "price_usd": 1e-6}


def test_organic_flow_is_not_flagged():
    """Most wallets trade once, sizes vary widely — the retail signature."""
    trades = []
    for i in range(60):
        trades.append(trade(f"w{i}", "buy", 1000 + i * 30, volume=5 + (i * 37) % 900))
    report = detect_wash_trading("P", trades)
    assert not report.is_suspicious


def test_uniform_wallet_cohort_is_flagged():
    """Ten wallets each cycling the same number of trades at the same size."""
    trades = []
    for w in range(10):
        for i in range(20):
            kind = "buy" if i % 2 == 0 else "sell"
            trades.append(trade(f"bot{w}", kind, 1000 + i * 10, volume=500.0))
    report = detect_wash_trading("P", trades)
    assert report.is_suspicious
    assert report.suspect_volume_share > 0.5


def test_adjusted_volume_removes_the_suspect_share():
    trades = []
    for w in range(8):
        for i in range(20):
            trades.append(trade(f"bot{w}", "buy" if i % 2 == 0 else "sell", 1000 + i * 10, 500.0))
    report = detect_wash_trading("P", trades)
    assert report.adjusted_volume(100_000) < 100_000


def test_too_little_activity_is_not_judged():
    report = detect_wash_trading("P", [trade("w1", "buy", 1)])
    assert not report.is_suspicious
    assert "too little activity" in report.reasons[0]


def test_round_trip_ratio_detects_flat_wallets():
    profiles = profile_wallets(
        [trade("w", "buy", i) for i in range(10)] + [trade("w", "sell", 10 + i) for i in range(10)]
    )
    assert profiles["w"].round_trip_ratio == pytest.approx(1.0)


def test_size_variation_is_measured():
    varied = profile_wallets([trade("w", "buy", i, volume=10 ** (i % 4)) for i in range(12)])
    uniform = profile_wallets([trade("w", "buy", i, volume=100.0) for i in range(12)])
    assert varied["w"].size_cv > uniform["w"].size_cv


def test_wallet_scoring_requires_a_minimum_track_record():
    """Three wins from three tokens is luck, not skill."""
    scores = score_wallets({"lucky": [("t", 1.0)] * 3}, base_rate=0.25, min_tokens=8)
    assert scores == []


def test_wallet_scoring_shrinks_short_records_toward_the_base_rate():
    entries = {
        "short": [("t", 1.0)] * 8,                       # 8/8 wins
        "long": [("t", 1.0)] * 45 + [("t", -1.0)] * 15,  # 45/60 wins
    }
    ranked = score_wallets(entries, base_rate=0.25, min_tokens=8)
    scores = {s.wallet: s for s in ranked}
    # The longer record must rank first despite the lower raw win rate, because
    # a short perfect record is what luck looks like when you screen thousands.
    assert ranked[0].wallet == "long"
    assert scores["long"].score > scores["short"].score
    assert scores["long"].win_rate < scores["short"].win_rate


def test_consistently_bad_wallets_are_marked_avoid():
    scores = score_wallets({"bad": [("t", -1.0)] * 30}, base_rate=0.30, min_tokens=8)
    assert scores[0].verdict == "avoid"
