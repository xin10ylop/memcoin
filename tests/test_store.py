"""Tests for the panel store."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alpha.data.geckoterminal import Pool, TimeframeStats


def make_pool(address="POOL1", liq=10_000.0, created_min_ago=5.0):
    now = datetime.now(timezone.utc)
    return Pool(
        address=address, name="TKN / SOL", dex="pumpswap", base_mint="MINT1",
        quote_mint="So11111111111111111111111111111111111111112",
        created_at=now - timedelta(minutes=created_min_ago),
        price_usd=1e-6, price_native=1e-8, fdv_usd=liq * 3, market_cap_usd=0.0,
        liquidity_usd=liq,
        timeframes={tf: TimeframeStats(buys=5, sells=2, buyers=4, sellers=2, volume_usd=500.0)
                    for tf in Pool.TIMEFRAMES},
        observed_at=now,
    )


def test_recording_a_pool_twice_does_not_duplicate_it(tmp_store):
    pools = [make_pool()]
    assert tmp_store.record_pools(pools) == 1
    assert tmp_store.record_pools(pools) == 0
    assert tmp_store.stats()["pools"] == 1


def test_birth_certificate_is_not_overwritten_by_later_observations(tmp_store):
    """The first-seen state must survive, since it is the unbiased record."""
    tmp_store.record_pools([make_pool(liq=1_000)])
    tmp_store.record_pools([make_pool(liq=999_999)])
    row = tmp_store.pool_row("POOL1")
    assert row["birth_liquidity_usd"] == 1_000


def test_snapshots_accumulate_over_time(tmp_store):
    pool = make_pool()
    tmp_store.record_pools([pool])
    tmp_store.record_snapshots([pool])
    later = make_pool()
    later.observed_at = pool.observed_at + timedelta(minutes=5)
    tmp_store.record_snapshots([later])
    assert len(tmp_store.snapshots_for("POOL1")) == 2


def test_due_pools_respects_the_schedule(tmp_store):
    tmp_store.record_pools([make_pool()])
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    tmp_store.schedule_many(["POOL1"], future)
    assert tmp_store.due_pools() == []
    assert tmp_store.due_pools(now=future + timedelta(minutes=1)) == ["POOL1"]


def test_retired_pools_are_not_returned_as_due(tmp_store):
    tmp_store.record_pools([make_pool()])
    tmp_store.schedule_many(["POOL1"], datetime.now(timezone.utc) - timedelta(minutes=1))
    assert tmp_store.due_pools() == ["POOL1"]
    tmp_store.retire(["POOL1"], "test")
    assert tmp_store.due_pools() == []


def test_trades_deduplicate_on_transaction_identity(tmp_store):
    """Regression: a float in the primary key let re-fetches inflate volume."""
    from alpha.data.geckoterminal import Trade

    tmp_store.record_pools([make_pool()])
    now = datetime.now(timezone.utc)
    first = Trade("TX1", 100, "W1", now, "buy", 100.0, 1e-6, 1.0, 1.0)
    # Same swap, volume re-serialised with a slightly different float.
    again = Trade("TX1", 100, "W1", now, "buy", 100.00000000001, 1e-6, 1.0, 1.0)
    tmp_store.record_trades("POOL1", [first])
    tmp_store.record_trades("POOL1", [again])
    assert tmp_store.stats()["trades"] == 1


def test_artifacts_round_trip(tmp_store):
    tmp_store.put_artifact("kind", "key", {"a": 1, "b": [2, 3]})
    assert tmp_store.get_artifact("kind", "key") == {"a": 1, "b": [2, 3]}
    assert tmp_store.get_artifact("kind", "missing") is None
