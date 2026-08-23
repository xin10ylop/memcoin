"""Tests for launch persistence and the deployer track record."""
from __future__ import annotations

from alpha.data.pumpportal import Launch, Migration


def make_launch(mint="M1", dev="DEV1", sol=2.0):
    return Launch.from_event({
        "txType": "create", "mint": mint, "traderPublicKey": dev, "signature": "S",
        "solAmount": sol, "initialBuy": 1000.0, "bondingCurveKey": "BC",
        "vSolInBondingCurve": 30.0 + sol, "vTokensInBondingCurve": 9e8,
        "marketCapSol": 30.0, "name": "n", "symbol": "s", "uri": "u", "pool": "pump",
    })


def test_launch_is_recorded_once(tmp_store):
    assert tmp_store.record_launch(make_launch()) is True
    assert tmp_store.record_launch(make_launch()) is False
    assert tmp_store.launch_stats()["launches"] == 1


def test_deployer_counters_track_launches(tmp_store):
    for i in range(3):
        tmp_store.record_launch(make_launch(mint=f"M{i}", dev="SERIAL", sol=1.5))
    record = tmp_store.dev_record("SERIAL")
    assert record["launches"] == 3
    assert record["total_dev_buy_sol"] == 4.5


def test_repeat_deployers_are_identifiable(tmp_store):
    tmp_store.record_launch(make_launch(mint="A", dev="ONCE"))
    for i in range(4):
        tmp_store.record_launch(make_launch(mint=f"B{i}", dev="FACTORY"))
    stats = tmp_store.launch_stats()
    assert stats["dev_wallets"] == 2
    assert stats["repeat_devs"] == 1


def test_duplicate_launch_does_not_double_count_the_deployer(tmp_store):
    tmp_store.record_launch(make_launch(mint="A", dev="D"))
    tmp_store.record_launch(make_launch(mint="A", dev="D"))
    assert tmp_store.dev_record("D")["launches"] == 1


def test_migration_marks_the_launch_and_credits_the_deployer(tmp_store):
    tmp_store.record_launch(make_launch(mint="GRAD", dev="GOODDEV"))
    migration = Migration.from_event({"txType": "migrate", "mint": "GRAD", "signature": "S", "pool": "ps"})
    assert tmp_store.record_migration(migration) is True
    assert tmp_store.dev_record("GOODDEV")["graduations"] == 1
    assert tmp_store.launch_stats()["graduation_rate"] == 1.0


def test_migration_is_idempotent(tmp_store):
    tmp_store.record_launch(make_launch(mint="G", dev="D"))
    migration = Migration.from_event({"txType": "migrate", "mint": "G", "signature": "S", "pool": "p"})
    tmp_store.record_migration(migration)
    assert tmp_store.record_migration(migration) is False
    assert tmp_store.dev_record("D")["graduations"] == 1


def test_launch_links_to_a_pool(tmp_store):
    tmp_store.record_launch(make_launch(mint="M"))
    tmp_store.link_launch_to_pool("M", "POOL1")
    assert tmp_store.launch_stats()["linked_to_pool"] == 1
