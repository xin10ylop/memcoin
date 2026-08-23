"""Tests for the real-time launch stream."""
from __future__ import annotations

import pytest

from alpha.data.pumpportal import INITIAL_VSOL, Launch, Migration, PumpPortalStream


def create_event(**kw):
    event = {
        "signature": "SIG", "mint": "MINT", "traderPublicKey": "DEV", "txType": "create",
        "initialBuy": 153_285_714.28, "solAmount": 5.0, "bondingCurveKey": "BC",
        "vTokensInBondingCurve": 919_714_285.7, "vSolInBondingCurve": 35.0,
        "marketCapSol": 38.05, "name": "Test", "symbol": "TST",
        "uri": "https://example/meta.json", "pool": "pump",
    }
    event.update(kw)
    return event


def test_create_event_parses():
    launch = Launch.from_event(create_event())
    assert launch is not None
    assert launch.mint == "MINT"
    assert launch.dev_wallet == "DEV"
    assert launch.dev_buy_sol == 5.0


def test_non_create_events_are_ignored():
    assert Launch.from_event({"txType": "buy", "mint": "M"}) is None
    assert Launch.from_event({"txType": "create"}) is None      # no mint
    assert Launch.from_event({}) is None


def test_dev_curve_share_is_relative_to_graduation():
    """5 SOL of a required 85.0054 is ~5.9% of the curve, bought at the lowest prices."""
    launch = Launch.from_event(create_event(solAmount=5.0))
    assert launch.dev_curve_share == pytest.approx(5.0 / 85.0054, rel=1e-6)


def test_zero_dev_buy_is_detected():
    launch = Launch.from_event(create_event(solAmount=0, vSolInBondingCurve=INITIAL_VSOL))
    assert not launch.is_self_funded
    assert launch.dev_curve_share == 0.0
    assert launch.net_sol == 0.0


def test_net_sol_is_measured_from_the_initial_reserve():
    launch = Launch.from_event(create_event(vSolInBondingCurve=45.0))
    assert launch.net_sol == pytest.approx(15.0)


def test_malformed_numbers_do_not_raise():
    launch = Launch.from_event(create_event(solAmount="not-a-number", marketCapSol=None))
    assert launch is not None
    assert launch.dev_buy_sol == 0.0
    assert launch.market_cap_sol == 0.0


def test_migration_event_parses():
    migration = Migration.from_event({"txType": "migrate", "mint": "M", "signature": "S", "pool": "pumpswap"})
    assert migration is not None and migration.mint == "M"


def test_migration_ignores_other_events():
    assert Migration.from_event({"txType": "create", "mint": "M"}) is None


def test_row_round_trips_every_field():
    row = Launch.from_event(create_event()).to_row()
    for key in ("mint", "dev_wallet", "dev_buy_sol", "dev_curve_share", "v_sol", "uri", "observed_at"):
        assert key in row


@pytest.mark.asyncio
async def test_dispatch_routes_and_counts():
    import json

    stream = PumpPortalStream()
    launches, migrations = [], []
    await stream._dispatch(json.dumps(create_event()), launches.append, migrations.append)
    await stream._dispatch(
        json.dumps({"txType": "migrate", "mint": "M2", "signature": "S", "pool": "p"}),
        launches.append, migrations.append,
    )
    # Subscription acknowledgements must not be counted as data.
    await stream._dispatch(json.dumps({"message": "Subscribed"}), launches.append, migrations.append)
    await stream._dispatch("not json", launches.append, migrations.append)

    assert len(launches) == 1 and len(migrations) == 1
    assert stream.stats["launches"] == 1
    assert stream.stats["migrations"] == 1
    assert stream.stats["errors"] == 1
