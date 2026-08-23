"""Shared fixtures."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dataclass
class FakeCandle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume_usd: float = 0.0


@pytest.fixture
def tmp_store(tmp_path):
    from alpha.data.store import Store
    return Store(tmp_path / "test.db")


def make_snapshot(pool="P", age=5.0, price=1e-6, liq=10_000.0, buys=10, sells=5, **kw):
    """Build a snapshot row shaped like the panel table."""
    row = {
        "pool": pool, "observed_at": f"2026-01-01T00:{int(age):02d}:00+00:00",
        "age_min": age, "price_usd": price, "price_native": price / 150,
        "fdv_usd": liq * 3, "liquidity_usd": liq,
    }
    for tf in ("m5", "m15", "h1", "h24"):
        row[f"buys_{tf}"] = buys
        row[f"sells_{tf}"] = sells
        row[f"buyers_{tf}"] = max(1, buys // 2)
        row[f"sellers_{tf}"] = max(1, sells // 2)
        row[f"vol_{tf}"] = buys * 100.0
        row[f"chg_{tf}"] = 0.0
    row.update(kw)
    return row
