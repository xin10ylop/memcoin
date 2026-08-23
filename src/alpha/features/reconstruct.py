"""Retroactive panel reconstruction from swap history.

The collector builds an unbiased panel going forward, but that takes wall-clock
time: a token's outcome is not known until its horizon has elapsed. This module
solves the resulting cold-start problem.

GeckoTerminal's ``trades`` endpoint returns the last ~300 swaps for a pool, each
carrying a timestamp, the trader's wallet, direction and USD size. For a young
pool that is frequently its *entire* trading history. From it we can recompute,
at any chosen instant, exactly the quantities the snapshot API reports —
buys, sells, unique buyers, unique sellers and volume over trailing windows —
and therefore rebuild the same feature vectors the live path would have seen.

Two calls per pool (trades + OHLCV) then yield many labelled decision points,
instead of one per observation cycle.

**Why this does not reintroduce survivorship bias.** The cohort is still chosen
prospectively: we only reconstruct pools the collector already recorded at
birth, before any outcome was known. Reconstruction changes how densely we
sample each pool's history, not which pools are in the sample. Applying the same
technique to a pool list gathered from today's *trending* page would be badly
biased, because that list is conditioned on success.

**Limitation, stated plainly.** Reconstruction is only faithful when the 300
returned trades reach back past the decision point. :func:`coverage_ok` checks
this, and rows failing it are dropped rather than silently approximated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

log = logging.getLogger(__name__)

# Trailing windows the snapshot API exposes, in seconds.
WINDOWS = {"m5": 300, "m15": 900, "h1": 3600, "h24": 86400}


@dataclass
class ReconstructedSnapshot:
    """A snapshot rebuilt from swap history, shaped like a stored panel row."""

    pool: str
    observed_at: str
    ts: int
    age_min: float
    price_usd: float
    liquidity_usd: float
    fdv_usd: float
    window_stats: dict[str, dict[str, float]]

    def as_row(self) -> dict[str, Any]:
        """Emit the same column layout the live snapshot table uses, so the
        feature builder cannot tell reconstructed rows from recorded ones."""
        row: dict[str, Any] = {
            "pool": self.pool,
            "observed_at": self.observed_at,
            "age_min": self.age_min,
            "price_usd": self.price_usd,
            "price_native": 0.0,
            "fdv_usd": self.fdv_usd,
            "liquidity_usd": self.liquidity_usd,
        }
        for tf, stats in self.window_stats.items():
            row[f"buys_{tf}"] = stats["buys"]
            row[f"sells_{tf}"] = stats["sells"]
            row[f"buyers_{tf}"] = stats["buyers"]
            row[f"sellers_{tf}"] = stats["sellers"]
            row[f"vol_{tf}"] = stats["volume"]
            row[f"chg_{tf}"] = stats["chg"]
        return row

    def keys(self) -> list[str]:
        return list(self.as_row())


def coverage_ok(trades: Sequence[Any], decision_ts: int, created_ts: int | None) -> bool:
    """Whether the trade sample reaches back far enough to be faithful at ``decision_ts``.

    If the oldest trade we hold is *after* the decision point, the windows would
    be computed from a truncated history and would understate activity. The one
    exception is when the oldest trade is at or before pool creation, meaning we
    genuinely have the complete history.
    """
    if not trades:
        return False
    oldest = min(_ts(t) for t in trades)
    if created_ts is not None and oldest <= created_ts + 60:
        return True
    # Require a full hour of history behind the decision point, since h1 is the
    # longest window that materially drives the features.
    return oldest <= decision_ts - 3600


def reconstruct_at(
    pool: str,
    trades: Sequence[Any],
    decision_ts: int,
    *,
    created_ts: int | None = None,
    liquidity_usd: float = 0.0,
    fdv_usd: float = 0.0,
) -> ReconstructedSnapshot | None:
    """Rebuild the snapshot a live observer would have seen at ``decision_ts``.

    Only trades at or before ``decision_ts`` are considered — the same
    point-in-time discipline the live feature path enforces.
    """
    past = [t for t in trades if _ts(t) <= decision_ts]
    if not past:
        return None

    price = _price(max(past, key=_ts))
    if price <= 0:
        return None

    window_stats: dict[str, dict[str, float]] = {}
    for name, seconds in WINDOWS.items():
        cutoff = decision_ts - seconds
        recent = [t for t in past if _ts(t) > cutoff]
        buyers: set[str] = set()
        sellers: set[str] = set()
        buys = sells = 0
        volume = 0.0
        for t in recent:
            kind, wallet = _kind(t), _wallet(t)
            volume += _volume(t)
            if kind == "buy":
                buys += 1
                if wallet:
                    buyers.add(wallet)
            else:
                sells += 1
                if wallet:
                    sellers.add(wallet)
        # Price change over the window, from the earliest trade inside it.
        chg = 0.0
        if recent:
            first_price = _price(min(recent, key=_ts))
            if first_price > 0:
                chg = 100.0 * (price / first_price - 1.0)
        window_stats[name] = {
            "buys": float(buys), "sells": float(sells),
            "buyers": float(len(buyers)), "sellers": float(len(sellers)),
            "volume": volume, "chg": chg,
        }

    age_min = ((decision_ts - created_ts) / 60.0) if created_ts else 0.0
    return ReconstructedSnapshot(
        pool=pool,
        observed_at=datetime.fromtimestamp(decision_ts, tz=timezone.utc).isoformat(),
        ts=decision_ts,
        age_min=max(0.0, age_min),
        price_usd=price,
        liquidity_usd=liquidity_usd,
        fdv_usd=fdv_usd,
        window_stats=window_stats,
    )


def reconstruct_series(
    pool: str,
    trades: Sequence[Any],
    decision_times: Sequence[int],
    *,
    created_ts: int | None = None,
    liquidity_usd: float = 0.0,
    fdv_usd: float = 0.0,
) -> list[dict[str, Any]]:
    """Reconstruct an ascending series of snapshot rows for one pool."""
    rows: list[dict[str, Any]] = []
    for ts in sorted(decision_times):
        if not coverage_ok(trades, ts, created_ts):
            continue
        snap = reconstruct_at(
            pool, trades, ts, created_ts=created_ts,
            liquidity_usd=liquidity_usd, fdv_usd=fdv_usd,
        )
        if snap is not None:
            rows.append(snap.as_row())
    return rows


def estimate_liquidity_at(
    trades: Sequence[Any], decision_ts: int, current_liquidity_usd: float, current_ts: int
) -> float:
    """Approximate pool liquidity at a past instant.

    Liquidity is not reported per trade, so we scale today's figure by the
    square root of the price ratio: in a constant-product pool the quote reserve
    moves as sqrt(price) when value is added or removed symmetrically. This is
    an approximation and is used only for cost modelling, never as a feature.
    """
    past = [t for t in trades if _ts(t) <= decision_ts]
    recent = [t for t in trades if _ts(t) <= current_ts]
    if not past or not recent or current_liquidity_usd <= 0:
        return current_liquidity_usd
    p_then = _price(max(past, key=_ts))
    p_now = _price(max(recent, key=_ts))
    if p_then <= 0 or p_now <= 0:
        return current_liquidity_usd
    scaled = current_liquidity_usd * (p_then / p_now) ** 0.5
    return max(0.0, scaled)


def _ts(t: Any) -> int:
    value = t["ts"] if hasattr(t, "keys") else getattr(t, "timestamp", None)
    if isinstance(value, str):
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    if isinstance(value, datetime):
        return int(value.timestamp())
    return int(value or 0)


def _kind(t: Any) -> str:
    return str(t["kind"] if hasattr(t, "keys") else getattr(t, "kind", ""))


def _wallet(t: Any) -> str:
    return str(t["wallet"] if hasattr(t, "keys") else getattr(t, "wallet", ""))


def _volume(t: Any) -> float:
    return float(t["volume_usd"] if hasattr(t, "keys") else getattr(t, "volume_usd", 0.0))


def _price(t: Any) -> float:
    return float(t["price_usd"] if hasattr(t, "keys") else getattr(t, "price_usd", 0.0))
