"""Dataset assembly — joins point-in-time features to forward-looking labels.

The join is where backtests usually go wrong, so it is worth stating the
contract explicitly:

* Features for a row come from :func:`alpha.features.build_features`, which can
  only see snapshots up to and including the decision index.
* Labels come from :func:`alpha.features.label.label_trade`, which only reads
  candles *strictly after* the decision timestamp.

The two halves are computed from disjoint time ranges that meet exactly at the
decision point. There is no overlap in which information could leak.

A second, subtler bias is handled here too. Each pool can contribute several
decision points, and rows from the same pool are highly correlated — they share
one price path. Treating them as independent samples inflates the effective
sample size and makes any significance test meaningless. Every row therefore
carries its ``pool`` so that cross-validation can split by pool rather than by
row, and a ``sample_weight`` inversely proportional to how many rows that pool
contributed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from alpha.data.geckoterminal import GeckoTerminalClient
from alpha.data.store import Store
from alpha.features.build import FEATURE_NAMES, build_features
from alpha.features.label import Barrier, Label, LabelConfig, label_trade

log = logging.getLogger(__name__)


@dataclass
class DatasetRow:
    """One (features, label) pair plus the execution context needed to replay it.

    Liquidity, price, venue and symbol are carried explicitly rather than being
    reconstructed from features later: the features are log-compressed and
    lossy, and a backtest that recovers position-sizing inputs by inverting them
    would quietly size every trade wrong.
    """

    pool: str
    decision_ts: int
    decision_age_min: float
    features: dict[str, float]
    label: Label
    sample_weight: float = 1.0
    liquidity_usd: float = 0.0
    price_usd: float = 0.0
    dex: str = ""
    symbol: str = ""
    mint: str = ""

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pool": self.pool,
            "decision_ts": self.decision_ts,
            "decision_age_min": self.decision_age_min,
            "sample_weight": self.sample_weight,
            "liquidity_usd": self.liquidity_usd,
            "price_usd": self.price_usd,
            "dex": self.dex,
            "symbol": self.symbol,
            "mint": self.mint,
        }
        row.update(self.features)
        row.update(
            {
                "y_net_return": self.label.net_return,
                "y_is_win": int(self.label.is_win),
                "y_barrier": self.label.barrier.value,
                "y_max_multiple": self.label.max_multiple,
                "y_minutes_held": self.label.minutes_held,
            }
        )
        return row


@dataclass
class DatasetConfig:
    """Which decision points to generate, and how to gate them."""

    label: LabelConfig = field(default_factory=LabelConfig)
    # Only consider entries in this age band. Before the lower bound there is
    # too little history to compute flow features; after the upper bound the
    # launch dynamics this system targets are over.
    min_decision_age_min: float = 3.0
    max_decision_age_min: float = 90.0
    # Minimum liquidity for an entry to be considered realistic at all.
    min_liquidity_usd: float = 2_000.0
    # Cap rows per pool so a single long-lived token cannot dominate training.
    max_rows_per_pool: int = 6
    require_candles_after: int = 3


class DatasetBuilder:
    """Turns the collected panel into a model-ready table."""

    def __init__(
        self,
        store: Store,
        client: GeckoTerminalClient | None = None,
        config: DatasetConfig | None = None,
    ) -> None:
        self.store = store
        self.client = client or GeckoTerminalClient()
        self.cfg = config or DatasetConfig()

    # ------------------------------------------------------------- backfill

    def backfill_candles(self, pools: Sequence[str], max_calls_per_pool: int = 2) -> int:
        """Fetch and cache minute candles for pools we have already discovered.

        Retroactive fetching is legitimate here precisely because the *cohort*
        was fixed prospectively. We chose which pools to study before knowing
        their outcomes; we are only now looking up what those outcomes were.
        Selecting pools retroactively would be biased — measuring pre-selected
        pools retroactively is not.
        """
        total = 0
        for pool in pools:
            try:
                candles = self.client.ohlcv_history(pool, max_calls=max_calls_per_pool)
            except Exception as exc:
                log.debug("candle backfill failed for %s: %s", pool, exc)
                continue
            if candles:
                total += self.store.record_candles(pool, candles)
        return total

    # -------------------------------------------------------------- assembly

    def rows_for_pool(self, pool: str) -> list[DatasetRow]:
        """Generate every valid (features, label) pair for one pool."""
        snapshots = [dict(r) for r in self.store.snapshots_for(pool)]
        if len(snapshots) < 2:
            return []
        candles = [dict(r) for r in self.store.candles_for(pool)]
        if len(candles) < self.cfg.require_candles_after:
            return []

        out: list[DatasetRow] = []
        for idx, snap in enumerate(snapshots):
            if len(out) >= self.cfg.max_rows_per_pool:
                break
            age = float(snap.get("age_min") or 0.0)
            if not (self.cfg.min_decision_age_min <= age <= self.cfg.max_decision_age_min):
                continue
            if float(snap.get("liquidity_usd") or 0.0) < self.cfg.min_liquidity_usd:
                continue
            decision_ts = _epoch(snap.get("observed_at"))
            if decision_ts is None:
                continue
            # Require that candles actually extend past the decision point,
            # otherwise we cannot distinguish "died" from "not yet collected".
            if not any(int(c["ts"]) > decision_ts for c in candles):
                if max(int(c["ts"]) for c in candles) < decision_ts - 3600:
                    continue  # stale backfill, not a genuine death

            label = label_trade(pool, candles, decision_ts, self.cfg.label)
            if label.barrier is Barrier.NO_ENTRY:
                continue
            fv = build_features(snapshots, idx)
            prow = self.store.pool_row(pool)
            out.append(
                DatasetRow(
                    pool=pool,
                    decision_ts=decision_ts,
                    decision_age_min=age,
                    features=fv.values,
                    label=label,
                    liquidity_usd=float(snap.get("liquidity_usd") or 0.0),
                    price_usd=float(snap.get("price_usd") or 0.0),
                    dex=str(prow["dex"]) if prow else "",
                    symbol=str(prow["name"]).split("/")[0].strip() if prow else "",
                    mint=str(prow["base_mint"]) if prow else "",
                )
            )

        # Down-weight pools that contributed many correlated rows.
        if out:
            weight = 1.0 / len(out)
            for row in out:
                row.sample_weight = weight
        return out

    def build(self, pools: Iterable[str] | None = None, limit: int | None = None) -> list[DatasetRow]:
        if pools is None:
            query = "SELECT DISTINCT pool FROM candles"
            pools = [r[0] for r in self.store.conn.execute(query).fetchall()]
        rows: list[DatasetRow] = []
        for pool in pools:
            rows.extend(self.rows_for_pool(pool))
            if limit and len(rows) >= limit:
                break
        return rows

    @staticmethod
    def to_frame(rows: Sequence[DatasetRow]):
        """Convert to a pandas DataFrame with a stable column order."""
        import pandas as pd

        if not rows:
            return pd.DataFrame(columns=["pool", "decision_ts", *FEATURE_NAMES, "y_net_return"])
        frame = pd.DataFrame([r.to_dict() for r in rows])
        ordered = ["pool", "decision_ts", "decision_age_min", "sample_weight",
                   "liquidity_usd", "price_usd", "dex", "symbol", "mint"]
        ordered += [c for c in FEATURE_NAMES if c in frame.columns]
        ordered += [c for c in frame.columns if c.startswith("y_")]
        return frame[[c for c in ordered if c in frame.columns]]


def _epoch(value: Any) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
