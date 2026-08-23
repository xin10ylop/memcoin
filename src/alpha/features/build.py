"""Point-in-time feature engineering.

Every function here obeys one rule: **a feature computed for decision time T may
only use information observable at or before T**. Violating this is the single
easiest way to produce a backtest that shows spectacular returns and loses money
live, because the model learns to read the future rather than predict it.

The rule is enforced structurally rather than by convention: :func:`build_features`
takes a list of snapshots and an index, and physically slices away everything
after that index before computing anything. There is no code path by which a
later snapshot can reach a feature.

Feature families:

* **Level** — size and depth right now (liquidity, FDV, market-cap ratios).
* **Flow** — who is trading and in which direction (buy/sell imbalance, unique
  buyer counts, volume relative to liquidity).
* **Acceleration** — how flow is *changing*, computed by differencing the
  cumulative windows the API exposes. A token whose last 5 minutes are hotter
  than its last hour is in a different state from one that is cooling, even when
  the two look identical on level features alone.
* **Microstructure** — average trade size, buyer concentration, churn.
* **Trajectory** — deltas against our own earlier snapshots of the same pool,
  which is information no single API call contains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# Timeframes present in a stored snapshot row.
TFS = ("m5", "m15", "h1", "h24")


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Division that never raises and never returns a non-finite value."""
    try:
        if b == 0 or b != b or a != a:
            return default
        out = a / b
    except (TypeError, ZeroDivisionError):
        return default
    return out if math.isfinite(out) else default


def _log1p_signed(x: float) -> float:
    """Signed log compression.

    Memecoin quantities span many orders of magnitude — liquidity from $50 to
    $50M in the same cross-section. Raw magnitudes make tree splits and any
    linear model behave badly, and a handful of whales dominate every mean.
    Signed log keeps sign, compresses scale, and is defined at zero.
    """
    if x != x or not math.isfinite(x):
        return 0.0
    return math.copysign(math.log1p(abs(x)), x)


def _get(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key) if isinstance(row, Mapping) else None
    if value is None:
        try:
            value = row[key]  # sqlite3.Row supports index-by-name
        except (KeyError, IndexError, TypeError):
            return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass
class FeatureVector:
    """Named feature values for one (pool, decision-time) pair."""

    pool: str
    observed_at: str
    age_min: float
    values: dict[str, float] = field(default_factory=dict)

    def as_list(self, names: Sequence[str]) -> list[float]:
        return [self.values.get(n, 0.0) for n in names]

    def __getitem__(self, key: str) -> float:
        return self.values[key]


# --------------------------------------------------------------------- windows

def _window_deltas(row: Mapping[str, Any]) -> dict[str, float]:
    """Differences between nested cumulative windows.

    The API reports *cumulative* stats per window: ``m15`` includes ``m5``, and
    ``h1`` includes ``m15``. Subtracting yields the activity in the exclusive
    band — e.g. minutes 5–15 — which is what makes acceleration measurable.
    Without this step, ``m5`` and ``h1`` are highly collinear and a model
    cannot distinguish "accelerating" from "large".
    """
    out: dict[str, float] = {}
    vol_m5, vol_m15, vol_h1 = (_get(row, f"vol_{t}") for t in ("m5", "m15", "h1"))
    tx_m5 = _get(row, "buys_m5") + _get(row, "sells_m5")
    tx_m15 = _get(row, "buys_m15") + _get(row, "sells_m15")
    tx_h1 = _get(row, "buys_h1") + _get(row, "sells_h1")

    # Exclusive-band activity (clamped: cumulative windows can be inconsistent
    # across a refresh boundary and go slightly negative).
    out["vol_5_15"] = max(0.0, vol_m15 - vol_m5)
    out["vol_15_60"] = max(0.0, vol_h1 - vol_m15)
    out["tx_5_15"] = max(0.0, tx_m15 - tx_m5)
    out["tx_15_60"] = max(0.0, tx_h1 - tx_m15)

    # Rate per minute in each band, then the ratio of recent to older rate.
    rate_m5 = vol_m5 / 5.0
    rate_5_15 = out["vol_5_15"] / 10.0
    rate_15_60 = out["vol_15_60"] / 45.0
    # >1 means the token is heating up; <1 means it is cooling.
    out["vol_accel_short"] = _safe_div(rate_m5, rate_5_15, default=1.0 if rate_m5 == 0 else 3.0)
    out["vol_accel_long"] = _safe_div(rate_5_15, rate_15_60, default=1.0 if rate_5_15 == 0 else 3.0)

    trate_m5 = tx_m5 / 5.0
    trate_5_15 = out["tx_5_15"] / 10.0
    out["tx_accel_short"] = _safe_div(trate_m5, trate_5_15, default=1.0 if trate_m5 == 0 else 3.0)
    return out


def _flow_features(row: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    liq = _get(row, "liquidity_usd")
    fdv = _get(row, "fdv_usd")

    for tf in TFS:
        buys, sells = _get(row, f"buys_{tf}"), _get(row, f"sells_{tf}")
        buyers, sellers = _get(row, f"buyers_{tf}"), _get(row, f"sellers_{tf}")
        vol = _get(row, f"vol_{tf}")
        txns = buys + sells

        # Buy pressure. 0.5 is neutral; sustained >0.5 means net accumulation.
        out[f"buy_ratio_{tf}"] = _safe_div(buys, txns, 0.5)
        # Unique-trader ratio separates many wallets from one wallet churning.
        out[f"trader_ratio_{tf}"] = _safe_div(buyers, buyers + sellers, 0.5)
        out[f"txns_{tf}"] = _log1p_signed(txns)
        out[f"unique_traders_{tf}"] = _log1p_signed(buyers + sellers)
        # Volume relative to liquidity: how many times the pool has turned over.
        # This is scale-free, so it compares a $5k token to a $5M one.
        out[f"turnover_{tf}"] = _safe_div(vol, liq)
        out[f"vol_per_txn_{tf}"] = _log1p_signed(_safe_div(vol, txns))
        # Repeat-trading intensity: >1 means wallets are trading more than once.
        out[f"txn_per_trader_{tf}"] = _safe_div(txns, buyers + sellers, 1.0)
        out[f"chg_{tf}"] = _get(row, f"chg_{tf}") / 100.0

    out["net_buyers_m5"] = _log1p_signed(_get(row, "buyers_m5") - _get(row, "sellers_m5"))
    out["net_buys_m5"] = _log1p_signed(_get(row, "buys_m5") - _get(row, "sells_m5"))
    # Liquidity as a fraction of fully-diluted value. Very low means the float
    # is thin relative to notional value: a small sell moves price a lot.
    out["liq_to_fdv"] = _safe_div(liq, fdv)
    return out


def _level_features(row: Mapping[str, Any], age_min: float) -> dict[str, float]:
    liq = _get(row, "liquidity_usd")
    fdv = _get(row, "fdv_usd")
    return {
        "log_liquidity": _log1p_signed(liq),
        "log_fdv": _log1p_signed(fdv),
        "log_price": _log1p_signed(_get(row, "price_usd") * 1e9),
        "log_age_min": math.log1p(max(0.0, age_min)),
        "log_vol_h24": _log1p_signed(_get(row, "vol_h24")),
        # Liquidity accumulated per minute of life — a growth rate that is
        # comparable across tokens of different ages.
        "liq_per_min": _log1p_signed(_safe_div(liq, max(age_min, 1.0))),
    }


def _trajectory_features(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Deltas against our own earlier observations of this pool.

    This is information no single API response contains, and it is the part of
    the feature set hardest for a competitor to replicate without having also
    been recording the token since birth.
    """
    out = {
        "n_observations": 0.0,
        "liq_growth": 0.0,
        "liq_growth_recent": 0.0,
        "price_growth": 0.0,
        "price_growth_recent": 0.0,
        "liq_drawdown": 0.0,
        "price_drawdown": 0.0,
        "buy_ratio_trend": 0.0,
        "obs_span_min": 0.0,
    }
    if len(history) < 2:
        return out

    out["n_observations"] = float(len(history))
    first, last = history[0], history[-1]
    prev = history[-2]

    liq_first, liq_last, liq_prev = (_get(r, "liquidity_usd") for r in (first, last, prev))
    px_first, px_last, px_prev = (_get(r, "price_usd") for r in (first, last, prev))

    out["liq_growth"] = _log1p_signed(_safe_div(liq_last - liq_first, liq_first))
    out["liq_growth_recent"] = _log1p_signed(_safe_div(liq_last - liq_prev, liq_prev))
    out["price_growth"] = _log1p_signed(_safe_div(px_last - px_first, px_first))
    out["price_growth_recent"] = _log1p_signed(_safe_div(px_last - px_prev, px_prev))

    # Drawdown from the best level we have personally observed. Liquidity
    # falling from its own peak is the clearest early signature of an exit.
    liqs = [_get(r, "liquidity_usd") for r in history]
    pxs = [_get(r, "price_usd") for r in history]
    peak_liq, peak_px = max(liqs), max(pxs)
    out["liq_drawdown"] = _safe_div(peak_liq - liq_last, peak_liq)
    out["price_drawdown"] = _safe_div(peak_px - px_last, peak_px)

    br_first = _safe_div(_get(first, "buys_m5"), _get(first, "buys_m5") + _get(first, "sells_m5"), 0.5)
    br_last = _safe_div(_get(last, "buys_m5"), _get(last, "buys_m5") + _get(last, "sells_m5"), 0.5)
    out["buy_ratio_trend"] = br_last - br_first

    a0, a1 = _get(first, "age_min"), _get(last, "age_min")
    out["obs_span_min"] = max(0.0, a1 - a0)
    return out


def build_features(
    history: Sequence[Mapping[str, Any]],
    index: int = -1,
    *,
    extra: Mapping[str, float] | None = None,
) -> FeatureVector:
    """Build the feature vector for ``history[index]``.

    Only ``history[:index + 1]`` is used. Snapshots after the decision point are
    sliced away before any computation, so lookahead is impossible by
    construction rather than by discipline.
    """
    if not history:
        raise ValueError("cannot build features from empty history")
    if index < 0:
        index = len(history) + index
    if not 0 <= index < len(history):
        raise IndexError(f"index {index} out of range for {len(history)} snapshots")

    # The decisive line: everything after the decision point is discarded.
    visible = list(history[: index + 1])
    row = visible[-1]
    age_min = _get(row, "age_min")

    values: dict[str, float] = {}
    values.update(_level_features(row, age_min))
    values.update(_flow_features(row))
    values.update(_window_deltas(row))
    values.update(_trajectory_features(visible))
    if extra:
        values.update({k: float(v) for k, v in extra.items() if math.isfinite(float(v))})

    # Final guard: no NaN or inf may reach a model.
    values = {k: (v if isinstance(v, float) and math.isfinite(v) else 0.0) for k, v in values.items()}

    return FeatureVector(
        pool=str(row["pool"]) if "pool" in row.keys() else "",
        observed_at=str(row["observed_at"]) if "observed_at" in row.keys() else "",
        age_min=age_min,
        values=values,
    )


def _canonical_names() -> list[str]:
    """Stable, sorted feature-name list derived from a synthetic row.

    Deriving the schema from the code (rather than hard-coding it) guarantees
    the training and inference feature order can never silently diverge.
    """
    blank: dict[str, Any] = {"pool": "", "observed_at": "", "age_min": 1.0}
    for key in ("price_usd", "price_native", "fdv_usd", "liquidity_usd"):
        blank[key] = 1.0
    for tf in TFS:
        for stem in ("buys", "sells", "buyers", "sellers", "vol", "chg"):
            blank[f"{stem}_{tf}"] = 1.0
    fv = build_features([blank, blank])
    return sorted(fv.values)


FEATURE_NAMES: list[str] = _canonical_names()
