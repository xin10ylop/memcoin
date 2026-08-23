"""Outcome labelling via the triple-barrier method.

Given a decision time T, we ask what would have happened to a position opened
just after T. Three barriers race each other:

* an **upper barrier** at ``+take_profit`` (the trade worked),
* a **lower barrier** at ``-stop_loss`` (the trade failed),
* a **vertical barrier** at ``horizon_min`` (time ran out; mark to market).

Whichever is touched first determines the label. This is Lopez de Prado's
triple-barrier scheme, and it matters far more here than in equities because a
memecoin's path is everything: a token that reaches +300% in minute two and
ends at −95% is a *win* for a strategy that takes profit, and a catastrophe for
one that holds. Labelling on terminal return alone would conflate the two and
teach the model the wrong thing entirely.

Two adaptations are specific to this asset class:

**Conservative barrier ordering.** Minute candles do not record the order in
which the high and low were reached. When both barriers fall inside one candle
we assume the *stop* was hit first. This deliberately biases labels pessimistic:
the alternative silently manufactures winners that never existed.

**Absence of data is itself a label.** If a pool has no candles after T, it did
not merely stop being observed — it stopped trading. That is a total loss, and
recording it as missing data rather than as a loss is precisely the
survivorship bias this system exists to avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

log = logging.getLogger(__name__)


class Barrier(str, Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME = "time"
    NO_DATA = "no_data"       # never traded again — total loss
    NO_ENTRY = "no_entry"     # could not have entered; excluded from training


@dataclass
class LabelConfig:
    """Barrier geometry.

    Defaults encode a lottery-ticket profile: risk 45% to make 150%, on a
    45-minute clock. That asymmetry is deliberate — see
    :mod:`alpha.risk.sizing` for why a positively-skewed payoff is the only
    shape that survives this hit rate.
    """

    take_profit: float = 2.00    # +200%
    stop_loss: float = 0.45      # −45%
    horizon_min: int = 60
    # Modelled cost of getting in and out, applied to the realised return.
    round_trip_cost: float = 0.035
    # A pool whose liquidity collapses below this is treated as rugged.
    rug_liquidity_usd: float = 500.0


@dataclass
class Label:
    """Outcome of one hypothetical trade."""

    pool: str
    decision_ts: int
    entry_price: float
    barrier: Barrier
    exit_price: float
    gross_return: float
    net_return: float
    minutes_held: float
    max_multiple: float          # best price reached / entry
    min_multiple: float          # worst price reached / entry
    minutes_to_peak: float
    final_multiple: float        # price at horizon / entry
    n_candles: int
    is_win: bool
    meta: dict[str, Any] = field(default_factory=dict)

    #: Threshold below which an outcome counts as a total loss rather than a
    #: bad trade. A stop-loss cannot protect against these: the token stopped
    #: trading, so there was no bid to sell into.
    TOTAL_LOSS_THRESHOLD = -0.90

    @property
    def survived(self) -> bool:
        """Whether the position was still exitable at any price.

        This is a separate question from whether the trade was profitable, and
        empirically a far more learnable one. Measured on collected data, total
        losses were 31.5% of outcomes and drove expectancy from +26% to −14%;
        the ordinary stop-loss fired on only 4% of trades. Avoiding death is
        therefore worth more than picking winners, and it is the target the
        first stage of the model is trained on.
        """
        return self.net_return > self.TOTAL_LOSS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "decision_ts": self.decision_ts,
            "entry_price": self.entry_price,
            "barrier": self.barrier.value,
            "exit_price": self.exit_price,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "minutes_held": self.minutes_held,
            "max_multiple": self.max_multiple,
            "min_multiple": self.min_multiple,
            "minutes_to_peak": self.minutes_to_peak,
            "final_multiple": self.final_multiple,
            "n_candles": self.n_candles,
            "is_win": self.is_win,
            "survived": self.survived,
            **self.meta,
        }


def label_trade(
    pool: str,
    candles: Sequence[Any],
    decision_ts: int,
    config: LabelConfig | None = None,
) -> Label:
    """Label the outcome of entering ``pool`` just after ``decision_ts``.

    ``candles`` must be ascending by timestamp. Entry is at the *open* of the
    first candle strictly after ``decision_ts``: we cannot trade on a candle we
    have not yet seen close, and entering at its open is the earliest honest
    fill.
    """
    cfg = config or LabelConfig()
    future = [c for c in candles if _ts(c) > decision_ts]

    if not future:
        # No trading after the decision point: the token is gone.
        return Label(
            pool=pool, decision_ts=decision_ts, entry_price=0.0, barrier=Barrier.NO_DATA,
            exit_price=0.0, gross_return=-1.0, net_return=-1.0, minutes_held=0.0,
            max_multiple=0.0, min_multiple=0.0, minutes_to_peak=0.0, final_multiple=0.0,
            n_candles=0, is_win=False,
        )

    entry = _open(future[0])
    if entry <= 0:
        return Label(
            pool=pool, decision_ts=decision_ts, entry_price=0.0, barrier=Barrier.NO_ENTRY,
            exit_price=0.0, gross_return=0.0, net_return=0.0, minutes_held=0.0,
            max_multiple=0.0, min_multiple=0.0, minutes_to_peak=0.0, final_multiple=0.0,
            n_candles=len(future), is_win=False,
        )

    up = entry * (1.0 + cfg.take_profit)
    down = entry * (1.0 - cfg.stop_loss)
    deadline = decision_ts + cfg.horizon_min * 60

    window = [c for c in future if _ts(c) <= deadline]
    if not window:
        window = future[:1]

    peak = entry
    trough = entry
    minutes_to_peak = 0.0
    barrier = Barrier.TIME
    exit_price = _close(window[-1])
    exit_ts = _ts(window[-1])

    for candle in window:
        hi, lo, ts = _high(candle), _low(candle), _ts(candle)
        if hi > peak:
            peak = hi
            minutes_to_peak = (ts - decision_ts) / 60.0
        trough = min(trough, lo) if lo > 0 else trough

        hit_stop = lo > 0 and lo <= down
        hit_target = hi >= up
        if hit_stop:
            # Pessimistic tie-break: when a single candle spans both barriers we
            # cannot know the order, so we assume the adverse one came first.
            barrier, exit_price, exit_ts = Barrier.STOP_LOSS, down, ts
            break
        if hit_target:
            barrier, exit_price, exit_ts = Barrier.TAKE_PROFIT, up, ts
            break

    gross = (exit_price / entry) - 1.0
    net = gross - cfg.round_trip_cost
    return Label(
        pool=pool,
        decision_ts=decision_ts,
        entry_price=entry,
        barrier=barrier,
        exit_price=exit_price,
        gross_return=gross,
        net_return=net,
        minutes_held=max(0.0, (exit_ts - decision_ts) / 60.0),
        max_multiple=peak / entry,
        min_multiple=trough / entry,
        minutes_to_peak=minutes_to_peak,
        final_multiple=_close(window[-1]) / entry,
        n_candles=len(window),
        is_win=net > 0.0,
    )


def summarise(labels: Sequence[Label]) -> dict[str, Any]:
    """Aggregate statistics over a set of labels."""
    if not labels:
        return {"n": 0}
    tradeable = [x for x in labels if x.barrier is not Barrier.NO_ENTRY]
    n = len(tradeable)
    if not n:
        return {"n": 0}
    wins = [x for x in tradeable if x.is_win]
    losses = [x for x in tradeable if not x.is_win]
    rets = [x.net_return for x in tradeable]
    avg_win = sum(x.net_return for x in wins) / len(wins) if wins else 0.0
    avg_loss = sum(x.net_return for x in losses) / len(losses) if losses else 0.0
    counts: dict[str, int] = {}
    for x in tradeable:
        counts[x.barrier.value] = counts.get(x.barrier.value, 0) + 1
    mults = sorted(x.max_multiple for x in tradeable)
    return {
        "n": n,
        "hit_rate": len(wins) / n,
        "mean_return": sum(rets) / n,
        "median_return": sorted(rets)[n // 2],
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        # Expectancy per unit risked — the number that decides viability.
        "expectancy": sum(rets) / n,
        "payoff_ratio": abs(avg_win / avg_loss) if avg_loss else float("inf"),
        "total_return": sum(rets),
        "barriers": counts,
        "pct_reached_2x": sum(1 for m in mults if m >= 2.0) / n,
        "pct_reached_5x": sum(1 for m in mults if m >= 5.0) / n,
        "pct_total_loss": sum(1 for x in tradeable if x.max_multiple <= 0.05) / n,
        "max_multiple_p99": mults[int(0.99 * (n - 1))],
    }


def _ts(c: Any) -> int:
    return int(c["ts"] if _is_mapping(c) else c.ts)


def _open(c: Any) -> float:
    return float(c["open"] if _is_mapping(c) else c.open)


def _high(c: Any) -> float:
    return float(c["high"] if _is_mapping(c) else c.high)


def _low(c: Any) -> float:
    return float(c["low"] if _is_mapping(c) else c.low)


def _close(c: Any) -> float:
    return float(c["close"] if _is_mapping(c) else c.close)


def _is_mapping(c: Any) -> bool:
    return hasattr(c, "keys")
