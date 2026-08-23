"""Trade signals — the system's actual product.

Execution on Solana is a solved, outsourced problem: sniper front-ends already
handle routing, priority fees, Jito bundles and signing. What they cannot supply
is the decision. This module packages the system's output as an explicit,
machine-readable instruction answering the only three questions that matter:

**What to buy** — a specific mint that cleared safety screening, wash-trading
checks and the survival model, with the reasons attached so the decision is
auditable rather than a black box.

**When to buy** — signals carry an explicit expiry, because the measurements
that justify them decay fast. Achievable multiples collapse from 318x at the
launch candle to 7.3x by minute two and 4.6x by minute five; a signal acted on
ten minutes late is not the same trade, it is a different and worse one. A
stale signal is therefore *invalid*, not merely less good.

**When to sell** — a complete exit plan fixed at entry: profit target, stop,
trailing stop, time limit, and the two path-dependent rules a fixed stop cannot
express. Deciding the exit in advance matters because a third of positions stop
trading entirely, and improvising an exit under those conditions is how a −45%
stop becomes a −100% outcome.

Signals are emitted to JSONL so a bot can consume them, and rendered to the
console so a human can read them. Nothing here executes a trade.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger(__name__)


@dataclass
class ExitPlan:
    """The complete sell-side instruction, fixed at entry."""

    take_profit_pct: float
    stop_loss_pct: float
    trailing_stop_pct: float
    max_hold_minutes: int
    #: Arm the trailing stop only once the position is this far ahead, so
    #: ordinary entry noise cannot knock us out of a working trade.
    trailing_arms_at_pct: float = 0.30
    #: Exit on a robust-outlier negative return of this many sigma. Uses median
    #: absolute deviation, because one dump inflates a standard deviation ~49x
    #: and MAD ~1.3x — a sigma threshold widens exactly when it must stay tight.
    dump_exit_sigma: float = 4.0
    #: Exit if pool liquidity falls this far from its observed peak. This fires
    #: before price does: a draining pool means the exit is closing while the
    #: chart still looks healthy.
    liquidity_exit_pct: float = 0.35

    def describe(self, entry_price: float) -> dict[str, Any]:
        """Concrete price levels a bot can act on directly."""
        return {
            "take_profit_price": entry_price * (1 + self.take_profit_pct),
            "stop_loss_price": entry_price * (1 - self.stop_loss_pct),
            "trailing_stop_pct": self.trailing_stop_pct,
            "trailing_arms_above": entry_price * (1 + self.trailing_arms_at_pct),
            "max_hold_minutes": self.max_hold_minutes,
            "dump_exit_sigma": self.dump_exit_sigma,
            "liquidity_exit_pct": self.liquidity_exit_pct,
        }


@dataclass
class Signal:
    """One actionable trade instruction."""

    mint: str
    symbol: str
    pool: str
    dex: str

    # --- what and why -----------------------------------------------------
    score: float                       # calibrated P(win)
    survival_probability: float        # P(still tradeable at horizon)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # --- when -------------------------------------------------------------
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_for_seconds: int = 90
    token_age_seconds: float = 0.0

    # --- how much ---------------------------------------------------------
    suggested_usd: float = 0.0
    max_usd: float = 0.0               # beyond this, price impact eats the trade
    liquidity_usd: float = 0.0
    expected_slippage_pct: float = 0.0

    # --- how to get out ---------------------------------------------------
    exit_plan: ExitPlan | None = None
    reference_price: float = 0.0

    # --- provenance -------------------------------------------------------
    safety_risk_score: float = 0.0
    wash_score: float = 0.0
    dev_wallet: str = ""
    dev_prior_launches: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> datetime:
        return self.created_at + timedelta(seconds=self.valid_for_seconds)

    def is_valid(self, now: datetime | None = None) -> bool:
        """Signals expire because the edge they encode expires."""
        return (now or datetime.now(timezone.utc)) < self.expires_at

    def seconds_remaining(self, now: datetime | None = None) -> float:
        return max(0.0, (self.expires_at - (now or datetime.now(timezone.utc))).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat()
        if self.exit_plan and self.reference_price > 0:
            data["exit_levels"] = self.exit_plan.describe(self.reference_price)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"BUY {self.symbol or self.mint[:10]}  ({self.mint})",
            f"  score {self.score:.1%}  survival {self.survival_probability:.1%}  "
            f"age {self.token_age_seconds:.0f}s  valid {self.seconds_remaining():.0f}s more",
            f"  size ${self.suggested_usd:,.0f} (max ${self.max_usd:,.0f})  "
            f"liquidity ${self.liquidity_usd:,.0f}  est. slippage {self.expected_slippage_pct:.1%}",
        ]
        if self.exit_plan and self.reference_price > 0:
            levels = self.exit_plan.describe(self.reference_price)
            lines.append(
                f"  EXIT: target {levels['take_profit_price']:.3e} "
                f"(+{self.exit_plan.take_profit_pct:.0%}) | "
                f"stop {levels['stop_loss_price']:.3e} (−{self.exit_plan.stop_loss_pct:.0%}) | "
                f"trail {self.exit_plan.trailing_stop_pct:.0%} once +{self.exit_plan.trailing_arms_at_pct:.0%} | "
                f"time-out {self.exit_plan.max_hold_minutes}m"
            )
            lines.append(
                f"        also exit on a {self.exit_plan.dump_exit_sigma:.0f}σ dump "
                f"or liquidity −{self.exit_plan.liquidity_exit_pct:.0%} from peak"
            )
        if self.reasons:
            lines.append("  why: " + "; ".join(self.reasons[:4]))
        if self.warnings:
            lines.append("  caution: " + "; ".join(self.warnings[:3]))
        return "\n".join(lines)


class SignalEmitter:
    """Writes signals to a JSONL feed and optionally to the console.

    JSONL is chosen so a separate execution process can tail the file without
    coupling to this one. Keeping decision and execution in different processes
    means a crash in the trading integration cannot take down data collection,
    and the signal history remains a complete audit trail either way.
    """

    def __init__(
        self,
        path: str | Path = "data/signals.jsonl",
        *,
        echo: bool = True,
        min_score: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.min_score = min_score
        self.emitted = 0
        self.suppressed = 0

    def emit(self, signal: Signal) -> bool:
        if signal.score < self.min_score:
            self.suppressed += 1
            return False
        with open(self.path, "a") as fh:
            fh.write(signal.to_json() + "\n")
        self.emitted += 1
        if self.echo:
            log.info("SIGNAL\n%s", signal.describe())
        return True

    def recent(self, limit: int = 20, valid_only: bool = False) -> list[dict[str, Any]]:
        """Read back recent signals, newest first."""
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        rows.reverse()
        if valid_only:
            now = datetime.now(timezone.utc)
            rows = [
                r for r in rows
                if _parse(r.get("expires_at")) and _parse(r["expires_at"]) > now
            ]
        return rows[:limit]


def build_exit_plan(
    *,
    take_profit: float = 2.00,
    stop_loss: float = 0.45,
    trailing_stop: float = 0.35,
    max_hold_min: int = 60,
) -> ExitPlan:
    """Default exit plan.

    The wide profit target is a measured choice rather than a preference:
    across 72 barrier geometries on the same decision points, expectancy
    improved monotonically as the target widened, while the total-loss rate
    stayed fixed at 33.1% regardless of stop placement. When a third of trades
    lose everything, the rare large winner is the only thing funding them.
    """
    return ExitPlan(
        take_profit_pct=take_profit,
        stop_loss_pct=stop_loss,
        trailing_stop_pct=trailing_stop,
        max_hold_minutes=max_hold_min,
    )


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
