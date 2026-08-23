"""Portfolio state, exposure limits, and circuit breakers.

The sizing layer decides how large one position may be. This layer decides
whether the position may be opened at all, given everything else the book is
already carrying.

Circuit breakers exist because the failure mode of an automated strategy on a
non-stationary market is not a single bad trade — it is continuing to trade a
regime the model was not built for. The market for new launches has hot and
cold phases, and an edge fitted in a hot phase can invert in a cold one. The
breakers below are deliberately blunt: on a bad enough day the correct action is
to stop, not to size down and keep going.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RiskState(str, Enum):
    NORMAL = "normal"
    THROTTLED = "throttled"   # reduced size after losses
    HALTED = "halted"         # no new positions


@dataclass
class Position:
    """An open position."""

    pool: str
    mint: str
    symbol: str
    entry_price: float
    tokens: float
    cost_usd: float             # what we actually paid, including costs
    opened_at: datetime
    dex: str = ""
    entry_liquidity_usd: float = 0.0
    peak_price: float = 0.0
    take_profit: float = 1.50
    stop_loss: float = 0.45
    trailing_stop: float = 0.35
    max_hold_min: int = 45
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.peak_price = max(self.peak_price, self.entry_price)

    def mark(self, price: float) -> float:
        """Update the high-water mark and return current value in USD."""
        if price > self.peak_price:
            self.peak_price = price
        return self.tokens * price

    def unrealised_pct(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (price / self.entry_price) - 1.0

    def drawdown_from_peak(self, price: float) -> float:
        if self.peak_price <= 0:
            return 0.0
        return 1.0 - (price / self.peak_price)

    def age_minutes(self, now: datetime | None = None) -> float:
        return ((now or _utcnow()) - self.opened_at).total_seconds() / 60.0

    def exit_signal(self, price: float, now: datetime | None = None) -> str | None:
        """Return the name of the triggered exit rule, or ``None`` to hold.

        Order matters: stops are evaluated before targets so that a violent
        candle which breached both is resolved conservatively.
        """
        pnl = self.unrealised_pct(price)
        if pnl <= -self.stop_loss:
            return "stop_loss"
        # A trailing stop only arms once the position is meaningfully ahead,
        # so ordinary entry noise cannot knock us out.
        if self.unrealised_pct(self.peak_price) >= 0.30 and self.drawdown_from_peak(price) >= self.trailing_stop:
            return "trailing_stop"
        if pnl >= self.take_profit:
            return "take_profit"
        if self.age_minutes(now) >= self.max_hold_min:
            return "max_hold"
        return None


@dataclass
class ClosedTrade:
    pool: str
    symbol: str
    opened_at: datetime
    closed_at: datetime
    cost_usd: float
    proceeds_usd: float
    reason: str

    @property
    def pnl_usd(self) -> float:
        return self.proceeds_usd - self.cost_usd

    @property
    def return_pct(self) -> float:
        return (self.proceeds_usd / self.cost_usd - 1.0) if self.cost_usd > 0 else 0.0


@dataclass
class PortfolioConfig:
    starting_equity_usd: float = 10_000.0
    max_open_positions: int = 8
    # Total capital that may be deployed at once. The remainder is dry powder:
    # being fully invested in an asset class where positions can gap to zero
    # removes the ability to act on anything better.
    max_total_exposure_pct: float = 0.15
    max_positions_per_dex: int = 6
    # Circuit breakers.
    daily_loss_halt_pct: float = 0.08      # stop for the day after −8%
    drawdown_halt_pct: float = 0.20        # stop entirely after −20% from peak
    consecutive_losses_throttle: int = 6   # halve size after this many losses
    throttle_size_multiplier: float = 0.5
    # Refuse to re-enter a token we already traded recently.
    reentry_cooldown_min: int = 120


class Portfolio:
    """Tracks positions and enforces book-level risk limits."""

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.cfg = config or PortfolioConfig()
        self.cash_usd = self.cfg.starting_equity_usd
        self.positions: dict[str, Position] = {}
        self.closed: list[ClosedTrade] = []
        self.peak_equity = self.cfg.starting_equity_usd
        self.consecutive_losses = 0
        self._day_start_equity = self.cfg.starting_equity_usd
        self._day = _utcnow().date()
        self._recent_exits: dict[str, datetime] = {}
        self.halted_reason: str | None = None

    # ------------------------------------------------------------- accounting

    def equity(self, prices: dict[str, float] | None = None) -> float:
        """Cash plus marked-to-market value of open positions."""
        prices = prices or {}
        holdings = sum(
            pos.tokens * prices.get(pool, pos.entry_price) for pool, pos in self.positions.items()
        )
        return self.cash_usd + holdings

    def exposure_usd(self) -> float:
        return sum(p.cost_usd for p in self.positions.values())

    def exposure_pct(self, equity: float | None = None) -> float:
        eq = equity if equity is not None else self.equity()
        return self.exposure_usd() / eq if eq > 0 else 1.0

    # ------------------------------------------------------------------ state

    def risk_state(self, prices: dict[str, float] | None = None) -> RiskState:
        equity = self.equity(prices)
        self._roll_day(equity)
        self.peak_equity = max(self.peak_equity, equity)

        if self.halted_reason:
            return RiskState.HALTED

        drawdown = 1.0 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.cfg.drawdown_halt_pct:
            self.halt(f"drawdown {drawdown:.1%} exceeded limit {self.cfg.drawdown_halt_pct:.1%}")
            return RiskState.HALTED

        day_loss = 1.0 - equity / self._day_start_equity if self._day_start_equity > 0 else 0.0
        if day_loss >= self.cfg.daily_loss_halt_pct:
            self.halt(f"daily loss {day_loss:.1%} exceeded limit {self.cfg.daily_loss_halt_pct:.1%}")
            return RiskState.HALTED

        if self.consecutive_losses >= self.cfg.consecutive_losses_throttle:
            return RiskState.THROTTLED
        return RiskState.NORMAL

    def halt(self, reason: str) -> None:
        if not self.halted_reason:
            log.warning("TRADING HALTED: %s", reason)
        self.halted_reason = reason

    def resume(self) -> None:
        """Clear a halt. Intended for explicit operator action or a new day."""
        self.halted_reason = None
        self.peak_equity = self.equity()

    def size_multiplier(self, prices: dict[str, float] | None = None) -> float:
        return self.cfg.throttle_size_multiplier if self.risk_state(prices) is RiskState.THROTTLED else 1.0

    # ------------------------------------------------------------- admission

    def can_open(self, pool: str, usd: float, dex: str = "", prices: dict[str, float] | None = None) -> tuple[bool, str]:
        """Whether a new position may be opened, and why not if it may not."""
        state = self.risk_state(prices)
        if state is RiskState.HALTED:
            return False, f"halted: {self.halted_reason}"
        if pool in self.positions:
            return False, "already holding this pool"
        if len(self.positions) >= self.cfg.max_open_positions:
            return False, f"at max open positions ({self.cfg.max_open_positions})"
        if usd > self.cash_usd:
            return False, f"insufficient cash (${self.cash_usd:,.2f} < ${usd:,.2f})"

        equity = self.equity(prices)
        if (self.exposure_usd() + usd) / max(equity, 1e-9) > self.cfg.max_total_exposure_pct:
            return False, f"would exceed max exposure {self.cfg.max_total_exposure_pct:.0%}"

        if dex:
            same = sum(1 for p in self.positions.values() if p.dex == dex)
            if same >= self.cfg.max_positions_per_dex:
                return False, f"at max positions for venue {dex}"

        last_exit = self._recent_exits.get(pool)
        if last_exit and _utcnow() - last_exit < timedelta(minutes=self.cfg.reentry_cooldown_min):
            return False, "within re-entry cooldown"
        return True, "ok"

    # ---------------------------------------------------------------- actions

    def open(self, position: Position, cash_spent: float) -> None:
        self.cash_usd -= cash_spent
        self.positions[position.pool] = position

    def close(self, pool: str, proceeds_usd: float, reason: str, now: datetime | None = None) -> ClosedTrade | None:
        pos = self.positions.pop(pool, None)
        if pos is None:
            return None
        self.cash_usd += proceeds_usd
        trade = ClosedTrade(
            pool=pool, symbol=pos.symbol, opened_at=pos.opened_at, closed_at=now or _utcnow(),
            cost_usd=pos.cost_usd, proceeds_usd=proceeds_usd, reason=reason,
        )
        self.closed.append(trade)
        self._recent_exits[pool] = trade.closed_at
        self.consecutive_losses = self.consecutive_losses + 1 if trade.pnl_usd <= 0 else 0
        return trade

    # ------------------------------------------------------------- reporting

    def stats(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        equity = self.equity(prices)
        wins = [t for t in self.closed if t.pnl_usd > 0]
        losses = [t for t in self.closed if t.pnl_usd <= 0]
        gross_win = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        return {
            "equity": round(equity, 2),
            "cash": round(self.cash_usd, 2),
            "open_positions": len(self.positions),
            "exposure_usd": round(self.exposure_usd(), 2),
            "exposure_pct": round(self.exposure_pct(equity), 4),
            "total_return_pct": round(equity / self.cfg.starting_equity_usd - 1.0, 4),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(1.0 - equity / self.peak_equity, 4) if self.peak_equity > 0 else 0.0,
            "trades": len(self.closed),
            "wins": len(wins),
            "losses": len(losses),
            "hit_rate": round(len(wins) / len(self.closed), 4) if self.closed else 0.0,
            "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
            "avg_win_usd": round(gross_win / len(wins), 2) if wins else 0.0,
            "avg_loss_usd": round(-gross_loss / len(losses), 2) if losses else 0.0,
            "consecutive_losses": self.consecutive_losses,
            "state": self.risk_state(prices).value,
            "halted_reason": self.halted_reason,
        }

    def _roll_day(self, equity: float) -> None:
        today = _utcnow().date()
        if today != self._day:
            self._day = today
            self._day_start_equity = equity
            # A daily-loss halt is a pause, not a permanent stop.
            if self.halted_reason and "daily loss" in self.halted_reason:
                self.halted_reason = None
