"""Event-driven backtester over the collected panel.

The simulation replays history in strict chronological order across all tokens
at once, rather than evaluating each token in isolation. That matters because
the binding constraints in this strategy are portfolio-level: capital, the cap
on concurrent positions, and the circuit breakers. A per-token backtest silently
assumes unlimited capital and would report returns that could never have been
achieved with one account.

Sequencing within each minute is deliberate and mirrors reality:

1. **Mark** open positions to the current candle.
2. **Exit** any position whose rule has triggered — before considering new
   entries, because exits free the capital and position slots that entries need,
   and doing it the other way round would let the book hold more risk than the
   limits allow.
3. **Enter** new positions from the candidates whose decision timestamp falls in
   this minute.

Fills use the *next* candle's open, never the current close. A signal computed
from a candle cannot be traded at that same candle's price — that is the classic
one-bar lookahead, and it is worth several percent per trade in a market this
volatile.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Sequence

from alpha.execution.costs import CostModel, FillSide
from alpha.risk.portfolio import Portfolio, PortfolioConfig, Position, RiskState
from alpha.risk.sizing import PositionSizer, SizingConfig

log = logging.getLogger(__name__)

# A scorer maps a feature dict to a probability of the trade winning.
Scorer = Callable[[dict[str, float]], float]


@dataclass
class Candidate:
    """A potential entry, produced by the dataset builder."""

    pool: str
    decision_ts: int
    features: dict[str, float]
    liquidity_usd: float
    price_usd: float
    dex: str = ""
    symbol: str = ""
    mint: str = ""


@dataclass
class BacktestConfig:
    min_score: float = 0.30
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    costs: CostModel = field(default_factory=CostModel)
    take_profit: float = 1.50
    stop_loss: float = 0.45
    trailing_stop: float = 0.35
    max_hold_min: int = 45
    # Positions still open at the end are liquidated at the last price so the
    # reported return reflects a fully closed book.
    liquidate_at_end: bool = True


@dataclass
class BacktestResult:
    stats: dict[str, Any]
    equity_curve: list[tuple[int, float]]
    trades: list[dict[str, Any]]
    rejected: dict[str, int]
    n_candidates: int
    n_entered: int

    def summary(self) -> str:
        s = self.stats
        pf = s.get("profit_factor")
        return (
            f"equity ${s['equity']:,.2f} ({s['total_return_pct']:+.2%})  "
            f"trades={s['trades']} hit={s['hit_rate']:.1%}  "
            f"pf={pf if pf is None else round(pf, 2)}  "
            f"maxDD={s['drawdown_pct']:.1%}  "
            f"entered {self.n_entered}/{self.n_candidates} candidates"
        )


class Backtester:
    """Replays the panel chronologically against a scoring function."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.cfg = config or BacktestConfig()
        self.costs = self.cfg.costs
        self.sizer = PositionSizer(self.cfg.sizing, self.costs)

    def run(
        self,
        candidates: Sequence[Candidate],
        candles_by_pool: dict[str, Sequence[Any]],
        scorer: Scorer,
    ) -> BacktestResult:
        portfolio = Portfolio(self.cfg.portfolio)
        rejected: dict[str, int] = defaultdict(int)
        equity_curve: list[tuple[int, float]] = []
        trades: list[dict[str, Any]] = []
        n_entered = 0

        # Index candles by minute for O(1) lookup during the replay.
        price_index: dict[str, dict[int, dict[str, float]]] = {}
        for pool, candles in candles_by_pool.items():
            price_index[pool] = {
                int(_g(c, "ts")): {
                    "open": _g(c, "open"), "high": _g(c, "high"),
                    "low": _g(c, "low"), "close": _g(c, "close"),
                }
                for c in candles
            }

        entries_by_ts: dict[int, list[Candidate]] = defaultdict(list)
        for cand in candidates:
            entries_by_ts[_floor_min(cand.decision_ts)].append(cand)

        all_ts = sorted(
            {t for idx in price_index.values() for t in idx} | set(entries_by_ts)
        )
        if not all_ts:
            return BacktestResult(portfolio.stats(), [], [], dict(rejected), len(candidates), 0)

        entry_liquidity: dict[str, float] = {}

        for ts in all_ts:
            now = datetime.fromtimestamp(ts, tz=timezone.utc)
            prices = self._prices_at(price_index, ts, portfolio)

            # --- 1. mark to market -------------------------------------------
            for pool, pos in portfolio.positions.items():
                if pool in prices:
                    pos.mark(prices[pool])

            # --- 2. exits before entries -------------------------------------
            for pool in list(portfolio.positions):
                pos = portfolio.positions[pool]
                price = prices.get(pool)
                if price is None or price <= 0:
                    # No trading in this pool: it has gone dark. Only force a
                    # writedown once the position has aged out, so a brief gap
                    # in candles is not mistaken for a rug.
                    if pos.age_minutes(now) >= pos.max_hold_min * 2:
                        trade = portfolio.close(pool, 0.0, "went_dark", now)
                        if trade:
                            trades.append(_trade_row(trade, 0.0))
                    continue
                reason = pos.exit_signal(price, now)
                if reason:
                    liq = entry_liquidity.get(pool, pos.entry_liquidity_usd)
                    notional = pos.tokens * price
                    fill = self.costs.simulate(
                        FillSide.SELL, notional, price, liq, dex=pos.dex, apply_latency=True
                    )
                    proceeds = fill.filled_usd if fill.ok else 0.0
                    trade = portfolio.close(pool, proceeds, reason, now)
                    if trade:
                        trades.append(_trade_row(trade, price))

            # --- 3. entries ----------------------------------------------------
            for cand in entries_by_ts.get(ts, []):
                score = scorer(cand.features)
                if score < self.cfg.min_score:
                    rejected["low_score"] += 1
                    continue

                # Fill at the NEXT candle's open — never the current close.
                nxt = self._next_open(price_index.get(cand.pool, {}), ts)
                if nxt is None or nxt <= 0:
                    rejected["no_next_price"] += 1
                    continue

                equity = portfolio.equity(prices)
                decision = self.sizer.size(
                    equity_usd=equity, win_prob=score,
                    liquidity_usd=cand.liquidity_usd, dex=cand.dex,
                )
                if not decision.approved:
                    rejected[f"size:{decision.reason.split('(')[0].strip()[:28]}"] += 1
                    continue

                usd = decision.usd * portfolio.size_multiplier(prices)
                allowed, why = portfolio.can_open(cand.pool, usd, cand.dex, prices)
                if not allowed:
                    rejected[f"portfolio:{why[:34]}"] += 1
                    continue

                fill = self.costs.simulate(
                    FillSide.BUY, usd, nxt, cand.liquidity_usd, dex=cand.dex, first_buy=True
                )
                if not fill.ok or fill.tokens <= 0:
                    rejected[f"fill:{fill.reason[:30]}"] += 1
                    continue

                total_cost = usd + fill.network_fee_usd
                if total_cost > portfolio.cash_usd:
                    rejected["insufficient_cash"] += 1
                    continue

                pos = Position(
                    pool=cand.pool, mint=cand.mint, symbol=cand.symbol or cand.pool[:8],
                    entry_price=fill.effective_price, tokens=fill.tokens, cost_usd=total_cost,
                    opened_at=now, dex=cand.dex, entry_liquidity_usd=cand.liquidity_usd,
                    take_profit=self.cfg.take_profit, stop_loss=self.cfg.stop_loss,
                    trailing_stop=self.cfg.trailing_stop, max_hold_min=self.cfg.max_hold_min,
                    meta={"score": score},
                )
                portfolio.open(pos, total_cost)
                entry_liquidity[cand.pool] = cand.liquidity_usd
                n_entered += 1

            if portfolio.risk_state(prices) is RiskState.HALTED:
                rejected["halted"] += 1

            equity_curve.append((ts, portfolio.equity(prices)))

        # --- final liquidation ------------------------------------------------
        if self.cfg.liquidate_at_end and portfolio.positions:
            last_ts = all_ts[-1]
            prices = self._prices_at(price_index, last_ts, portfolio)
            for pool in list(portfolio.positions):
                pos = portfolio.positions[pool]
                price = prices.get(pool, 0.0)
                notional = pos.tokens * price
                proceeds = 0.0
                if price > 0:
                    fill = self.costs.simulate(
                        FillSide.SELL, notional, price,
                        entry_liquidity.get(pool, pos.entry_liquidity_usd), dex=pos.dex,
                    )
                    proceeds = fill.filled_usd if fill.ok else 0.0
                trade = portfolio.close(pool, proceeds, "end_of_backtest")
                if trade:
                    trades.append(_trade_row(trade, price))

        return BacktestResult(
            stats=portfolio.stats(),
            equity_curve=equity_curve,
            trades=trades,
            rejected=dict(rejected),
            n_candidates=len(candidates),
            n_entered=n_entered,
        )

    @staticmethod
    def _prices_at(
        index: dict[str, dict[int, dict[str, float]]], ts: int, portfolio: Portfolio
    ) -> dict[str, float]:
        """Close prices at ``ts`` for held pools plus any pool with a candle."""
        out: dict[str, float] = {}
        for pool, by_ts in index.items():
            bar = by_ts.get(ts)
            if bar:
                out[pool] = bar["close"]
        return out

    @staticmethod
    def _next_open(by_ts: dict[int, dict[str, float]], ts: int) -> float | None:
        """Open of the first candle strictly after ``ts``."""
        later = [t for t in by_ts if t > ts]
        if not later:
            return None
        return by_ts[min(later)]["open"]


def _trade_row(trade: Any, exit_price: float) -> dict[str, Any]:
    return {
        "pool": trade.pool,
        "symbol": trade.symbol,
        "opened_at": trade.opened_at.isoformat(),
        "closed_at": trade.closed_at.isoformat(),
        "held_min": round((trade.closed_at - trade.opened_at).total_seconds() / 60.0, 2),
        "cost_usd": round(trade.cost_usd, 2),
        "proceeds_usd": round(trade.proceeds_usd, 2),
        "pnl_usd": round(trade.pnl_usd, 2),
        "return_pct": round(trade.return_pct, 4),
        "reason": trade.reason,
        "exit_price": exit_price,
    }


def _g(c: Any, key: str) -> float:
    if hasattr(c, "keys"):
        return float(c[key])
    return float(getattr(c, key))


def _floor_min(ts: int) -> int:
    return (int(ts) // 60) * 60
