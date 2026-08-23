"""The live trading loop.

This is the component that behaves like a bot: it watches for new pools, decides
what to buy, and manages positions to an exit. It composes the layers rather
than reimplementing them, so the logic it applies is identical to what the
backtester measured.

Each cycle runs in a fixed order, and the order matters:

1. **Manage open positions first.** Exits free capital and position slots, and
   an exit that is late costs more than an entry that is late. In an asset class
   where a position can lose half its value in a minute, letting a new-entry
   scan delay a stop is indefensible.
2. **Discover and screen** new candidates.
3. **Score, size and enter** whatever survives.

Every decision is recorded — including rejections and why — so the live
decision distribution can be compared against the backtest's. A live system
whose rejection reasons look nothing like the simulation's is not running the
strategy that was tested, and that discrepancy is worth catching early.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from alpha.data.geckoterminal import GeckoTerminalClient, Pool
from alpha.data.store import Store, utcnow
from alpha.execution.broker import Broker, PaperBroker
from alpha.features.build import build_features
from alpha.risk.exits import DumpDetectorConfig, ExitMonitor
from alpha.signals import Signal, SignalEmitter, build_exit_plan
from alpha.risk.portfolio import Portfolio, PortfolioConfig, Position, RiskState
from alpha.risk.sizing import PositionSizer, SizingConfig
from alpha.safety.screen import SafetyScreener, Verdict

log = logging.getLogger(__name__)

Scorer = Callable[[dict[str, float]], float]


@dataclass
class TraderConfig:
    db_path: str = "data/alpha.db"
    # Minimum calibrated win probability to open a position. The default sits
    # above the ~26% break-even implied by realistic round-trip costs.
    min_score: float = 0.32
    # Age band for entry. Too young and there is no flow history to judge;
    # too old and the launch dynamics are over.
    min_age_min: float = 4.0
    max_age_min: float = 60.0
    min_liquidity_usd: float = 4_000.0
    max_liquidity_usd: float = 3_000_000.0
    min_buyers_m5: int = 8
    # Reject tokens whose recent volume looks manufactured.
    max_wash_score: float = 0.55
    cycle_seconds: float = 45.0
    requests_per_minute: float = 20.0
    # Screen at most this many new candidates per cycle: safety screening costs
    # two API calls per token and the budget is shared with position management.
    max_screens_per_cycle: int = 6
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    take_profit: float = 2.00
    stop_loss: float = 0.45
    trailing_stop: float = 0.35
    max_hold_min: int = 60
    dry_run_only: bool = True
    #: Path-dependent exit rules. A stop-loss cannot protect against a token
    #: that stops trading, so these watch for the dump and the liquidity
    #: withdrawal that precede it.
    exits: DumpDetectorConfig = field(default_factory=DumpDetectorConfig)
    use_dump_exits: bool = True
    #: Emit machine-readable signals alongside (or instead of) paper trades, so
    #: an external sniper front-end can execute them. Signals are the product;
    #: paper trading exists to measure whether they are any good.
    emit_signals: bool = True
    signals_path: str = "data/signals.jsonl"
    #: A signal is invalid once this elapses. Achievable multiples collapse
    #: within minutes, so a stale signal is a different trade, not a late one.
    signal_ttl_seconds: int = 90

    def max_open(self) -> int:
        return self.portfolio.max_open_positions


@dataclass
class TraderStats:
    cycles: int = 0
    screened: int = 0
    entered: int = 0
    exited: int = 0
    errors: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    started_at: datetime = field(default_factory=utcnow)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


class Trader:
    """Runs the strategy against live market data."""

    def __init__(
        self,
        config: TraderConfig | None = None,
        broker: Broker | None = None,
        scorer: Scorer | None = None,
        client: GeckoTerminalClient | None = None,
        screener: SafetyScreener | None = None,
    ) -> None:
        self.cfg = config or TraderConfig()
        self.broker = broker or PaperBroker()
        if self.broker.is_live and self.cfg.dry_run_only:
            raise RuntimeError(
                "A live broker was supplied while dry_run_only is set. Set "
                "dry_run_only=False explicitly to trade real funds."
            )
        # Without a trained model, score every candidate at zero so the loop runs
        # and reports what it *would* consider without ever opening a position.
        self.scorer = scorer or (lambda _f: 0.0)
        self.client = client or GeckoTerminalClient()
        self.client.http.bucket.rate = self.cfg.requests_per_minute / 60.0
        self.screener = screener or SafetyScreener()
        self.store = Store(self.cfg.db_path)
        self.portfolio = Portfolio(self.cfg.portfolio)
        self.sizer = PositionSizer(self.cfg.sizing)
        self.stats = TraderStats()
        self._stop = False
        self._screened_cache: set[str] = set()
        #: Per-position price/liquidity history, kept outside Position so the
        #: position object does not grow an unbounded buffer.
        self._monitors: dict[str, ExitMonitor] = {}
        self.emitter = (
            SignalEmitter(self.cfg.signals_path, echo=False, min_score=self.cfg.min_score)
            if self.cfg.emit_signals else None
        )

    def request_stop(self, *_: object) -> None:
        log.info("stop requested — will exit after this cycle")
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass

    # ------------------------------------------------------------------- loop

    def run(self, max_cycles: int | None = None) -> TraderStats:
        mode = "LIVE" if self.broker.is_live else "PAPER"
        log.info(
            "trader starting in %s mode: equity=$%.2f min_score=%.2f",
            mode, self.portfolio.cash_usd, self.cfg.min_score,
        )
        while not self._stop:
            start = time.monotonic()
            try:
                self.run_cycle()
            except Exception:
                self.stats.errors += 1
                log.exception("cycle failed — continuing")
            self.stats.cycles += 1

            if max_cycles and self.stats.cycles >= max_cycles:
                break
            remaining = self.cfg.cycle_seconds - (time.monotonic() - start)
            end = time.monotonic() + max(0.0, remaining)
            while not self._stop and time.monotonic() < end:
                time.sleep(min(0.5, end - time.monotonic()))
        log.info("trader stopped: %s", self.summary())
        return self.stats

    def run_cycle(self) -> None:
        # 1. Positions first: a late exit costs more than a late entry.
        self.manage_positions()
        # 2. Then look for new opportunities.
        self.scan_and_enter()

    # -------------------------------------------------------------- positions

    def manage_positions(self) -> None:
        if not self.portfolio.positions:
            return
        pools = list(self.portfolio.positions)
        observed = {p.address: p for p in self.client.pools_multi(pools)}
        now = datetime.now(timezone.utc)

        for pool_addr in pools:
            pos = self.portfolio.positions.get(pool_addr)
            if pos is None:
                continue
            live = observed.get(pool_addr)
            if live is None or live.price_usd <= 0:
                # Pool has gone dark. Only write off once it has aged out, so a
                # transient API gap is not mistaken for a rug.
                if pos.age_minutes(now) >= pos.max_hold_min * 2:
                    self.portfolio.close(pool_addr, 0.0, "went_dark", now)
                    self._monitors.pop(pool_addr, None)
                    self.stats.exited += 1
                    log.warning("position %s written off: pool went dark", pool_addr[:12])
                continue

            price = live.price_usd
            pos.mark(price)

            monitor = self._monitors.get(pool_addr)
            if monitor is not None:
                monitor.update(price, live.liquidity_usd)

            reason = pos.exit_signal(price, now)
            if reason is None and self.cfg.use_dump_exits and monitor is not None:
                # Path-dependent rules run only when the barrier rules are
                # silent: a dump large enough to breach the stop is already
                # handled, and this catches the ones that are not.
                signal = monitor.check()
                if signal.triggered:
                    reason = "dump_exit"
                    log.warning("dump exit on %s: %s", pos.symbol, signal.reason)
            if not reason:
                continue

            result = self.broker.sell(
                pool=pool_addr, mint=pos.mint, tokens=pos.tokens, price=price,
                liquidity_usd=live.liquidity_usd, dex=pos.dex,
            )
            proceeds = result.usd if result.ok else 0.0
            trade = self.portfolio.close(pool_addr, proceeds, reason, now)
            self._monitors.pop(pool_addr, None)
            self.stats.exited += 1
            if trade:
                log.info(
                    "EXIT %s %s: $%.2f -> $%.2f (%+.1f%%) after %.0fm [%s]",
                    trade.symbol, reason, trade.cost_usd, trade.proceeds_usd,
                    100 * trade.return_pct, (now - trade.opened_at).total_seconds() / 60, reason,
                )
                self.store.put_artifact("closed_trade", f"{pool_addr}:{trade.closed_at.isoformat()}", {
                    "pool": pool_addr, "symbol": trade.symbol, "reason": reason,
                    "cost": trade.cost_usd, "proceeds": trade.proceeds_usd,
                    "return_pct": trade.return_pct,
                })

    # ------------------------------------------------------------------ entry

    def scan_and_enter(self) -> None:
        state = self.portfolio.risk_state()
        if state is RiskState.HALTED:
            self.stats.reject(f"halted: {self.portfolio.halted_reason}")
            return
        if len(self.portfolio.positions) >= self.cfg.max_open():
            self.stats.reject("at max open positions")
            return

        candidates = self._prefilter(self.client.all_new_pools(max_pages=4))
        for pool in candidates[: self.cfg.max_screens_per_cycle]:
            try:
                self._consider(pool)
            except Exception:
                self.stats.errors += 1
                log.exception("failed considering %s", pool.address[:12])

    def _prefilter(self, pools: list[Pool]) -> list[Pool]:
        """Cheap filters applied before spending API calls on safety screening."""
        cfg = self.cfg
        out: list[Pool] = []
        for p in pools:
            if p.address in self._screened_cache or p.address in self.portfolio.positions:
                continue
            if not p.is_sol_quoted:
                self.stats.reject("not SOL-quoted")
                continue
            age = p.age_minutes
            if age != age or not (cfg.min_age_min <= age <= cfg.max_age_min):
                self.stats.reject("outside age band")
                continue
            if not (cfg.min_liquidity_usd <= p.liquidity_usd <= cfg.max_liquidity_usd):
                self.stats.reject("outside liquidity band")
                continue
            if p.tf("m5").buyers < cfg.min_buyers_m5:
                self.stats.reject("too few unique buyers")
                continue
            out.append(p)
        # Prefer the most active candidates when the screening budget binds.
        out.sort(key=lambda p: p.tf("m5").volume_usd, reverse=True)
        return out

    def _consider(self, pool: Pool) -> None:
        self._screened_cache.add(pool.address)
        self.stats.screened += 1

        report = self.screener.screen(pool.base_mint, liquidity_usd=pool.liquidity_usd)
        if report.verdict is Verdict.REJECT:
            self.stats.reject(f"safety: {report.blocking[0].name if report.blocking else 'error'}")
            return
        if report.verdict is Verdict.IMMATURE:
            # Not unsafe, just young. Allow a re-look on a later cycle.
            self._screened_cache.discard(pool.address)
            self.stats.reject("immature")
            return

        history = [dict(r) for r in self.store.snapshots_for(pool.address)]
        if len(history) < 2:
            history = [pool.to_row(), pool.to_row()]
        features = build_features(history).values
        score = float(self.scorer(features))
        if score < self.cfg.min_score:
            self.stats.reject(f"score {score:.2f} < {self.cfg.min_score:.2f}")
            return

        equity = self.portfolio.equity()
        decision = self.sizer.size(
            equity_usd=equity, win_prob=score, liquidity_usd=pool.liquidity_usd, dex=pool.dex
        )
        if not decision.approved:
            self.stats.reject(f"sizing: {decision.reason[:40]}")
            return

        usd = decision.usd * self.portfolio.size_multiplier()
        allowed, why = self.portfolio.can_open(pool.address, usd, pool.dex)
        if not allowed:
            self.stats.reject(f"portfolio: {why[:40]}")
            return

        # Emit the signal before executing. The signal is the product; paper
        # execution exists to measure whether the signal was any good, and an
        # external sniper front-end may be the thing that actually trades it.
        if self.emitter is not None:
            self.emitter.emit(
                self._build_signal(pool, score, decision, usd, report)
            )

        result = self.broker.buy(
            pool=pool.address, mint=pool.base_mint, usd=usd, price=pool.price_usd,
            liquidity_usd=pool.liquidity_usd, dex=pool.dex, first_buy=True,
        )
        if not result.ok:
            self.stats.reject(f"fill: {result.error}")
            return

        position = Position(
            pool=pool.address, mint=pool.base_mint, symbol=pool.name.split("/")[0].strip()[:16],
            entry_price=result.price, tokens=result.tokens, cost_usd=usd + result.fee_usd,
            opened_at=datetime.now(timezone.utc), dex=pool.dex,
            entry_liquidity_usd=pool.liquidity_usd, take_profit=self.cfg.take_profit,
            stop_loss=self.cfg.stop_loss, trailing_stop=self.cfg.trailing_stop,
            max_hold_min=self.cfg.max_hold_min,
            meta={"score": score, "risk_score": report.risk_score},
        )
        self.portfolio.open(position, usd + result.fee_usd)
        monitor = ExitMonitor(config=self.cfg.exits)
        monitor.update(result.price, pool.liquidity_usd)
        self._monitors[pool.address] = monitor
        self.stats.entered += 1
        log.info(
            "ENTER %s $%.2f @ %.3e score=%.3f liq=$%.0f slip=%.2f%% [%s]",
            position.symbol, usd, result.price, score, pool.liquidity_usd,
            100 * result.slippage_pct, pool.dex,
        )

    def _build_signal(self, pool: Pool, score: float, decision: Any, usd: float, report: Any) -> Signal:
        """Package a decision as an executable instruction."""
        reasons: list[str] = []
        warnings: list[str] = []

        tf = pool.tf("m5")
        if tf.buyers:
            reasons.append(f"{tf.buyers} unique buyers in 5m, buy ratio {tf.buy_ratio:.2f}")
        if pool.liquidity_usd:
            reasons.append(f"liquidity ${pool.liquidity_usd:,.0f}")
        if report is not None and getattr(report, "risk_score", 0) == 0:
            reasons.append("clean safety screen")

        dev_launches = 0
        dev_wallet = ""
        launch = self.store.conn.execute(
            "SELECT dev_wallet FROM launches WHERE mint=?", (pool.base_mint,)
        ).fetchone()
        if launch:
            dev_wallet = launch["dev_wallet"] or ""
            record = self.store.dev_record(dev_wallet) or {}
            dev_launches = max(0, int(record.get("launches", 1)) - 1)
            if dev_launches == 0:
                reasons.append("deployer has no prior launches on record")
            elif dev_launches >= 3:
                warnings.append(f"deployer has launched {dev_launches} tokens before")

        if report is not None and getattr(report, "failures", None):
            for check in report.failures[:2]:
                warnings.append(check.name)

        survival = score
        scorer = getattr(self.scorer, "__self__", None)
        if scorer is not None and hasattr(scorer, "survival_probability"):
            history = [dict(r) for r in self.store.snapshots_for(pool.address)] or [pool.to_row()] * 2
            survival = scorer.survival_probability(build_features(history).values)

        return Signal(
            mint=pool.base_mint,
            symbol=pool.name.split("/")[0].strip()[:16],
            pool=pool.address,
            dex=pool.dex,
            score=score,
            survival_probability=survival,
            reasons=reasons,
            warnings=warnings,
            valid_for_seconds=self.cfg.signal_ttl_seconds,
            token_age_seconds=pool.age_minutes * 60 if pool.created_at else 0.0,
            suggested_usd=usd,
            max_usd=pool.liquidity_usd * self.cfg.sizing.max_pool_fraction,
            liquidity_usd=pool.liquidity_usd,
            expected_slippage_pct=getattr(decision, "round_trip_cost", 0.0) / 2,
            exit_plan=build_exit_plan(
                take_profit=self.cfg.take_profit, stop_loss=self.cfg.stop_loss,
                trailing_stop=self.cfg.trailing_stop, max_hold_min=self.cfg.max_hold_min,
            ),
            reference_price=pool.price_usd,
            safety_risk_score=getattr(report, "risk_score", 0.0) if report else 0.0,
            dev_wallet=dev_wallet,
            dev_prior_launches=dev_launches,
        )

    # -------------------------------------------------------------- reporting

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "live" if self.broker.is_live else "paper",
            **self.stats.__dict__ | {"started_at": self.stats.started_at.isoformat()},
            "portfolio": self.portfolio.stats(),
        }
