"""The cohort collector — the system's data engine.

Runs two interleaved jobs against a fixed API budget:

1. **Discovery** sweeps ``new_pools`` and writes a birth certificate for every
   pool it has never seen. Solana creates ~14 pools/minute and the endpoint
   exposes ~14 minutes of history across 10 pages, so a once-per-minute sweep
   captures essentially the entire universe with a wide safety margin.
2. **Tracking** re-observes known pools on a logarithmic schedule using batched
   ``pools/multi`` calls (30 pools per request).

The observation schedule is logarithmic because memecoin lifecycles are: most
of the informative variance happens in the first ten minutes, and a token that
is still alive at six hours changes slowly. Sampling densely early and sparsely
later puts the API budget where the information is.

The critical property is that pools are recorded *before* their outcome is
known. A dataset assembled retrospectively from tokens that are still visible
today would contain only survivors and would teach a model nothing about death.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from alpha.data.geckoterminal import GeckoTerminalClient, Pool
from alpha.data.store import Store, utcnow

log = logging.getLogger(__name__)

# Minutes after discovery at which we re-observe. Dense early, sparse late.
OBSERVATION_SCHEDULE_MIN = (
    1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 95, 130, 180, 250, 360, 540, 720, 1080, 1440,
)

# A pool this illiquid for this many consecutive observations is dead; stop
# spending budget on it.
DEAD_LIQUIDITY_USD = 150.0
DEAD_STRIKES = 3


@dataclass
class CollectorConfig:
    db_path: str = "data/alpha.db"
    requests_per_minute: float = 25.0
    discovery_pages: int = 10
    # Fraction of the per-cycle budget reserved for discovery. Discovery is
    # non-negotiable: a missed pool can never be recovered, whereas a missed
    # observation only thins one token's time series.
    discovery_share: float = 0.45
    cycle_seconds: float = 60.0
    track_trades: bool = True
    # Only fetch trade-level data for pools showing real activity, since each
    # costs a full API call.
    trades_min_liquidity_usd: float = 3_000.0
    trades_min_buys_m5: int = 15
    sol_quoted_only: bool = True
    max_runtime_seconds: float | None = None
    stats_every: int = 5


@dataclass
class CollectorStats:
    cycles: int = 0
    discovered: int = 0
    snapshots: int = 0
    trades: int = 0
    retired: int = 0
    errors: int = 0
    started_at: datetime = field(default_factory=utcnow)

    def as_dict(self) -> dict[str, Any]:
        elapsed = (utcnow() - self.started_at).total_seconds()
        return {
            "cycles": self.cycles,
            "discovered": self.discovered,
            "snapshots": self.snapshots,
            "trades": self.trades,
            "retired": self.retired,
            "errors": self.errors,
            "elapsed_min": round(elapsed / 60, 2),
            "pools_per_hour": round(self.discovered / max(elapsed / 3600, 1e-9), 1),
        }


def next_due(observations: int, discovered_at: datetime) -> datetime | None:
    """When to next observe a pool that has been observed ``observations`` times.

    Returns ``None`` once the schedule is exhausted, meaning the pool should be
    retired from active tracking.
    """
    if observations >= len(OBSERVATION_SCHEDULE_MIN):
        return None
    return discovered_at + timedelta(minutes=OBSERVATION_SCHEDULE_MIN[observations])


class Collector:
    """Continuously discovers and tracks new pools within an API budget."""

    def __init__(self, config: CollectorConfig | None = None, client: GeckoTerminalClient | None = None):
        self.cfg = config or CollectorConfig()
        self.store = Store(self.cfg.db_path)
        self.client = client or GeckoTerminalClient()
        self.client.http.bucket.rate = self.cfg.requests_per_minute / 60.0
        self.stats = CollectorStats()
        self._stop = False
        self._dead_strikes: dict[str, int] = {}

    def request_stop(self, *_: object) -> None:
        log.info("stop requested — finishing current cycle")
        self._stop = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.request_stop)
            except (ValueError, OSError):
                pass  # not on the main thread

    # ------------------------------------------------------------------- cycle

    def run(self) -> CollectorStats:
        """Main loop. Runs until stopped, or until ``max_runtime_seconds``."""
        started = time.monotonic()
        log.info(
            "collector starting: %.0f req/min, %d discovery pages, %.0fs cycles",
            self.cfg.requests_per_minute, self.cfg.discovery_pages, self.cfg.cycle_seconds,
        )
        while not self._stop:
            cycle_start = time.monotonic()
            try:
                self.run_cycle()
            except Exception:
                self.stats.errors += 1
                log.exception("cycle failed — continuing")
            self.stats.cycles += 1

            if self.stats.cycles % self.cfg.stats_every == 0:
                log.info("collector stats: %s | store: %s", self.stats.as_dict(), self.store.stats())

            if self.cfg.max_runtime_seconds and time.monotonic() - started >= self.cfg.max_runtime_seconds:
                log.info("max runtime reached")
                break

            # Pace to the configured cycle length; the rate limiter may already
            # have consumed most of it.
            remaining = self.cfg.cycle_seconds - (time.monotonic() - cycle_start)
            end = time.monotonic() + max(0.0, remaining)
            while not self._stop and time.monotonic() < end:
                time.sleep(min(0.5, end - time.monotonic()))

        log.info("collector stopped: %s", self.stats.as_dict())
        return self.stats

    def run_cycle(self) -> None:
        budget = max(1, int(self.cfg.requests_per_minute * self.cfg.cycle_seconds / 60.0))
        discovery_budget = max(1, int(budget * self.cfg.discovery_share))
        self.discover(pages=min(self.cfg.discovery_pages, discovery_budget))
        self.track(budget=budget - discovery_budget)

    # --------------------------------------------------------------- discovery

    def discover(self, pages: int) -> int:
        """Sweep new-pool pages and record any pool we have not seen before."""
        found: dict[str, Pool] = {}
        for page in range(1, pages + 1):
            batch = self.client.new_pools(page)
            if not batch:
                break
            for p in batch:
                if p.address and (not self.cfg.sol_quoted_only or p.is_sol_quoted):
                    found.setdefault(p.address, p)

        if not found:
            return 0
        pools = list(found.values())
        new_count = self.store.record_pools(pools)
        self.store.record_snapshots(pools)
        # Newly discovered pools become due for their first tracked observation
        # one minute after discovery.
        self.store.schedule_many(
            [p.address for p in pools], utcnow() + timedelta(minutes=OBSERVATION_SCHEDULE_MIN[0])
        )
        self.stats.discovered += new_count
        self.stats.snapshots += len(pools)
        log.debug("discovery: %d pools seen, %d new", len(pools), new_count)
        return new_count

    # ---------------------------------------------------------------- tracking

    def track(self, budget: int) -> int:
        """Re-observe pools that are due, in batches of 30 per API call."""
        if budget <= 0:
            return 0
        # Reserve a slice of the budget for trade fetches.
        trade_budget = max(0, int(budget * 0.25)) if self.cfg.track_trades else 0
        snapshot_calls = budget - trade_budget
        due = self.store.due_pools(limit=snapshot_calls * 30)
        if not due:
            return 0

        counts = self.store.observation_counts(due)
        observed: list[Pool] = []
        for i in range(0, len(due), 30):
            chunk = due[i : i + 30]
            observed.extend(self.client.pools_multi(chunk))

        if not observed:
            return 0
        self.store.record_snapshots(observed)
        self.stats.snapshots += len(observed)

        now = utcnow()
        updates: list[tuple[str, datetime]] = []
        retire: list[str] = []
        by_address = {p.address: p for p in observed}

        for pool_addr in due:
            pool = by_address.get(pool_addr)
            if pool is None:
                # The API did not return it — likely delisted. Retry once later,
                # then give up via the strike counter.
                strikes = self._dead_strikes.get(pool_addr, 0) + 1
                self._dead_strikes[pool_addr] = strikes
                if strikes >= DEAD_STRIKES:
                    retire.append(pool_addr)
                else:
                    updates.append((pool_addr, now + timedelta(minutes=10)))
                continue

            if pool.liquidity_usd < DEAD_LIQUIDITY_USD:
                strikes = self._dead_strikes.get(pool_addr, 0) + 1
                self._dead_strikes[pool_addr] = strikes
                if strikes >= DEAD_STRIKES:
                    retire.append(pool_addr)
                    continue
            else:
                self._dead_strikes.pop(pool_addr, None)

            n = counts.get(pool_addr, 0) + 1
            row = self.store.pool_row(pool_addr)
            discovered_at = _parse(row["discovered_at"]) if row else now
            due_at = next_due(n, discovered_at or now)
            if due_at is None:
                retire.append(pool_addr)
            else:
                # Never schedule in the past; that would busy-loop the pool.
                updates.append((pool_addr, max(due_at, now + timedelta(seconds=30))))

        self.store.mark_observed(updates)
        if retire:
            self.store.retire(retire, "schedule_complete_or_dead")
            self.stats.retired += len(retire)
            for p in retire:
                self._dead_strikes.pop(p, None)

        if trade_budget:
            self._fetch_trades(observed, trade_budget)
        return len(observed)

    def _fetch_trades(self, pools: list[Pool], budget: int) -> None:
        """Pull swap-level data for the most active pools we just observed.

        Trades are expensive (one call each) but uniquely valuable: they carry
        the trading wallet, which is the only free route to wallet-level
        attribution and bundler/sniper detection.
        """
        candidates = [
            p for p in pools
            if p.liquidity_usd >= self.cfg.trades_min_liquidity_usd
            and p.tf("m5").buys >= self.cfg.trades_min_buys_m5
        ]
        candidates.sort(key=lambda p: p.tf("m5").volume_usd, reverse=True)
        for pool in candidates[:budget]:
            trades = self.client.trades(pool.address)
            if trades:
                self.stats.trades += self.store.record_trades(pool.address, trades)


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
