"""Real-time pump.fun launch stream via PumpPortal.

This is the system's low-latency path. The polling collector discovers pools
from GeckoTerminal at 1–15 minutes old, which measurement showed is far too late
for the large multiples: from a 2-minute entry the best outcome observed was
7.3x, and from 5 minutes it was 4.6x, whereas entries at the launch candle
reached 318x. The websocket delivers creations within seconds.

Each ``create`` event carries more than a pool listing ever does:

* ``traderPublicKey`` — the deployer's wallet, at the moment of deployment.
* ``solAmount`` / ``initialBuy`` — how much of their own token the deployer
  bought in the creating transaction. This is the bundle size, available
  instantly rather than inferred later from holder distributions.
* ``vSolInBondingCurve`` / ``vTokensInBondingCurve`` — exact curve state, which
  feeds :mod:`alpha.data.bondingcurve` directly with no estimation.
* ``uri`` — the metadata document, so static social presence can be scored
  before the token has traded at all.

Seeing every deployer at creation also builds something no vendor sells: a local
record of which wallets launch which tokens and how those tokens turn out. A
deployer's history is the strongest free rug signal available, and after running
this stream for a while the record is first-hand rather than bought.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

WS_URL = "wss://pumpportal.fun/api/data"

LAMPORTS_PER_SOL = 1_000_000_000
#: Virtual SOL reserve of a fresh pump.fun curve, before any buying.
INITIAL_VSOL = 30.0


@dataclass(slots=True)
class Launch:
    """A pump.fun token creation, observed at the moment it happened."""

    mint: str
    name: str
    symbol: str
    uri: str
    dev_wallet: str
    signature: str
    bonding_curve: str
    pool: str
    dev_buy_sol: float
    dev_tokens: float
    v_sol: float
    v_tokens: float
    market_cap_sol: float
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def dev_curve_share(self) -> float:
        """Fraction of the way to graduation the deployer bought for themselves.

        Graduation requires 85.0054 SOL of net buying. A deployer who puts in
        5 SOL has taken roughly 6% of the curve at the lowest prices on it, and
        holds a position they can dump into any subsequent rally.
        """
        from alpha.data.bondingcurve import GRADUATION_SOL

        return max(0.0, self.dev_buy_sol) / GRADUATION_SOL

    @property
    def net_sol(self) -> float:
        """Net SOL bought into the curve so far."""
        return max(0.0, self.v_sol - INITIAL_VSOL)

    @property
    def is_self_funded(self) -> bool:
        """Whether the deployer bought their own token at creation at all."""
        return self.dev_buy_sol > 0.01

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "Launch | None":
        if event.get("txType") != "create" or not event.get("mint"):
            return None

        def num(key: str) -> float:
            try:
                return float(event.get(key) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        return cls(
            mint=str(event.get("mint", "")),
            name=str(event.get("name", "")),
            symbol=str(event.get("symbol", "")),
            uri=str(event.get("uri", "")),
            dev_wallet=str(event.get("traderPublicKey", "")),
            signature=str(event.get("signature", "")),
            bonding_curve=str(event.get("bondingCurveKey", "")),
            pool=str(event.get("pool", "")),
            dev_buy_sol=num("solAmount"),
            dev_tokens=num("initialBuy"),
            v_sol=num("vSolInBondingCurve"),
            v_tokens=num("vTokensInBondingCurve"),
            market_cap_sol=num("marketCapSol"),
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "name": self.name,
            "symbol": self.symbol,
            "uri": self.uri,
            "dev_wallet": self.dev_wallet,
            "signature": self.signature,
            "bonding_curve": self.bonding_curve,
            "pool_kind": self.pool,
            "dev_buy_sol": self.dev_buy_sol,
            "dev_tokens": self.dev_tokens,
            "dev_curve_share": self.dev_curve_share,
            "v_sol": self.v_sol,
            "v_tokens": self.v_tokens,
            "market_cap_sol": self.market_cap_sol,
            "observed_at": self.observed_at.isoformat(),
        }


@dataclass(slots=True)
class Migration:
    """A token graduating from the bonding curve to PumpSwap."""

    mint: str
    signature: str
    pool: str
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> "Migration | None":
        if event.get("txType") != "migrate" or not event.get("mint"):
            return None
        return cls(
            mint=str(event.get("mint", "")),
            signature=str(event.get("signature", "")),
            pool=str(event.get("pool", "")),
        )


class PumpPortalStream:
    """Resilient websocket client for the PumpPortal data API.

    Reconnects with exponential backoff. A dropped connection on this stream is
    unrecoverable data loss — a launch missed is a launch that can never be
    observed at t=0 again — so reconnection is aggressive rather than polite.
    """

    def __init__(
        self,
        *,
        url: str = WS_URL,
        subscribe_launches: bool = True,
        subscribe_migrations: bool = True,
        max_backoff: float = 30.0,
    ) -> None:
        self.url = url
        self.subscribe_launches = subscribe_launches
        self.subscribe_migrations = subscribe_migrations
        self.max_backoff = max_backoff
        self.stats = {"launches": 0, "migrations": 0, "reconnects": 0, "errors": 0, "other": 0}
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    async def run(
        self,
        on_launch: Callable[[Launch], Awaitable[None] | None] | None = None,
        on_migration: Callable[[Migration], Awaitable[None] | None] | None = None,
        max_seconds: float | None = None,
    ) -> dict[str, int]:
        import websockets

        started = time.monotonic()
        backoff = 1.0

        while not self._stop:
            if max_seconds and time.monotonic() - started >= max_seconds:
                break
            try:
                async with websockets.connect(self.url, open_timeout=20, ping_interval=20) as ws:
                    if self.subscribe_launches:
                        await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    if self.subscribe_migrations:
                        await ws.send(json.dumps({"method": "subscribeMigration"}))
                    backoff = 1.0  # connected cleanly; reset the backoff
                    log.info("pumpportal stream connected")

                    while not self._stop:
                        if max_seconds and time.monotonic() - started >= max_seconds:
                            break
                        remaining = (
                            max_seconds - (time.monotonic() - started) if max_seconds else 60.0
                        )
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, remaining))
                        except asyncio.TimeoutError:
                            continue
                        await self._dispatch(raw, on_launch, on_migration)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop:
                    break
                self.stats["reconnects"] += 1
                log.warning("pumpportal stream dropped (%s); reconnecting in %.1fs",
                            type(exc).__name__, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

        return dict(self.stats)

    async def _dispatch(
        self,
        raw: str | bytes,
        on_launch: Callable[[Launch], Awaitable[None] | None] | None,
        on_migration: Callable[[Migration], Awaitable[None] | None] | None,
    ) -> None:
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            self.stats["errors"] += 1
            return
        if not isinstance(event, dict):
            self.stats["errors"] += 1
            return
        # Subscription acknowledgements arrive as a lone "message" key.
        if set(event) == {"message"}:
            return

        launch = Launch.from_event(event)
        if launch is not None:
            self.stats["launches"] += 1
            if on_launch:
                await _maybe_await(on_launch(launch))
            return

        migration = Migration.from_event(event)
        if migration is not None:
            self.stats["migrations"] += 1
            if on_migration:
                await _maybe_await(on_migration(migration))
            return

        self.stats["other"] += 1


async def _maybe_await(value: Awaitable[None] | None) -> None:
    if value is not None and hasattr(value, "__await__"):
        with contextlib.suppress(Exception):
            await value
