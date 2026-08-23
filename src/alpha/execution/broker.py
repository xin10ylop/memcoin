"""Broker abstraction: paper and live.

Every order in the system goes through a :class:`Broker`. The paper broker is
the default and the only one enabled without explicit configuration, so the
system cannot spend real money by accident — a live broker must be constructed
deliberately, with a signer, by someone who has read what it does.

The paper broker is not a toy. It prices fills through the same
:class:`~alpha.execution.costs.CostModel` the backtester uses, so paper results
and backtest results are directly comparable, and a strategy that looks
profitable on paper is not being flattered by a more forgiving fill model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from alpha.execution.costs import CostModel, Fill, FillSide

log = logging.getLogger(__name__)


@dataclass
class OrderResult:
    ok: bool
    side: FillSide
    pool: str
    mint: str
    usd: float
    tokens: float
    price: float
    fee_usd: float
    slippage_pct: float
    tx_signature: str | None = None
    error: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        if not self.ok:
            return f"{self.side.value.upper()} {self.pool[:10]} FAILED: {self.error}"
        return (
            f"{self.side.value.upper()} {self.pool[:10]} ${self.usd:,.2f} "
            f"@ {self.price:.3e} slip={self.slippage_pct:.2%}"
        )


class Broker(ABC):
    """Interface every execution venue must implement."""

    #: Whether this broker moves real funds. Used for loud logging and guards.
    is_live: bool = False

    @abstractmethod
    def buy(self, *, pool: str, mint: str, usd: float, price: float, liquidity_usd: float,
            dex: str = "", first_buy: bool = True) -> OrderResult:
        ...

    @abstractmethod
    def sell(self, *, pool: str, mint: str, tokens: float, price: float, liquidity_usd: float,
             dex: str = "") -> OrderResult:
        ...


class PaperBroker(Broker):
    """Simulated execution using the shared AMM cost model."""

    is_live = False

    def __init__(self, costs: CostModel | None = None) -> None:
        self.costs = costs or CostModel()
        self.orders: list[OrderResult] = []

    def buy(self, *, pool: str, mint: str, usd: float, price: float, liquidity_usd: float,
            dex: str = "", first_buy: bool = True) -> OrderResult:
        fill = self.costs.simulate(
            FillSide.BUY, usd, price, liquidity_usd, dex=dex, first_buy=first_buy
        )
        result = self._from_fill(FillSide.BUY, pool, mint, fill, usd)
        self.orders.append(result)
        return result

    def sell(self, *, pool: str, mint: str, tokens: float, price: float, liquidity_usd: float,
             dex: str = "") -> OrderResult:
        notional = tokens * price
        fill = self.costs.simulate(FillSide.SELL, notional, price, liquidity_usd, dex=dex)
        result = self._from_fill(FillSide.SELL, pool, mint, fill, notional)
        if result.ok:
            result.tokens = tokens
            result.usd = fill.filled_usd
        self.orders.append(result)
        return result

    @staticmethod
    def _from_fill(side: FillSide, pool: str, mint: str, fill: Fill, requested: float) -> OrderResult:
        if not fill.ok:
            return OrderResult(
                ok=False, side=side, pool=pool, mint=mint, usd=requested, tokens=0.0,
                price=fill.spot_price, fee_usd=0.0, slippage_pct=0.0, error=fill.reason,
            )
        return OrderResult(
            ok=True, side=side, pool=pool, mint=mint,
            usd=fill.filled_usd if side is FillSide.SELL else requested,
            tokens=fill.tokens, price=fill.effective_price,
            fee_usd=fill.network_fee_usd + fill.swap_fee_usd,
            slippage_pct=fill.slippage_pct, tx_signature="paper",
        )


class LiveBroker(Broker):
    """Real on-chain execution via Jupiter.

    Intentionally left unimplemented. Wiring this up requires a funded keypair,
    a reliable RPC endpoint, and transaction signing — and switching it on
    should be a deliberate act by someone who has validated the strategy on
    paper first, not a default that happens to be reachable.

    The integration points are documented in ``docs/GOING_LIVE.md``; the class
    exists so that the rest of the system is already written against the
    interface a live venue would satisfy.
    """

    is_live = True

    def __init__(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError(
            "Live execution is not enabled. This system is designed to be validated "
            "in paper mode first. See docs/GOING_LIVE.md for what implementing this "
            "requires and what to verify before risking capital."
        )

    def buy(self, **_: Any) -> OrderResult:  # pragma: no cover - unreachable
        raise NotImplementedError

    def sell(self, **_: Any) -> OrderResult:  # pragma: no cover - unreachable
        raise NotImplementedError
