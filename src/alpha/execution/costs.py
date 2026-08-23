"""Realistic execution cost modelling for constant-product AMMs.

Most memecoin backtests fail here rather than in the signal. They assume fills
at the mid price, which for this asset class is not a small approximation — it
is the difference between a profitable strategy and a losing one. On a pool with
$30k of liquidity, a $500 order moves the price roughly 3.3% on entry and again
on exit, so a round trip starts about 7% underwater before any thesis is tested.

The maths, for a constant-product pool ``x·y = k`` where ``x`` is the token
reserve and ``y`` the quote reserve:

    buying with quote amount ``A`` (effective ``A' = A(1−f)`` after the swap fee)
        tokens_out       = x·A' / (y + A')
        effective price  = A / tokens_out
        spot price       = y / x
        slippage         = effective/spot − 1 = 1/(1−f) · (1 + A'/y) − 1

The dominant term is ``A/y``: **cost scales with position size relative to pool
depth**. This single fact drives position sizing throughout the system, and it
is why the risk layer caps orders as a fraction of liquidity rather than as a
fixed dollar amount.

GeckoTerminal reports ``reserve_in_usd`` as total pool value, which for a
balanced constant-product pool is twice the quote reserve — hence the
``quote_reserve = liquidity/2`` conversion below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class FillSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


# Swap fees by venue, as a fraction of notional (verified from venue docs).
VENUE_FEES = {
    "pump-fun": 0.0100,     # pump.fun bonding curve
    "pumpswap": 0.0025,     # post-migration AMM
    "raydium": 0.0025,
    "raydium-clmm": 0.0025,
    "meteora": 0.0020,
    "orca": 0.0030,
    "_default": 0.0030,
}


@dataclass
class Fill:
    """Result of simulating one order."""

    side: FillSide
    requested_usd: float
    filled_usd: float
    tokens: float
    spot_price: float
    effective_price: float
    price_impact: float      # fraction, from pool depth alone
    swap_fee_usd: float
    network_fee_usd: float
    total_cost_usd: float    # everything lost vs a hypothetical mid fill
    slippage_pct: float      # total, including fees
    rejected: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected


@dataclass
class CostModel:
    """Parameters governing simulated execution.

    Defaults reflect a competitive-but-not-elite Solana setup: a priority fee
    and Jito tip large enough to land reliably during congestion, and a latency
    penalty representing the price drift between deciding and landing.
    """

    # Fixed per-transaction costs, denominated in SOL.
    base_fee_sol: float = 0.000005          # 5,000 lamports signature fee
    priority_fee_sol: float = 0.0008        # compute-unit price for reliable landing
    jito_tip_sol: float = 0.0012            # tip to be included in a bundle
    # Some routers/bots charge a percentage on top of the venue fee.
    platform_fee_pct: float = 0.0
    # Account rent for an associated token account, paid on first buy.
    ata_rent_sol: float = 0.00204
    sol_price_usd: float = 150.0

    # Adverse price drift between signal and landed transaction. Memecoin
    # launches move fast and we are not the fastest bot on the chain, so
    # assuming zero here would be self-flattery.
    latency_slippage_pct: float = 0.010
    # Refuse orders that would consume more than this share of pool depth:
    # beyond it the fill is so poor the trade cannot be profitable.
    max_pool_fraction: float = 0.02
    # Fraction of failed/reverted transactions whose fees are paid for nothing.
    failed_tx_rate: float = 0.05

    def venue_fee(self, dex: str) -> float:
        return VENUE_FEES.get((dex or "").lower(), VENUE_FEES["_default"])

    def network_fee_usd(self, *, first_buy: bool = False) -> float:
        sol = self.base_fee_sol + self.priority_fee_sol + self.jito_tip_sol
        if first_buy:
            sol += self.ata_rent_sol
        # Amortise the cost of transactions that pay fees but never land.
        sol /= max(1e-9, 1.0 - self.failed_tx_rate)
        return sol * self.sol_price_usd

    def price_impact(self, usd_amount: float, liquidity_usd: float, fee: float) -> float:
        """Constant-product price impact for an order of ``usd_amount``.

        ``liquidity_usd`` is total pool value; the quote reserve is half of it.
        """
        quote_reserve = max(liquidity_usd, 1e-9) / 2.0
        effective = usd_amount * (1.0 - fee)
        if quote_reserve <= 0:
            return 1.0
        ratio = (1.0 / (1.0 - fee)) * (1.0 + effective / quote_reserve)
        impact = ratio - 1.0
        return impact if math.isfinite(impact) else 1.0

    def simulate(
        self,
        side: FillSide,
        usd_amount: float,
        spot_price: float,
        liquidity_usd: float,
        *,
        dex: str = "",
        first_buy: bool = False,
        apply_latency: bool = True,
    ) -> Fill:
        """Simulate a market order and return the realised fill."""
        fee = self.venue_fee(dex)
        net_fee = self.network_fee_usd(first_buy=first_buy and side is FillSide.BUY)

        if usd_amount <= 0 or spot_price <= 0:
            return Fill(side, usd_amount, 0, 0, spot_price, 0, 0, 0, net_fee, net_fee, 0,
                        rejected=True, reason="non-positive amount or price")

        if liquidity_usd <= 0:
            return Fill(side, usd_amount, 0, 0, spot_price, 0, 1.0, 0, net_fee, net_fee, 1.0,
                        rejected=True, reason="no liquidity")

        pool_fraction = usd_amount / liquidity_usd
        if pool_fraction > self.max_pool_fraction:
            return Fill(
                side, usd_amount, 0, 0, spot_price, 0, pool_fraction, 0, 0, 0, pool_fraction,
                rejected=True,
                reason=f"order is {pool_fraction:.1%} of pool depth (limit {self.max_pool_fraction:.1%})",
            )

        impact = self.price_impact(usd_amount, liquidity_usd, fee)
        latency = self.latency_slippage_pct if apply_latency else 0.0

        # Both impact and latency work against us on each side.
        if side is FillSide.BUY:
            effective_price = spot_price * (1.0 + impact + latency)
        else:
            effective_price = spot_price * (1.0 - impact - latency)
        effective_price = max(effective_price, 1e-18)

        platform_fee = usd_amount * self.platform_fee_pct
        swap_fee = usd_amount * fee
        gross = usd_amount - platform_fee
        tokens = gross / effective_price if side is FillSide.BUY else gross / spot_price

        if side is FillSide.BUY:
            filled_usd = gross
        else:
            # Selling: we receive fewer dollars than the mid price implies.
            filled_usd = gross * (1.0 - impact - latency)

        # Total cost relative to a frictionless mid-price fill.
        mid_value = usd_amount
        realised = filled_usd if side is FillSide.SELL else usd_amount * (spot_price / effective_price)
        total_cost = abs(mid_value - realised) + net_fee + platform_fee
        slippage_pct = impact + latency + fee + self.platform_fee_pct + net_fee / max(usd_amount, 1e-9)

        return Fill(
            side=side,
            requested_usd=usd_amount,
            filled_usd=filled_usd,
            tokens=tokens,
            spot_price=spot_price,
            effective_price=effective_price,
            price_impact=impact,
            swap_fee_usd=swap_fee,
            network_fee_usd=net_fee,
            total_cost_usd=total_cost,
            slippage_pct=slippage_pct,
        )

    def round_trip_cost_pct(self, usd_amount: float, liquidity_usd: float, dex: str = "") -> float:
        """Total round-trip cost as a fraction of notional.

        This is the hurdle every trade must clear before it makes a cent, and
        the number the strategy layer uses to decide whether a token is even
        worth considering at a given size.
        """
        buy = self.simulate(FillSide.BUY, usd_amount, 1.0, liquidity_usd, dex=dex, first_buy=True)
        sell = self.simulate(FillSide.SELL, usd_amount, 1.0, liquidity_usd, dex=dex)
        if buy.rejected or sell.rejected:
            return 1.0
        return buy.slippage_pct + sell.slippage_pct
