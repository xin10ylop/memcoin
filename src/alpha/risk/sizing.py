"""Position sizing for fat-tailed, negatively-skewed-in-frequency payoffs.

Two independent forces set the size of an order, and they pull in opposite
directions.

**Cost efficiency.** Round-trip cost as a fraction of notional is U-shaped in
order size. Fixed network costs (signature, priority fee, Jito tip, ATA rent)
are amortised over the notional, so they punish small orders; AMM price impact
grows with the order's share of pool depth, so it punishes large ones. Writing
cost as a function of order size ``A`` against pool liquidity ``L``:

    cost(A) ≈ 2f + 2·latency + 4A/L + 2·netfee/A

Differentiating and solving ``d cost/dA = 0`` gives a closed form for the
cost-minimising order:

    A* = sqrt(netfee · L / 2)

That is roughly $98 into a $30k pool and $400 into a $500k pool — and it is why
sizing must scale with the square root of liquidity rather than being a fixed
dollar amount.

**Risk of ruin.** The payoff distribution here is extreme: most trades lose,
a few return multiples. Kelly is the right framework but full Kelly is not:
Kelly assumes the probability estimate is exact, and ours is an estimate from a
noisy model on a non-stationary market. Overestimating edge produces a fraction
that is superlinearly too large, and the resulting drawdowns compound. The
system therefore uses a small fraction of Kelly (default one-fifth) and caps it
hard.

The final size is the *minimum* of the two answers, so neither cost efficiency
nor risk tolerance can be violated to satisfy the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alpha.execution.costs import CostModel


def kelly_fraction(win_prob: float, win_payoff: float, loss_payoff: float = 1.0) -> float:
    """Kelly fraction for a binary bet.

    ``win_payoff`` is the profit per unit staked when the trade wins,
    ``loss_payoff`` the loss per unit when it loses (1.0 = lose the stake).
    Returns 0 when the edge is non-positive — there is no such thing as a
    correctly-sized bet on a negative expectation.
    """
    if not (0.0 < win_prob < 1.0) or win_payoff <= 0 or loss_payoff <= 0:
        return 0.0
    b = win_payoff / loss_payoff
    f = (win_prob * (b + 1.0) - 1.0) / b
    return max(0.0, min(1.0, f)) if math.isfinite(f) else 0.0


def optimal_order_usd(liquidity_usd: float, network_fee_usd: float) -> float:
    """Cost-minimising order size, ``A* = sqrt(netfee · L / 2)``.

    Derived by minimising total round-trip cost in the module docstring.
    """
    if liquidity_usd <= 0 or network_fee_usd <= 0:
        return 0.0
    value = math.sqrt(network_fee_usd * liquidity_usd / 2.0)
    return value if math.isfinite(value) else 0.0


@dataclass
class SizingConfig:
    """Limits on any single position."""

    # Fraction of Kelly to actually bet. Full Kelly assumes a perfectly known
    # edge; ours is estimated, so we take a small multiple of it.
    kelly_fraction_used: float = 0.20
    # Hard ceiling on any one position as a share of equity, regardless of how
    # confident the model is. No single memecoin may be able to hurt the book.
    max_position_pct: float = 0.02
    min_position_usd: float = 25.0
    max_position_usd: float = 2_000.0
    # Never take more than this share of pool depth, even if equity allows it.
    max_pool_fraction: float = 0.015
    # Reject any trade whose round-trip cost exceeds this: the friction alone
    # would eat a normal winner.
    max_round_trip_cost: float = 0.12
    # Assumed payoff geometry, matching the default label barriers.
    take_profit: float = 2.00
    stop_loss: float = 0.45


@dataclass
class SizeDecision:
    usd: float
    approved: bool
    reason: str
    kelly_raw: float = 0.0
    kelly_used: float = 0.0
    cost_optimal_usd: float = 0.0
    round_trip_cost: float = 0.0
    pool_fraction: float = 0.0

    def __str__(self) -> str:
        verdict = f"${self.usd:,.2f}" if self.approved else "REJECT"
        return f"{verdict} ({self.reason})"


class PositionSizer:
    """Combines cost-optimal sizing with fractional-Kelly risk sizing."""

    def __init__(self, config: SizingConfig | None = None, costs: CostModel | None = None) -> None:
        self.cfg = config or SizingConfig()
        self.costs = costs or CostModel()

    def size(
        self,
        *,
        equity_usd: float,
        win_prob: float,
        liquidity_usd: float,
        dex: str = "",
    ) -> SizeDecision:
        """Decide the dollar size of a position, or reject the trade."""
        cfg = self.cfg
        if equity_usd <= 0:
            return SizeDecision(0.0, False, "no equity")

        # --- risk-based ceiling -------------------------------------------
        kelly = kelly_fraction(win_prob, cfg.take_profit, cfg.stop_loss)
        if kelly <= 0:
            return SizeDecision(0.0, False, f"no edge at p={win_prob:.3f}", kelly_raw=kelly)
        kelly_used = min(kelly * cfg.kelly_fraction_used, cfg.max_position_pct)
        risk_usd = equity_usd * kelly_used

        # --- cost-based target --------------------------------------------
        net_fee = self.costs.network_fee_usd(first_buy=True)
        cost_opt = optimal_order_usd(liquidity_usd, net_fee)

        # --- depth ceiling -------------------------------------------------
        depth_cap = liquidity_usd * cfg.max_pool_fraction

        usd = min(risk_usd, max(cost_opt, cfg.min_position_usd), depth_cap, cfg.max_position_usd)

        if usd < cfg.min_position_usd:
            return SizeDecision(
                0.0, False,
                f"size ${usd:,.2f} below minimum ${cfg.min_position_usd:,.0f}",
                kelly_raw=kelly, kelly_used=kelly_used, cost_optimal_usd=cost_opt,
            )

        rt_cost = self.costs.round_trip_cost_pct(usd, liquidity_usd, dex)
        if rt_cost > cfg.max_round_trip_cost:
            return SizeDecision(
                0.0, False,
                f"round-trip cost {rt_cost:.1%} exceeds limit {cfg.max_round_trip_cost:.1%}",
                kelly_raw=kelly, kelly_used=kelly_used, cost_optimal_usd=cost_opt,
                round_trip_cost=rt_cost,
            )

        return SizeDecision(
            usd=round(usd, 2),
            approved=True,
            reason=f"kelly={kelly:.3f}×{cfg.kelly_fraction_used} cost_opt=${cost_opt:,.0f}",
            kelly_raw=kelly,
            kelly_used=kelly_used,
            cost_optimal_usd=cost_opt,
            round_trip_cost=rt_cost,
            pool_fraction=usd / max(liquidity_usd, 1e-9),
        )

    def breakeven_win_prob(self, round_trip_cost: float = 0.0) -> float:
        """Minimum win probability for a positive expectation.

        With take-profit ``T``, stop ``S`` and round-trip cost ``c``:
            p·(T−c) − (1−p)·(S+c) = 0  ⟹  p = (S+c) / (T+S)
        """
        cfg = self.cfg
        return (cfg.stop_loss + round_trip_cost) / (cfg.take_profit + cfg.stop_loss)
