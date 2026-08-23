"""pump.fun bonding-curve mechanics.

Pre-migration, a pump.fun token does not trade against a normal pool. It trades
against a *virtual* constant-product curve with fixed, publicly-known starting
constants, which means its entire pre-graduation price path is determined by one
number: how much net SOL has been bought.

That has a consequence worth stating loudly, because it bounds what any
pre-graduation strategy can possibly earn:

    Graduation occurs at 85.0054 SOL of net buying. At that point the virtual
    SOL reserve has gone 30 → 115.0054 and virtual tokens 1.073B → 279.9M.
    Price is proportional to vSol/vTokens, so the price at graduation is
    exactly 14.696x the launch price.

**A token that never graduates can never have risen more than 14.696x.** Any
strategy hunting 50x or 100x on the bonding curve is hunting something that
cannot exist there; those returns only occur post-migration. This directly
validates a take-profit target well below the ceiling, and it means a model
predicting "will this 100x" is predicting an impossibility for the majority of
its universe.

A second consequence shapes exits: price is *continuous* across migration, but
depth is not. The PumpSwap pool receives materially less depth than the curve
quoted against, so there is no "graduation pop" to hold for — only a step down
in liquidity, which raises the cost of every subsequent exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Exact launch constants of the pump.fun bonding curve.
INITIAL_VIRTUAL_TOKEN_RESERVES = 1_073_000_000_000_000    # 1.073B tokens (6 decimals)
INITIAL_VIRTUAL_SOL_RESERVES = 30_000_000_000             # 30 SOL (lamports)
TOTAL_SUPPLY = 1_000_000_000_000_000                      # 1B tokens (6 decimals)

LAMPORTS_PER_SOL = 1_000_000_000
TOKEN_DECIMALS = 10**6

#: Net SOL of buying required to graduate to PumpSwap.
GRADUATION_SOL = 85.0054
#: Virtual SOL reserve at graduation.
GRADUATION_VSOL = 115.0054
#: Tokens remaining on the curve at graduation.
GRADUATION_VTOKENS = 279_900_000.0

#: Real SOL actually migrated into the PumpSwap pool at graduation. The curve
#: quotes against 115.0054 SOL of *virtual* depth but only ~85 SOL is real, so
#: depth drops ~26% the moment a token graduates.
MIGRATED_REAL_SOL = 85.0054
#: Circulating tokens at graduation.
MIGRATED_TOKENS = 793_100_000.0

#: Measured platform-wide graduation rate. Published studies report 0.63% over
#: 655,770 tokens (Sept 2025) and ~1.4% all-time; this system's own launch
#: stream measured 1.85% on a smaller sample.
PLATFORM_GRADUATION_RATE = 0.0140

#: Hard structural ceiling on pre-graduation appreciation.
#: (vSol_grad/vTokens_grad) / (vSol_0/vTokens_0)
MAX_PREGRAD_MULTIPLE = (GRADUATION_VSOL / GRADUATION_VTOKENS) / (
    (INITIAL_VIRTUAL_SOL_RESERVES / LAMPORTS_PER_SOL)
    / (INITIAL_VIRTUAL_TOKEN_RESERVES / TOKEN_DECIMALS)
)


@dataclass(frozen=True)
class CurveState:
    """Bonding-curve state expressed in whole SOL and whole tokens."""

    virtual_sol: float
    virtual_tokens: float

    @property
    def price_sol(self) -> float:
        """Spot price in SOL per token."""
        return self.virtual_sol / self.virtual_tokens if self.virtual_tokens else 0.0

    @property
    def net_sol_bought(self) -> float:
        return self.virtual_sol - (INITIAL_VIRTUAL_SOL_RESERVES / LAMPORTS_PER_SOL)

    @property
    def progress(self) -> float:
        """Fraction of the way to graduation, in [0, 1]."""
        return max(0.0, min(1.0, self.net_sol_bought / GRADUATION_SOL))

    @property
    def multiple_from_launch(self) -> float:
        return self.price_sol / launch_price_sol()

    @property
    def has_graduated(self) -> bool:
        return self.net_sol_bought >= GRADUATION_SOL


def launch_price_sol() -> float:
    """Price of the very first token bought on a fresh curve."""
    return (INITIAL_VIRTUAL_SOL_RESERVES / LAMPORTS_PER_SOL) / (
        INITIAL_VIRTUAL_TOKEN_RESERVES / TOKEN_DECIMALS
    )


def state_from_net_sol(net_sol: float) -> CurveState:
    """Curve state after ``net_sol`` of net buying.

    Constant product is preserved: ``vSol · vTokens = k``.
    """
    v_sol0 = INITIAL_VIRTUAL_SOL_RESERVES / LAMPORTS_PER_SOL
    v_tok0 = INITIAL_VIRTUAL_TOKEN_RESERVES / TOKEN_DECIMALS
    k = v_sol0 * v_tok0
    v_sol = v_sol0 + max(0.0, net_sol)
    return CurveState(virtual_sol=v_sol, virtual_tokens=k / v_sol)


def buy_tokens_out(state: CurveState, sol_in: float, fee: float = 0.01) -> float:
    """Tokens received for ``sol_in``, exactly as the curve computes it."""
    if sol_in <= 0:
        return 0.0
    effective = sol_in * (1.0 - fee)
    k = state.virtual_sol * state.virtual_tokens
    new_sol = state.virtual_sol + effective
    return state.virtual_tokens - k / new_sol


def sell_sol_out(state: CurveState, tokens_in: float, fee: float = 0.01) -> float:
    """SOL received for selling ``tokens_in`` back into the curve."""
    if tokens_in <= 0:
        return 0.0
    k = state.virtual_sol * state.virtual_tokens
    new_tokens = state.virtual_tokens + tokens_in
    sol_out = state.virtual_sol - k / new_tokens
    return max(0.0, sol_out) * (1.0 - fee)


def breakeven_price_multiple(net_sol: float) -> float:
    """Multiple required for a curve buy-and-hold to break even.

    Selling back into a constant-product curve recovers less than was paid, and
    the shortfall grows with position size relative to curve depth. Expressed
    against the graduation reserve, the break-even condition is

        p(vSol) > (vSol / 115)^2

    which is why a naive buy-and-hold on the curve alone loses money: price must
    outrun a quadratic, not merely rise.
    """
    state = state_from_net_sol(net_sol)
    return (state.virtual_sol / GRADUATION_VSOL) ** 2


def breakeven_graduation_probability(net_sol: float) -> float:
    """Minimum P(graduate) that makes buying at ``net_sol`` and holding worthwhile.

    On the curve, price is proportional to the square of the virtual SOL
    reserve, so buying at ``vSol`` and holding to graduation returns exactly
    ``(115.0054 / vSol)^2``. Setting expected value to zero for an all-or-nothing
    bet (graduate and win that multiple, or fail and lose everything) gives

        p* = vSol^2 / 115.0054^2

    This is the most important number in the domain, because it can be compared
    directly against a base rate:

        at launch (vSol=30):  p* = 6.80%   actual base rate ~0.6-1.4%
        at vSol=50:           p* = 18.9%
        at vSol=80:           p* = 48.4%

    **Buying at launch and holding for graduation is therefore 5-10x negative
    expected value at the unconditional base rate**, and it gets worse further up
    the curve, not better. A graduation-hold strategy only works if conditioning
    features lift P(graduate) above p* — which is precisely what deployer
    reputation does, since elite deployers graduate at 40-71%.
    """
    state = state_from_net_sol(net_sol)
    p = (state.virtual_sol / GRADUATION_VSOL) ** 2
    return min(1.0, max(0.0, p))


def graduation_edge(net_sol: float, p_graduate: float) -> float:
    """Expected return of a hold-to-graduation bet, as a fraction of stake.

    Positive only when ``p_graduate`` exceeds
    :func:`breakeven_graduation_probability` at this point on the curve.
    """
    breakeven = breakeven_graduation_probability(net_sol)
    if breakeven <= 0:
        return 0.0
    payoff = 1.0 / breakeven          # multiple achieved if it graduates
    return p_graduate * payoff - 1.0


def post_migration_dead_liquidity() -> float:
    """Fraction of migrated SOL that can never be extracted by holders.

    At graduation ~85.0054 real SOL and 793.1M tokens enter the PumpSwap pool.
    Under constant product, selling *every* circulating token back into that
    pool leaves SOL stuck at ``k / (tokens_in_pool + circulating)``. Holders
    collectively pay in 85 SOL and can extract at most ~67.4 SOL.

    **The graduated cohort is negative-sum by construction, at roughly -21%
    before fees.** Holding through migration is therefore not a neutral act with
    upside; it is a bet that you exit ahead of the queue. This is a stronger
    argument for exiting on the curve than any sentiment signal.
    """
    k = MIGRATED_REAL_SOL * (GRADUATION_VTOKENS - 73_000_000.0 + 73_000_000.0)
    # Pool starts with ~206.9M tokens against 85.0054 SOL.
    pool_tokens = 206_900_000.0
    k = MIGRATED_REAL_SOL * pool_tokens
    final_sol = k / (pool_tokens + MIGRATED_TOKENS)
    extractable = MIGRATED_REAL_SOL - final_sol
    return 1.0 - extractable / MIGRATED_REAL_SOL


def graduation_progress_features(
    net_sol: float, cumulative_swaps: int
) -> dict[str, float]:
    """Curve-derived features, including trade-count efficiency.

    ``sol_per_swap`` is the strongest published single predictor of graduation:
    reaching a given point on the curve in *fewer, larger* trades indicates
    genuine capital commitment, whereas reaching the same point via a swarm of
    tiny trades indicates bot churn that does not persist.
    """
    state = state_from_net_sol(net_sol)
    swaps = max(1, cumulative_swaps)
    return {
        "curve_progress": state.progress,
        "curve_net_sol": state.net_sol_bought,
        "curve_multiple": state.multiple_from_launch,
        # Capital committed per trade — high is good.
        "curve_sol_per_swap": state.net_sol_bought / swaps,
        # How much of the theoretical pre-graduation ceiling has been used up.
        "curve_headroom": max(0.0, 1.0 - state.multiple_from_launch / MAX_PREGRAD_MULTIPLE),
        "curve_breakeven_multiple": breakeven_price_multiple(net_sol),
        # Probability of graduation this entry point needs in order to break
        # even. Compare against a model's calibrated P(graduate).
        "curve_breakeven_p_grad": breakeven_graduation_probability(net_sol),
    }


def estimate_net_sol_from_marketcap(market_cap_sol: float) -> float:
    """Invert market cap back to net SOL bought.

    Useful because most data providers report FDV or market cap rather than the
    curve's internal reserves.
    """
    if market_cap_sol <= 0:
        return 0.0
    v_sol0 = INITIAL_VIRTUAL_SOL_RESERVES / LAMPORTS_PER_SOL
    v_tok0 = INITIAL_VIRTUAL_TOKEN_RESERVES / TOKEN_DECIMALS
    k = v_sol0 * v_tok0
    supply = TOTAL_SUPPLY / TOKEN_DECIMALS
    # market_cap = price * supply = (vSol / (k/vSol)) * supply = vSol^2 * supply / k
    v_sol = math.sqrt(max(0.0, market_cap_sol) * k / supply)
    return max(0.0, v_sol - v_sol0)
