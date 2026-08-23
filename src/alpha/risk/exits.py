"""Exit rules beyond fixed barriers.

Fixed take-profit and stop-loss levels are necessary but not sufficient here.
The dominant way a memecoin position dies is a *dump*: a single, violent
negative return as a large holder exits, which blows through a percentage stop
at a far worse price than the stop implies. Published analysis finds that
**92% of pump.fun tokens with at least 30 swaps experience at least one dump**,
defined as a log-return shock beyond four robust standard deviations.

The detector below uses median absolute deviation rather than standard
deviation, for a specific reason: the standard deviation of a memecoin's returns
is itself dominated by the dumps we are trying to detect, so a σ-based threshold
inflates in exactly the situation where it must stay tight. MAD is resistant to
the outliers it is being used to find.

A second rule watches **liquidity** rather than price. Liquidity leaving a pool
is the mechanical precursor of a rug: the price has not moved yet, but the exit
door is narrowing, and a position that looks fine on price may already be
unsellable at size.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence

# Consistency constant making MAD a standard-deviation estimator for normal data.
MAD_TO_SIGMA = 1.4826


@dataclass
class DumpDetectorConfig:
    #: Shock threshold in robust standard deviations.
    sigma_threshold: float = 4.0
    #: Minimum observations before the detector is allowed to fire. Below this
    #: the MAD estimate is too noisy and would trigger on ordinary volatility.
    min_observations: int = 20
    #: Trailing window of returns used for the MAD estimate.
    window: int = 200
    #: Floor on the robust sigma, as a fraction. Without it, a token that has
    #: barely moved produces a near-zero MAD and then flags any tick as a dump.
    min_sigma: float = 0.02
    #: Liquidity withdrawal that constitutes an exit signal on its own.
    liquidity_drop_pct: float = 0.35


@dataclass
class DumpSignal:
    triggered: bool
    reason: str = ""
    z_score: float = 0.0
    last_return: float = 0.0
    robust_sigma: float = 0.0
    n_observations: int = 0


def robust_sigma(returns: Sequence[float], floor: float = 0.02) -> float:
    """Median-absolute-deviation estimate of return dispersion.

    Resistant to the very outliers it is used to detect, unlike the standard
    deviation, which those outliers inflate.
    """
    if len(returns) < 2:
        return floor
    median = statistics.median(returns)
    mad = statistics.median([abs(r - median) for r in returns])
    return max(mad * MAD_TO_SIGMA, floor)


def detect_dump(
    prices: Sequence[float], config: DumpDetectorConfig | None = None
) -> DumpSignal:
    """Flag a robust-outlier negative return in the most recent step.

    ``prices`` must be ascending in time. Only the latest return is tested; the
    prior window supplies the dispersion estimate.
    """
    cfg = config or DumpDetectorConfig()
    clean = [p for p in prices if p and p > 0]
    if len(clean) < max(3, cfg.min_observations):
        return DumpSignal(False, "insufficient history", n_observations=len(clean))

    log_returns = [math.log(b / a) for a, b in zip(clean, clean[1:])]
    window = log_returns[-cfg.window :]
    if len(window) < cfg.min_observations:
        return DumpSignal(False, "insufficient returns", n_observations=len(window))

    # Estimate dispersion from everything *except* the return under test, so a
    # large shock does not widen the very threshold meant to catch it.
    history, last = window[:-1], window[-1]
    sigma = robust_sigma(history, cfg.min_sigma)
    z = last / sigma if sigma > 0 else 0.0

    if last < 0 and z <= -cfg.sigma_threshold:
        return DumpSignal(
            triggered=True,
            reason=f"dump: {z:.1f}σ negative return ({100 * (math.exp(last) - 1):.1f}%)",
            z_score=z, last_return=last, robust_sigma=sigma, n_observations=len(window),
        )
    return DumpSignal(False, "no dump", z_score=z, last_return=last,
                      robust_sigma=sigma, n_observations=len(window))


def detect_liquidity_exit(
    liquidities: Sequence[float], config: DumpDetectorConfig | None = None
) -> DumpSignal:
    """Flag liquidity being withdrawn from the pool.

    Watches the drop from the observed peak rather than the last value, because
    withdrawals often happen in several steps and each individual step can look
    unremarkable.
    """
    cfg = config or DumpDetectorConfig()
    clean = [x for x in liquidities if x and x > 0]
    if len(clean) < 3:
        return DumpSignal(False, "insufficient history", n_observations=len(clean))
    peak, current = max(clean), clean[-1]
    drop = 1.0 - current / peak if peak > 0 else 0.0
    if drop >= cfg.liquidity_drop_pct:
        return DumpSignal(
            triggered=True,
            reason=f"liquidity down {drop:.0%} from peak (${peak:,.0f} → ${current:,.0f})",
            z_score=-drop, n_observations=len(clean),
        )
    return DumpSignal(False, "liquidity stable", z_score=-drop, n_observations=len(clean))


@dataclass
class ExitMonitor:
    """Tracks a position's price and liquidity path and reports exit signals.

    Held separately from :class:`~alpha.risk.portfolio.Position` because these
    rules need a *history*, whereas the barrier rules only need the current
    price. Keeping the stateful part separate stops the position object from
    quietly growing an unbounded buffer.
    """

    config: DumpDetectorConfig = field(default_factory=DumpDetectorConfig)
    prices: list[float] = field(default_factory=list)
    liquidities: list[float] = field(default_factory=list)
    max_history: int = 400

    def update(self, price: float, liquidity: float | None = None) -> None:
        if price and price > 0:
            self.prices.append(float(price))
            if len(self.prices) > self.max_history:
                del self.prices[: len(self.prices) - self.max_history]
        if liquidity and liquidity > 0:
            self.liquidities.append(float(liquidity))
            if len(self.liquidities) > self.max_history:
                del self.liquidities[: len(self.liquidities) - self.max_history]

    def check(self) -> DumpSignal:
        """Evaluate every path-dependent exit rule, liquidity first.

        Liquidity is checked before price because a pool being drained is the
        more urgent condition: price may still look healthy while the ability to
        exit at size has already gone.
        """
        liq = detect_liquidity_exit(self.liquidities, self.config)
        if liq.triggered:
            return liq
        return detect_dump(self.prices, self.config)
