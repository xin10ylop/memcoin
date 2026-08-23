"""Deployer reputation — the strongest pre-trade signal available.

Published analysis of 655,770 pump.fun tokens puts the platform-wide graduation
rate at **0.63%**, while a small set of elite deployers bond at **40–71%**. That
is a 20–100x lift in prior probability, and unlike anything derived from order
flow it is available *before the token has traded at all*.

The same analysis found **no tradeable predictive signal at t=0** from the token
itself. Both facts together resolve the strategy: the edge is not in reacting to
a launch faster than anyone else — over half of pump.fun tokens are bought by
deployer-funded wallets inside the creation block, a race that is unwinnable by
construction rather than by insufficient hardware. The edge is in knowing *whose*
launch it is.

The statistical difficulty is that a 0.63% base rate makes small samples
worthless. A deployer with one graduation from three launches shows a raw rate of
33% — a 53x apparent lift that is almost certainly noise. This module therefore
never uses raw rates. It applies a Beta-Binomial posterior anchored on the
platform base rate, so a deployer must accumulate real evidence before the score
moves, and reports a conservative lower credible bound alongside the mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

#: Platform-wide pump.fun graduation rate (655,770 tokens observed).
PLATFORM_GRADUATION_RATE = 0.0063

#: Strength of the prior, in pseudo-launches. At 60, a deployer needs roughly
#: a dozen launches before their own record meaningfully moves the estimate.
#: Set deliberately high: with a 0.63% base rate, the cost of promoting a lucky
#: deployer is far greater than the cost of being slow to recognise a good one.
PRIOR_STRENGTH = 60.0


@dataclass
class DeployerScore:
    """Reputation of a single deployer wallet."""

    wallet: str
    launches: int = 0
    graduations: int = 0
    total_dev_buy_sol: float = 0.0

    #: Posterior mean graduation probability, shrunk toward the platform rate.
    posterior_rate: float = PLATFORM_GRADUATION_RATE
    #: Conservative lower bound of the credible interval.
    lower_bound: float = 0.0
    #: Posterior mean divided by the platform base rate.
    lift: float = 1.0
    tier: str = "unknown"
    notes: list[str] = field(default_factory=list)

    @property
    def raw_rate(self) -> float:
        """Unshrunk rate. Reported for context only — never used for decisions."""
        return self.graduations / self.launches if self.launches else 0.0

    @property
    def is_elite(self) -> bool:
        return self.tier == "elite"

    @property
    def is_factory(self) -> bool:
        """A high-volume deployer with no graduations: a launch factory."""
        return self.tier == "factory"

    def as_features(self) -> dict[str, float]:
        return {
            "dev_launches": float(self.launches),
            "dev_graduations": float(self.graduations),
            "dev_posterior_rate": self.posterior_rate,
            "dev_lower_bound": self.lower_bound,
            "dev_lift": min(self.lift, 200.0),
            "dev_is_elite": float(self.is_elite),
            "dev_is_factory": float(self.is_factory),
            "dev_avg_buy_sol": self.total_dev_buy_sol / self.launches if self.launches else 0.0,
        }


def score_deployer(
    wallet: str,
    launches: int,
    graduations: int,
    *,
    total_dev_buy_sol: float = 0.0,
    base_rate: float = PLATFORM_GRADUATION_RATE,
    prior_strength: float = PRIOR_STRENGTH,
) -> DeployerScore:
    """Beta-Binomial reputation for one deployer.

    The prior is ``Beta(base_rate * k, (1 - base_rate) * k)`` with ``k`` pseudo-
    launches, so the posterior mean is

        (graduations + base_rate * k) / (launches + k)

    which equals the platform rate when we know nothing and converges to the
    deployer's own rate only once they have a substantial record.
    """
    score = DeployerScore(
        wallet=wallet, launches=max(0, launches), graduations=max(0, graduations),
        total_dev_buy_sol=total_dev_buy_sol,
    )

    alpha0 = base_rate * prior_strength
    beta0 = (1.0 - base_rate) * prior_strength
    alpha = alpha0 + score.graduations
    beta = beta0 + max(0, score.launches - score.graduations)

    score.posterior_rate = alpha / (alpha + beta)
    score.lift = score.posterior_rate / base_rate if base_rate > 0 else 1.0

    # Normal approximation to the Beta posterior's lower 5% bound. Adequate here
    # because alpha + beta is never small — the prior guarantees it.
    total = alpha + beta
    variance = (alpha * beta) / (total * total * (total + 1.0))
    score.lower_bound = max(0.0, score.posterior_rate - 1.645 * math.sqrt(variance))

    score.tier, score.notes = _classify(score, base_rate)
    return score


def _classify(score: DeployerScore, base_rate: float) -> tuple[str, list[str]]:
    notes: list[str] = []

    if score.launches == 0:
        return "unknown", ["no launch history on record"]

    # Elite requires both a strong posterior AND enough launches for the lower
    # bound to have cleared the base rate by a wide margin.
    if score.launches >= 5 and score.lower_bound > base_rate * 5:
        notes.append(
            f"{score.graduations}/{score.launches} graduated; "
            f"lower bound {score.lower_bound:.2%} is {score.lower_bound / base_rate:.0f}x the platform rate"
        )
        return "elite", notes

    if score.launches >= 3 and score.lower_bound > base_rate * 2:
        notes.append(f"{score.graduations}/{score.launches} graduated — promising but thin")
        return "promising", notes

    # A high-volume deployer with nothing to show is running a factory.
    if score.launches >= 15 and score.graduations == 0:
        notes.append(f"{score.launches} launches, none graduated — launch factory")
        return "factory", notes

    if score.launches >= 8 and score.graduations == 0:
        notes.append(f"{score.launches} launches, none graduated")
        return "poor", notes

    if score.graduations > 0:
        notes.append(f"{score.graduations}/{score.launches} graduated — too few to judge")
    return "neutral", notes


class DeployerRegistry:
    """Deployer reputations backed by the local launch record.

    The record is accumulated first-hand from the launch stream rather than
    bought: every creation observed adds a launch, every migration adds a
    graduation. That makes it slow to bootstrap and impossible for a competitor
    to copy.
    """

    def __init__(self, store: Any, *, base_rate: float = PLATFORM_GRADUATION_RATE) -> None:
        self.store = store
        self.base_rate = base_rate
        self._cache: dict[str, DeployerScore] = {}

    def score(self, wallet: str, *, use_cache: bool = True) -> DeployerScore:
        if not wallet:
            return DeployerScore(wallet="", notes=["no deployer wallet"])
        if use_cache and wallet in self._cache:
            return self._cache[wallet]
        record = self.store.dev_record(wallet) or {}
        score = score_deployer(
            wallet,
            int(record.get("launches", 0) or 0),
            int(record.get("graduations", 0) or 0),
            total_dev_buy_sol=float(record.get("total_dev_buy_sol", 0.0) or 0.0),
            base_rate=self.base_rate,
        )
        self._cache[wallet] = score
        return score

    def invalidate(self, wallet: str) -> None:
        self._cache.pop(wallet, None)

    def observed_base_rate(self) -> float:
        """Graduation rate measured from our own record.

        Falls back to the published platform rate until we have enough launches
        for our own estimate to be meaningful.
        """
        stats = self.store.launch_stats()
        launches = int(stats.get("launches", 0) or 0)
        if launches < 2_000:
            return self.base_rate
        return float(stats.get("graduation_rate", self.base_rate) or self.base_rate)

    def leaderboard(self, limit: int = 20, min_launches: int = 3) -> list[DeployerScore]:
        rows = self.store.conn.execute(
            """SELECT dev_wallet, launches, graduations, total_dev_buy_sol
               FROM dev_wallets WHERE launches >= ?
               ORDER BY graduations DESC, launches DESC LIMIT ?""",
            (min_launches, limit * 3),
        ).fetchall()
        scores = [
            score_deployer(
                r["dev_wallet"], r["launches"], r["graduations"],
                total_dev_buy_sol=r["total_dev_buy_sol"] or 0.0, base_rate=self.base_rate,
            )
            for r in rows
        ]
        scores.sort(key=lambda s: (s.lower_bound, s.launches), reverse=True)
        return scores[:limit]
