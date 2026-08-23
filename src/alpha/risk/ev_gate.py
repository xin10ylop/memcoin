"""Expected-value gate for bonding-curve entries.

This is the system's sharpest decision rule, and it comes from combining two
facts that are each exact rather than estimated.

**The payoff is fixed by the curve.** Price on a pump.fun bonding curve is
proportional to the square of the virtual SOL reserve, so buying at ``vSol`` and
holding to graduation returns exactly ``(115.0054 / vSol)^2``. That fixes the
break-even probability at ``p* = vSol^2 / 115.0054^2`` — 6.80% at launch, 18.9%
at 50 SOL, 48.4% at 80 SOL.

**The base rate is far below it.** Published work puts platform-wide graduation
at 0.63% (655,770 tokens) to ~1.4% all-time; this system's own launch stream
measured 1.85%. Against a 6.80% break-even, an unconditional hold-to-graduation
bet is roughly five times short of viable — and it gets *worse* further up the
curve, not better, because the remaining multiple shrinks quadratically while
the required probability rises.

So the gate asks one question: **is there a specific reason to believe this
token's graduation probability exceeds the break-even for where it sits on the
curve?** Absent such a reason, the correct action is not to trade.

The only conditioning variable large enough to close a 5x gap is deployer
reputation: elite deployers graduate at 40-71% against a 0.63-2% base, a lift of
20-100x. That is why :mod:`alpha.features.deployer` exists and why its posterior
feeds directly into this gate.

Note this gate governs the *hold-to-graduation* thesis specifically. A momentum
trade that takes profit at +50% on the curve has a different payoff structure and
is evaluated by the barrier model instead; :meth:`EvGate.evaluate` reports both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha.data.bondingcurve import (
    GRADUATION_VSOL,
    PLATFORM_GRADUATION_RATE,
    breakeven_graduation_probability,
    graduation_edge,
    post_migration_dead_liquidity,
    state_from_net_sol,
)


@dataclass
class EvVerdict:
    """Outcome of evaluating one entry against its break-even."""

    approved: bool
    net_sol: float
    breakeven_p: float
    estimated_p: float
    edge: float
    payoff_multiple: float
    margin: float
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = "APPROVE" if self.approved else "REJECT"
        return (
            f"{head} net_sol={self.net_sol:.1f} needs p>{self.breakeven_p:.1%} "
            f"has p={self.estimated_p:.1%} edge={self.edge:+.0%}"
        )

    def as_features(self) -> dict[str, float]:
        return {
            "ev_breakeven_p": self.breakeven_p,
            "ev_estimated_p": self.estimated_p,
            "ev_edge": self.edge,
            "ev_margin": self.margin,
            "ev_payoff_multiple": self.payoff_multiple,
        }


@dataclass
class EvGateConfig:
    #: Require the estimated probability to exceed break-even by this ratio, not
    #: merely to match it. The probability estimate is noisy and the payoff is
    #: all-or-nothing, so trading at exactly break-even is a coin flip with
    #: execution costs on top.
    safety_margin: float = 1.5
    #: Refuse entries this far up the curve regardless of edge: the remaining
    #: multiple is small, the required probability is high, and the depth cut at
    #: graduation makes the exit worse.
    max_net_sol: float = 55.0
    #: Fall back to this when no deployer history exists.
    fallback_p: float = PLATFORM_GRADUATION_RATE
    #: Whether a token may be held through graduation. The graduated cohort is
    #: negative-sum by construction (~21% of migrated SOL is unextractable), so
    #: the default is to exit on the curve.
    allow_hold_through_migration: bool = False


class EvGate:
    """Decides whether a hold-to-graduation entry is justified."""

    def __init__(self, config: EvGateConfig | None = None) -> None:
        self.cfg = config or EvGateConfig()

    def evaluate(
        self,
        *,
        net_sol: float,
        p_graduate: float | None = None,
        deployer_score: Any | None = None,
    ) -> EvVerdict:
        """Evaluate an entry at ``net_sol`` given a graduation probability.

        ``p_graduate`` may be supplied directly, or derived from a
        :class:`~alpha.features.deployer.DeployerScore`. When a deployer score is
        used, its **lower confidence bound** is preferred over its point estimate:
        with thousands of deployers, some post a short perfect record by chance,
        and sizing an all-or-nothing bet off an unshrunk point estimate is how
        that noise becomes a loss.
        """
        cfg = self.cfg
        reasons: list[str] = []

        estimated = self._resolve_probability(p_graduate, deployer_score, reasons)
        breakeven = breakeven_graduation_probability(net_sol)
        payoff = (1.0 / breakeven) if breakeven > 0 else float("inf")
        edge = graduation_edge(net_sol, estimated)
        margin = estimated / breakeven if breakeven > 0 else 0.0

        approved = True
        if net_sol > cfg.max_net_sol:
            approved = False
            state = state_from_net_sol(net_sol)
            reasons.append(
                f"too far up the curve: {net_sol:.0f} SOL raised "
                f"({state.progress:.0%} to graduation), only {payoff:.1f}x left "
                f"and {breakeven:.0%} probability required"
            )
        if margin < cfg.safety_margin:
            approved = False
            reasons.append(
                f"estimated P(graduate)={estimated:.2%} does not clear break-even "
                f"{breakeven:.2%} by the required {cfg.safety_margin:.1f}x margin "
                f"(actual {margin:.2f}x)"
            )
        if approved:
            reasons.append(
                f"P(graduate)={estimated:.1%} is {margin:.1f}x the {breakeven:.1%} "
                f"break-even; payoff {payoff:.1f}x, edge {edge:+.0%}"
            )

        return EvVerdict(
            approved=approved, net_sol=net_sol, breakeven_p=breakeven,
            estimated_p=estimated, edge=edge, payoff_multiple=payoff,
            margin=margin, reasons=reasons,
        )

    def _resolve_probability(
        self, p_graduate: float | None, deployer_score: Any | None, reasons: list[str]
    ) -> float:
        if p_graduate is not None:
            return max(0.0, min(1.0, p_graduate))
        if deployer_score is not None:
            # Prefer the lower bound: an all-or-nothing bet sized off an
            # optimistic point estimate is where short lucky records do damage.
            lower = getattr(deployer_score, "lower_bound", None)
            if lower is not None:
                launches = getattr(deployer_score, "launches", 0)
                reasons.append(
                    f"deployer has {launches} prior launches, "
                    f"{getattr(deployer_score, 'graduations', 0)} graduations"
                )
                return max(0.0, min(1.0, float(lower)))
        reasons.append("no deployer history — using platform base rate")
        return self.cfg.fallback_p

    def exit_recommendation(self, net_sol: float) -> str:
        """Where to exit, given the depth cut at graduation."""
        if self.cfg.allow_hold_through_migration:
            return "hold_through_migration"
        state = state_from_net_sol(net_sol)
        if state.virtual_sol >= GRADUATION_VSOL * 0.97:
            return (
                f"exit on the curve now: graduation moves ~{post_migration_dead_liquidity():.0%} "
                "of migrated SOL beyond reach of holders collectively, and pool depth "
                "drops ~26% versus the curve"
            )
        return "exit on the curve before graduation"
