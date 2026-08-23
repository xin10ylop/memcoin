"""Real-time launch scanner — the low-latency decision path.

The polling collector sees pools at 1–15 minutes old. Measurement on this panel
showed that is already too late for anything above roughly 5x: from a two-minute
entry the best outcome observed was 7.3x, from five minutes 4.6x, against 318x
from the launch candle. This scanner sits on the PumpPortal websocket instead and
decides within seconds of deployment.

It deliberately does **not** try to win block zero. Over half of pump.fun tokens
are sniped in their own creation block by a bundled create-and-buy, and
competitive snipers run sub-60ms detect-to-submit against roughly 800ms for a
public-RPC path. Those races are lost before they start. What is reachable is the
first 30–60 seconds, which is where the measurements still show 5–10x outcomes.

The decision itself is deliberately austere, because the arithmetic is austere.
An unconditional hold-to-graduation bet is negative expected value at every point
on the curve — break-even at launch is 6.80% against a 0.63–1.85% base rate. So
the scanner's default answer is *no*, and a launch has to present a specific,
quantified reason to overturn it. In practice that reason is deployer reputation,
the only conditioning variable with enough lift (40–71% for elite deployers) to
close a fivefold gap.

Expect it to fire rarely. At the strictest published threshold there are only a
few dozen elite deployers on the entire platform. Firing rarely is the design,
not a defect: the base rate says almost every launch is a losing bet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from alpha.data.bondingcurve import PLATFORM_GRADUATION_RATE
from alpha.data.pumpportal import Launch
from alpha.risk.ev_gate import EvGate, EvGateConfig, EvVerdict

log = logging.getLogger(__name__)


@dataclass
class ScannerConfig:
    """Filters applied before the expected-value gate.

    These are cheap rejections that avoid spending a deployer lookup, and each
    encodes a measured fact rather than a preference.
    """

    #: A deployer who buys none of their own token has no capital at risk and no
    #: reason to support it. A deployer who buys too much holds a position they
    #: can dump into any rally we participate in.
    min_dev_buy_sol: float = 0.05
    max_dev_buy_sol: float = 8.0
    #: Beyond this the deployer's own allocation is a large share of the entire
    #: run to graduation, bought at the lowest prices on the curve.
    max_dev_curve_share: float = 0.12
    #: Reject launches already well up the curve at the moment of creation:
    #: a large same-block buy means the cheap part of the curve is gone.
    max_net_sol_at_creation: float = 12.0
    #: Require a metadata URI. Its absence means no name, image or socials, and
    #: static social presence is one of the few free pre-trade signals.
    require_metadata: bool = True
    #: A deployer needs at least this many prior launches before their record is
    #: treated as evidence rather than noise.
    min_deployer_history: int = 3
    #: Emit at most this many signals per hour, whatever the gate says.
    max_signals_per_hour: int = 12


@dataclass
class ScanResult:
    """What the scanner decided about one launch, and why."""

    launch: Launch
    approved: bool
    verdict: EvVerdict | None = None
    deployer: Any | None = None
    rejected_by: str = ""
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        tag = "SIGNAL" if self.approved else "pass"
        return (
            f"[{tag}] {self.launch.symbol[:14]:14} dev_buy={self.launch.dev_buy_sol:>5.2f} SOL "
            f"{'— ' + self.rejected_by if self.rejected_by else ''}"
        )


@dataclass
class ScannerStats:
    seen: int = 0
    signalled: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "signalled": self.signalled,
            "signal_rate": round(self.signalled / self.seen, 5) if self.seen else 0.0,
            "top_rejections": dict(
                sorted(self.rejections.items(), key=lambda kv: -kv[1])[:8]
            ),
        }


class LaunchScanner:
    """Applies deployer reputation and the EV gate to live launches."""

    def __init__(
        self,
        *,
        deployer_registry: Any | None = None,
        gate: EvGate | None = None,
        config: ScannerConfig | None = None,
        on_signal: Callable[[ScanResult], None] | None = None,
    ) -> None:
        self.registry = deployer_registry
        self.gate = gate or EvGate(EvGateConfig())
        self.cfg = config or ScannerConfig()
        self.on_signal = on_signal
        self.stats = ScannerStats()
        self._recent_signal_times: list[datetime] = []

    def scan(self, launch: Launch) -> ScanResult:
        """Evaluate one launch. Returns a result whether or not it approved."""
        self.stats.seen += 1
        cfg = self.cfg

        # --- cheap structural filters, before any lookup -------------------
        if cfg.require_metadata and not launch.uri:
            return self._reject(launch, "no metadata URI")

        if launch.dev_buy_sol < cfg.min_dev_buy_sol:
            return self._reject(launch, "deployer bought none of their own token")

        if launch.dev_buy_sol > cfg.max_dev_buy_sol:
            return self._reject(
                launch, f"deployer allocation too large ({launch.dev_buy_sol:.1f} SOL)"
            )

        if launch.dev_curve_share > cfg.max_dev_curve_share:
            return self._reject(
                launch,
                f"deployer took {launch.dev_curve_share:.0%} of the run to graduation",
            )

        if launch.net_sol > cfg.max_net_sol_at_creation:
            return self._reject(
                launch, f"already {launch.net_sol:.0f} SOL up the curve at creation"
            )

        # --- deployer reputation -------------------------------------------
        score = None
        if self.registry is not None and launch.dev_wallet:
            try:
                score = self.registry.score(launch.dev_wallet)
            except Exception as exc:
                log.debug("deployer lookup failed for %s: %s", launch.dev_wallet[:12], exc)

        if score is not None and getattr(score, "launches", 0) < cfg.min_deployer_history:
            return self._reject(
                launch,
                f"deployer has only {getattr(score, 'launches', 0)} prior launches "
                f"(need {cfg.min_deployer_history})",
            )

        # --- expected-value gate --------------------------------------------
        verdict = self.gate.evaluate(net_sol=launch.net_sol, deployer_score=score)
        if not verdict.approved:
            return self._reject(launch, "expected value below break-even", verdict, score)

        if not self._within_rate_limit():
            return self._reject(launch, "signal rate limit reached", verdict, score)

        result = ScanResult(
            launch=launch, approved=True, verdict=verdict, deployer=score,
            reasons=list(verdict.reasons),
        )
        self.stats.signalled += 1
        self._recent_signal_times.append(datetime.now(timezone.utc))
        log.info(
            "SIGNAL %s dev=%s P(grad)=%.1f%% vs breakeven %.1f%% edge=%+.0f%%",
            launch.symbol[:16], launch.dev_wallet[:12],
            100 * verdict.estimated_p, 100 * verdict.breakeven_p, 100 * verdict.edge,
        )
        if self.on_signal:
            self.on_signal(result)
        return result

    def _reject(
        self,
        launch: Launch,
        reason: str,
        verdict: EvVerdict | None = None,
        score: Any | None = None,
    ) -> ScanResult:
        self.stats.reject(reason)
        return ScanResult(
            launch=launch, approved=False, verdict=verdict, deployer=score,
            rejected_by=reason,
        )

    def _within_rate_limit(self) -> bool:
        """Cap signals per hour so a registry bug cannot flood the book."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 3600
        self._recent_signal_times = [
            t for t in self._recent_signal_times if t.timestamp() > cutoff
        ]
        return len(self._recent_signal_times) < self.cfg.max_signals_per_hour

    def expected_signal_rate(self, launches_per_day: int = 9600) -> dict[str, Any]:
        """Rough expectation of how often this fires, for sanity-checking.

        Firing rarely is correct. If the scanner starts approving a meaningful
        fraction of launches, the gate or the deployer registry is broken, since
        the platform-wide graduation rate is under 2%.
        """
        rate = self.stats.signalled / self.stats.seen if self.stats.seen else 0.0
        return {
            "observed_signal_rate": round(rate, 6),
            "implied_signals_per_day": round(rate * launches_per_day, 2),
            "platform_graduation_rate": PLATFORM_GRADUATION_RATE,
            "sane": rate <= 0.02,
        }
