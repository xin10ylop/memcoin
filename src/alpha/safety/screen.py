"""Safety screening — the hard-reject layer.

This runs *before* any alpha model sees a token, and it is the most important
component in the system. The dominant risk in memecoin trading is not picking a
token that underperforms; it is picking a token you cannot sell, or one whose
issuer can mint infinite supply into your bid. Those are ‑100% outcomes that no
amount of upside capture compensates for.

The design principle is asymmetry: rejecting a good token costs one missed
opportunity out of roughly twenty thousand launched per day, while accepting a
malicious one costs the entire position. Every threshold here is therefore set
to be aggressive, and every check defaults to *reject* when its data source is
unavailable rather than assuming the token is fine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from alpha.safety.rugcheck import RugCheckClient, RugCheckReport
from alpha.safety.solana_rpc import MintInfo, SolanaRpc

log = logging.getLogger(__name__)


class Severity(IntEnum):
    """How badly a check failed. ``FATAL`` is an unconditional reject."""

    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FATAL = 4


class Verdict(IntEnum):
    """Outcome of screening.

    The distinction between ``REJECT`` and ``IMMATURE`` is operationally
    important. A token that fails on holder count or liquidity has not done
    anything wrong — it is simply young, and those conditions routinely resolve
    within minutes as a launch gathers participants. A token whose deployer can
    mint fresh supply will never become safe. Collapsing both into a single
    "rejected" bucket would throw away most of the tradeable universe, since
    nearly every token is immature at the moment we first see it.
    """

    PASS = 0
    IMMATURE = 1   # may become tradeable; re-screen later
    REJECT = 2     # structurally unsafe; never buy


# RugCheck assigns "danger" to several conditions that are merely maturity
# states for a new bonding-curve launch: liquidity is thin and LP is unlocked
# by design before a token migrates. Rather than trusting their level field, we
# map risk names onto our own model and let our numeric thresholds govern the
# quantitative checks.
STRUCTURAL_RUGCHECK_RISKS = frozenset(
    {
        "Creator history of rugged tokens",
        "Freeze Authority still enabled",
        "Mint Authority still enabled",
        "Honeypot",
    }
)

# Real signals, but not disqualifying on their own — they feed the risk score.
ADVISORY_RUGCHECK_RISKS = frozenset(
    {
        "Copycat token",
        "High holder correlation",
        "Mutable metadata",
        "Low amount of LP Providers",
        "Large Amount of LP Unlocked",
    }
)

# Purely maturity-related; handled by our own liquidity/holder thresholds.
MATURITY_RUGCHECK_RISKS = frozenset(
    {"Low Liquidity", "Low Amount of holders"}
)

# Concentration risks RugCheck names, which our own numeric thresholds already
# evaluate directly from ``topHolders``. Listing them here prevents the same
# condition from rejecting a token twice under two different names.
_QUANTITATIVE_RUGCHECK_RISKS = frozenset(
    {
        "Single holder ownership",
        "High ownership",
        "Top 10 holders high ownership",
        "High holder concentration",
    }
)


@dataclass(slots=True)
class SafetyCheck:
    """One screening result.

    ``structural`` marks a check whose failure reflects a permanent property of
    the token (authorities, deployer history) rather than a transient one
    (liquidity, holder count). Only structural failures cause a hard reject.
    """

    name: str
    passed: bool
    severity: Severity
    detail: str
    value: Any = None
    structural: bool = True

    def __str__(self) -> str:
        mark = "PASS" if self.passed else self.severity.name
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class SafetyThresholds:
    """Tunable limits. Defaults are deliberately conservative."""

    # A single wallet holding this much can dump the whole market on us. The
    # bonding-curve/AMM account itself is excluded via knownAccounts.
    max_single_holder_pct: float = 25.0
    max_top10_holder_pct: float = 70.0
    # Wallets RugCheck's graph analysis flags as connected to the deployer.
    max_insider_pct: float = 25.0
    # A deployer with a long token history is running a launch factory.
    max_creator_tokens: int = 15
    max_creator_balance_pct: float = 15.0
    # Below this there is no market: exiting costs more than the position.
    min_liquidity_usd: float = 2_500.0
    min_holders: int = 25
    # A transfer fee is a tax on every exit; anything material is disqualifying.
    max_transfer_fee_bps: int = 100  # 1%
    # RugCheck's own normalised score. Higher is riskier.
    max_rugcheck_score: float = 40_000.0
    # Reject if the deployer still controls metadata (can rewrite name/image
    # to impersonate a different project after we buy).
    reject_mutable_metadata: bool = False
    require_rugcheck: bool = True
    require_mint_info: bool = True


@dataclass
class SafetyReport:
    """Aggregated result of screening one token."""

    mint: str
    checks: list[SafetyCheck] = field(default_factory=list)
    mint_info: MintInfo | None = None
    rugcheck: RugCheckReport | None = None
    error: str | None = None

    def add(
        self,
        name: str,
        passed: bool,
        severity: Severity,
        detail: str,
        value: Any = None,
        *,
        structural: bool = True,
    ) -> None:
        self.checks.append(SafetyCheck(name, passed, severity, detail, value, structural))

    @property
    def failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if not c.passed]

    @property
    def fatal(self) -> list[SafetyCheck]:
        return [c for c in self.failures if c.severity == Severity.FATAL]

    @property
    def blocking(self) -> list[SafetyCheck]:
        """Fatal failures that are structural — these are permanent rejects."""
        return [c for c in self.fatal if c.structural]

    @property
    def immaturity(self) -> list[SafetyCheck]:
        """Fatal failures that are merely maturity conditions."""
        return [c for c in self.fatal if not c.structural]

    @property
    def verdict(self) -> Verdict:
        if self.error or self.blocking:
            return Verdict.REJECT
        if self.immaturity:
            return Verdict.IMMATURE
        return Verdict.PASS

    @property
    def passed(self) -> bool:
        """True only if the token is tradeable right now."""
        return self.verdict == Verdict.PASS

    @property
    def retryable(self) -> bool:
        """True if the token is not unsafe, just not yet ready."""
        return self.verdict == Verdict.IMMATURE

    @property
    def risk_score(self) -> float:
        """0–100 continuous risk measure for ranking among *passing* tokens.

        This is intentionally separate from the pass/fail decision: hard rejects
        are binary, but among survivors we still prefer the cleaner token.
        """
        if not self.passed:
            return 100.0
        weights = {Severity.LOW: 3.0, Severity.MEDIUM: 8.0, Severity.HIGH: 18.0}
        score = sum(weights.get(c.severity, 0.0) for c in self.failures)
        return min(100.0, score)

    def summary(self) -> str:
        if self.error:
            return f"{self.mint[:12]}… ERROR: {self.error}"
        reasons = "; ".join(c.name for c in self.failures) or "clean"
        return f"{self.mint[:12]}… {self.verdict.name} risk={self.risk_score:.0f} — {reasons}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mint": self.mint,
            "verdict": self.verdict.name,
            "passed": self.passed,
            "risk_score": self.risk_score,
            "error": self.error,
            "failures": [
                {
                    "name": c.name,
                    "severity": c.severity.name,
                    "detail": c.detail,
                    "value": c.value,
                    "structural": c.structural,
                }
                for c in self.failures
            ],
        }


class SafetyScreener:
    """Runs the full battery of safety checks against a mint."""

    def __init__(
        self,
        rugcheck: RugCheckClient | None = None,
        rpc: SolanaRpc | None = None,
        thresholds: SafetyThresholds | None = None,
    ) -> None:
        self.rugcheck = rugcheck or RugCheckClient()
        self.rpc = rpc or SolanaRpc()
        self.t = thresholds or SafetyThresholds()

    def screen(self, mint: str, *, liquidity_usd: float | None = None) -> SafetyReport:
        """Screen one token. ``liquidity_usd`` from a pool snapshot is preferred
        over RugCheck's own figure, which lags."""
        report = SafetyReport(mint=mint)
        try:
            info = self.rpc.get_mint_info(mint)
            report.mint_info = info
            rc = self.rugcheck.report(mint)
            report.rugcheck = rc
        except Exception as exc:
            report.error = f"screening failed: {exc}"
            log.warning("safety screen error for %s: %s", mint, exc)
            return report

        self._check_authorities(report, info)
        self._check_extensions(report, info)
        self._check_rugcheck_verdict(report, rc)
        self._check_concentration(report, rc)
        self._check_creator(report, rc)
        self._check_liquidity(report, rc, liquidity_usd)
        self._check_metadata(report, rc, info)
        return report

    # --------------------------------------------------------------- checks

    def _check_authorities(self, r: SafetyReport, info: MintInfo) -> None:
        if not info.ok:
            # Fail closed: an unreadable mint is not a safe mint.
            r.add(
                "mint_readable",
                not self.t.require_mint_info,
                Severity.FATAL if self.t.require_mint_info else Severity.MEDIUM,
                "could not read mint account from RPC",
            )
            return
        r.add("mint_readable", True, Severity.INFO, "mint account parsed")
        r.add(
            "mint_authority_revoked",
            not info.can_mint_more,
            Severity.FATAL,
            f"mint authority is {info.mint_authority} — supply can be inflated"
            if info.can_mint_more else "mint authority revoked",
            info.mint_authority,
        )
        r.add(
            "freeze_authority_revoked",
            not info.can_freeze,
            Severity.FATAL,
            f"freeze authority is {info.freeze_authority} — our account can be frozen"
            if info.can_freeze else "freeze authority revoked",
            info.freeze_authority,
        )

    def _check_extensions(self, r: SafetyReport, info: MintInfo) -> None:
        if not info.ok:
            return
        bad = info.dangerous_extensions
        r.add(
            "no_dangerous_extensions",
            not bad,
            Severity.FATAL,
            f"Token-2022 extensions allow issuer control: {bad}" if bad else "no dangerous extensions",
            bad,
        )
        fee_bps = info.transfer_fee_bps
        r.add(
            "transfer_fee_acceptable",
            fee_bps <= self.t.max_transfer_fee_bps,
            Severity.FATAL,
            f"transfer fee {fee_bps}bps exceeds limit {self.t.max_transfer_fee_bps}bps",
            fee_bps,
        )

    def _check_rugcheck_verdict(self, r: SafetyReport, rc: RugCheckReport) -> None:
        if not rc.ok:
            r.add(
                "rugcheck_available",
                not self.t.require_rugcheck,
                Severity.FATAL if self.t.require_rugcheck else Severity.MEDIUM,
                "RugCheck report unavailable",
            )
            return
        r.add("rugcheck_available", True, Severity.INFO, "report retrieved")
        r.add("not_flagged_rugged", not rc.rugged, Severity.FATAL, "RugCheck flags this token as rugged")

        names = set(rc.risk_names)
        structural_hits = sorted(names & STRUCTURAL_RUGCHECK_RISKS)
        r.add(
            "no_structural_risks",
            not structural_hits,
            Severity.FATAL,
            f"structural risks: {structural_hits}" if structural_hits else "no structural risks",
            structural_hits,
        )
        advisory_hits = sorted(names & ADVISORY_RUGCHECK_RISKS)
        r.add(
            "no_advisory_risks",
            not advisory_hits,
            Severity.MEDIUM,
            f"advisory risks: {advisory_hits}" if advisory_hits else "no advisory risks",
            advisory_hits,
        )
        # Anything RugCheck flags as danger that we have not explicitly
        # classified is treated as structural: unknown risks default to unsafe.
        unknown_danger = sorted(
            {
                str(risk.get("name", ""))
                for risk in rc.risks
                if str(risk.get("level", "")).lower() in ("danger", "critical")
            }
            - STRUCTURAL_RUGCHECK_RISKS
            - ADVISORY_RUGCHECK_RISKS
            - MATURITY_RUGCHECK_RISKS
            - _QUANTITATIVE_RUGCHECK_RISKS
        )
        r.add(
            "no_unclassified_danger",
            not unknown_danger,
            Severity.FATAL,
            f"unrecognised danger risks: {unknown_danger}" if unknown_danger else "none",
            unknown_danger,
        )
        r.add(
            "rugcheck_score",
            rc.score_normalised <= self.t.max_rugcheck_score,
            Severity.HIGH,
            f"normalised score {rc.score_normalised:.0f} > {self.t.max_rugcheck_score:.0f}",
            rc.score_normalised,
        )

    def _check_concentration(self, r: SafetyReport, rc: RugCheckReport) -> None:
        if not rc.ok:
            return
        top = rc.top_holder_pct
        r.add(
            "single_holder_concentration",
            top <= self.t.max_single_holder_pct,
            Severity.FATAL,
            f"largest non-pool holder owns {top:.1f}% (limit {self.t.max_single_holder_pct}%)",
            round(top, 2),
        )
        top10 = rc.top10_pct
        r.add(
            "top10_concentration",
            top10 <= self.t.max_top10_holder_pct,
            Severity.HIGH,
            f"top-10 own {top10:.1f}% (limit {self.t.max_top10_holder_pct}%)",
            round(top10, 2),
        )
        ins = rc.insider_pct
        r.add(
            "insider_concentration",
            ins <= self.t.max_insider_pct,
            Severity.FATAL,
            f"wallets flagged as insiders own {ins:.1f}% (limit {self.t.max_insider_pct}%)",
            round(ins, 2),
        )
        # Holder count is a maturity condition, not a safety property: a token
        # five minutes old legitimately has few holders.
        r.add(
            "holder_count",
            rc.total_holders >= self.t.min_holders,
            Severity.FATAL,
            f"only {rc.total_holders} holders (min {self.t.min_holders})",
            rc.total_holders,
            structural=False,
        )

    def _check_creator(self, r: SafetyReport, rc: RugCheckReport) -> None:
        if not rc.ok:
            return
        # A deployer with many prior tokens is running a launch factory. Each
        # individual launch from such a wallet has a very low survival prior.
        n = rc.creator_token_count
        r.add(
            "creator_not_serial_deployer",
            n <= self.t.max_creator_tokens,
            Severity.FATAL,
            f"deployer has launched {n} tokens (limit {self.t.max_creator_tokens})",
            n,
        )
        r.add(
            "creator_holdings",
            rc.creator_balance_pct <= self.t.max_creator_balance_pct,
            Severity.HIGH,
            f"deployer still holds {rc.creator_balance_pct:.1f}% of supply",
            round(rc.creator_balance_pct, 3),
        )

    def _check_liquidity(self, r: SafetyReport, rc: RugCheckReport, liquidity_usd: float | None) -> None:
        liq = liquidity_usd if liquidity_usd is not None else (rc.total_market_liquidity if rc.ok else 0.0)
        # Liquidity is likewise a maturity condition — it grows as a launch
        # gathers buyers — but it is a hard gate on entry because exiting an
        # illiquid pool costs more than the position is worth.
        r.add(
            "min_liquidity",
            liq >= self.t.min_liquidity_usd,
            Severity.FATAL,
            f"liquidity ${liq:,.0f} below floor ${self.t.min_liquidity_usd:,.0f}",
            round(liq, 2),
            structural=False,
        )

    def _check_metadata(self, r: SafetyReport, rc: RugCheckReport, info: MintInfo) -> None:
        if not rc.ok:
            return
        mutable = rc.metadata_mutable
        r.add(
            "metadata_immutable",
            (not mutable) or (not self.t.reject_mutable_metadata),
            Severity.HIGH if not self.t.reject_mutable_metadata else Severity.FATAL,
            "metadata is mutable — name/image can be rewritten after purchase"
            if mutable else "metadata immutable",
            mutable,
        )
