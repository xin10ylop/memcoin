"""RugCheck.xyz client.

RugCheck exposes a free, key-less endpoint that aggregates most of the on-chain
checks this system needs, including several that are expensive to compute
ourselves from a public RPC: holder concentration with insider flagging, LP
locker detection, and — most valuable — ``creatorTokens``, the list of other
tokens the same deployer has launched.

Deployer history is the strongest single rug signal available: a wallet that has
minted dozens of tokens is running a factory, and the prior of any individual
token from that factory surviving is very low.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alpha.http import HttpClient

log = logging.getLogger(__name__)

BASE = "https://api.rugcheck.xyz/v1"

# Wallets/programs that legitimately hold large balances and must not be
# counted as concentration risk.
BENIGN_OWNER_TYPES = frozenset({"AMM", "LOCKER", "BURN", "DEX", "MARKET"})


@dataclass(slots=True)
class Holder:
    address: str
    owner: str
    pct: float
    insider: bool


@dataclass(slots=True)
class RugCheckReport:
    """Normalised subset of the RugCheck report that the screener consumes."""

    mint: str
    ok: bool = False
    score: float = 0.0
    score_normalised: float = 0.0
    rugged: bool = False
    risks: list[dict[str, Any]] = field(default_factory=list)
    mint_authority: str | None = None
    freeze_authority: str | None = None
    update_authority: str | None = None
    metadata_mutable: bool = False
    token_program: str = ""
    creator: str = ""
    creator_token_count: int = 0
    creator_balance_pct: float = 0.0
    total_holders: int = 0
    total_lp_providers: int = 0
    total_market_liquidity: float = 0.0
    lp_locked_pct: float = 0.0
    transfer_fee_pct: float = 0.0
    top_holders: list[Holder] = field(default_factory=list)
    insiders_detected: int = 0
    launchpad: str = ""
    name: str = ""
    symbol: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def top_holder_pct(self) -> float:
        """Largest non-benign holder's share of supply, in percent."""
        return max((h.pct for h in self.non_benign_holders), default=0.0)

    @property
    def top10_pct(self) -> float:
        return sum(h.pct for h in self.non_benign_holders[:10])

    @property
    def insider_pct(self) -> float:
        return sum(h.pct for h in self.top_holders if h.insider)

    @property
    def non_benign_holders(self) -> list[Holder]:
        known = self.raw.get("knownAccounts") or {}

        def benign(h: Holder) -> bool:
            entry = known.get(h.owner) or known.get(h.address) or {}
            return str(entry.get("type", "")).upper() in BENIGN_OWNER_TYPES

        return [h for h in self.top_holders if not benign(h)]

    @property
    def risk_names(self) -> list[str]:
        return [str(r.get("name", "")) for r in self.risks]

    @property
    def has_danger_risk(self) -> bool:
        return any(str(r.get("level", "")).lower() in ("danger", "critical") for r in self.risks)

    @classmethod
    def from_api(cls, mint: str, d: dict[str, Any] | None) -> "RugCheckReport":
        if not d:
            return cls(mint=mint, ok=False)
        token = d.get("token") or {}
        meta = d.get("tokenMeta") or {}
        fee = d.get("transferFee") or {}
        holders = [
            Holder(
                address=str(h.get("address", "")),
                owner=str(h.get("owner", "")),
                pct=float(h.get("pct") or 0.0),
                insider=bool(h.get("insider")),
            )
            for h in (d.get("topHolders") or [])
        ]
        supply = float(token.get("supply") or 0) or 1.0
        creator_balance = float(d.get("creatorBalance") or 0)

        return cls(
            mint=mint,
            ok=True,
            score=float(d.get("score") or 0),
            score_normalised=float(d.get("score_normalised") or 0),
            rugged=bool(d.get("rugged")),
            risks=list(d.get("risks") or []),
            mint_authority=token.get("mintAuthority") or d.get("mintAuthority"),
            freeze_authority=token.get("freezeAuthority") or d.get("freezeAuthority"),
            update_authority=meta.get("updateAuthority"),
            metadata_mutable=bool(meta.get("mutable")),
            token_program=str(d.get("tokenProgram") or ""),
            creator=str(d.get("creator") or ""),
            creator_token_count=len(d.get("creatorTokens") or []),
            creator_balance_pct=100.0 * creator_balance / supply,
            total_holders=int(d.get("totalHolders") or 0),
            total_lp_providers=int(d.get("totalLPProviders") or 0),
            total_market_liquidity=float(d.get("totalMarketLiquidity") or 0.0),
            lp_locked_pct=float(d.get("lpLockedPct") or 0.0),
            transfer_fee_pct=float(fee.get("pct") or 0.0),
            top_holders=holders,
            insiders_detected=int(d.get("graphInsidersDetected") or 0),
            launchpad=str((d.get("launchpad") or {}).get("name", "") or d.get("deployPlatform", "")),
            name=str(meta.get("name") or (d.get("fileMeta") or {}).get("name") or ""),
            symbol=str(meta.get("symbol") or (d.get("fileMeta") or {}).get("symbol") or ""),
            raw=d,
        )


class RugCheckClient:
    """Client for the free RugCheck endpoints."""

    def __init__(self, http: HttpClient | None = None) -> None:
        # RugCheck is undocumented on limits; stay conservative.
        self.http = http or HttpClient(requests_per_minute=25.0, burst=4)
        self._cache: dict[str, RugCheckReport] = {}

    def report(self, mint: str, *, use_cache: bool = True) -> RugCheckReport:
        """Full report for a mint. Cached in-process: reports are near-static
        for the fields we care about (authorities, creator history)."""
        if use_cache and mint in self._cache:
            return self._cache[mint]
        data = self.http.get_json(f"{BASE}/tokens/{mint}/report")
        report = RugCheckReport.from_api(mint, data if isinstance(data, dict) else None)
        if report.ok:
            self._cache[mint] = report
        return report

    def summary(self, mint: str) -> dict[str, Any] | None:
        """Lighter endpoint — score and risks only."""
        data = self.http.get_json(f"{BASE}/tokens/{mint}/report/summary")
        return data if isinstance(data, dict) else None
