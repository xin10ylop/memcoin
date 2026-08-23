"""Minimal Solana JSON-RPC client for token safety checks.

Only the handful of read methods the safety layer needs are implemented, so the
system has no dependency on the (heavy) official SDK in its default paper-trading
mode. The public ``api.mainnet-beta.solana.com`` endpoint works for light calls
but aggressively rate-limits heavier ones such as ``getTokenLargestAccounts``;
supply a dedicated RPC URL for production use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alpha.http import HttpClient

log = logging.getLogger(__name__)

PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# Token-2022 extensions that let an issuer seize, freeze, tax or block transfers.
# Any of these on a memecoin is disqualifying: they make a position unsellable
# at the issuer's discretion, which is a total loss regardless of price.
DANGEROUS_EXTENSIONS = frozenset(
    {
        "permanentDelegate",        # issuer can transfer your tokens away at will
        "nonTransferable",          # soulbound: you can never sell
        "transferHook",             # arbitrary program gates every transfer
        "defaultAccountState",      # new token accounts can default to frozen
        "confidentialTransferMint",
    }
)


@dataclass(slots=True)
class MintInfo:
    """Parsed SPL / Token-2022 mint account."""

    mint: str
    ok: bool = False
    decimals: int = 0
    supply: float = 0.0
    mint_authority: str | None = None
    freeze_authority: str | None = None
    is_initialized: bool = False
    program: str = ""
    extensions: list[str] = field(default_factory=list)
    transfer_fee_bps: int = 0
    metadata_update_authority: str | None = None
    name: str = ""
    symbol: str = ""

    @property
    def is_token_2022(self) -> bool:
        return self.program == TOKEN_2022_PROGRAM

    @property
    def dangerous_extensions(self) -> list[str]:
        return [e for e in self.extensions if e in DANGEROUS_EXTENSIONS]

    @property
    def can_mint_more(self) -> bool:
        """A live mint authority means supply can be inflated at will."""
        return bool(self.mint_authority)

    @property
    def can_freeze(self) -> bool:
        """A live freeze authority means our token account can be frozen,
        which makes the position unsellable."""
        return bool(self.freeze_authority)


class SolanaRpc:
    """Read-only Solana JSON-RPC client."""

    def __init__(self, url: str = PUBLIC_RPC, http: HttpClient | None = None) -> None:
        self.url = url
        self.http = http or HttpClient(requests_per_minute=40.0, burst=6)
        self._id = 0

    def call(self, method: str, params: list[Any] | None = None) -> Any | None:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or []}
        resp = self.http.request(
            "POST", self.url, json=payload, headers={"Content-Type": "application/json"}
        )
        if resp is None:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if "error" in data:
            log.debug("rpc error on %s: %s", method, data["error"])
            return None
        return data.get("result")

    def get_health(self) -> bool:
        return self.call("getHealth") == "ok"

    def get_slot(self) -> int:
        return int(self.call("getSlot") or 0)

    def get_mint_info(self, mint: str) -> MintInfo:
        """Fetch and parse a mint account, including Token-2022 extensions."""
        result = self.call("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        value = (result or {}).get("value") if isinstance(result, dict) else None
        if not value:
            return MintInfo(mint=mint, ok=False)

        program = str(value.get("owner") or "")
        parsed = ((value.get("data") or {}).get("parsed") or {}).get("info") or {}
        if not parsed:
            return MintInfo(mint=mint, ok=False, program=program)

        ext_states = parsed.get("extensions") or []
        ext_names = [str(e.get("extension", "")) for e in ext_states if isinstance(e, dict)]

        transfer_fee_bps = 0
        name = symbol = ""
        meta_authority = None
        for e in ext_states:
            if not isinstance(e, dict):
                continue
            kind, state = e.get("extension"), e.get("state") or {}
            if kind == "transferFeeConfig":
                newer = state.get("newerTransferFee") or {}
                transfer_fee_bps = int(newer.get("transferFeeBasisPoints") or 0)
            elif kind == "tokenMetadata":
                name = str(state.get("name") or "")
                symbol = str(state.get("symbol") or "")
                meta_authority = state.get("updateAuthority")

        decimals = int(parsed.get("decimals") or 0)
        raw_supply = float(parsed.get("supply") or 0)
        return MintInfo(
            mint=mint,
            ok=True,
            decimals=decimals,
            supply=raw_supply / (10**decimals) if decimals else raw_supply,
            mint_authority=parsed.get("mintAuthority"),
            freeze_authority=parsed.get("freezeAuthority"),
            is_initialized=bool(parsed.get("isInitialized")),
            program=program,
            extensions=ext_names,
            transfer_fee_bps=transfer_fee_bps,
            metadata_update_authority=meta_authority,
            name=name,
            symbol=symbol,
        )

    def get_token_largest_accounts(self, mint: str) -> list[dict[str, Any]]:
        """Top-20 token accounts. Heavily rate-limited on the public endpoint."""
        result = self.call("getTokenLargestAccounts", [mint])
        return ((result or {}).get("value") or []) if isinstance(result, dict) else []

    def get_token_supply(self, mint: str) -> float:
        result = self.call("getTokenSupply", [mint])
        value = (result or {}).get("value") or {} if isinstance(result, dict) else {}
        return float(value.get("uiAmount") or 0.0)
