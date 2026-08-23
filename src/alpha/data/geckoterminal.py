"""GeckoTerminal API client.

GeckoTerminal is the backbone data source for this system: it is free, needs no
API key, and — critically — it retains OHLCV history for pools that have already
died. That last property is what makes an unbiased backtest possible at all.

Verified behaviour (probed 2026-08-23, Solana network):

* ``new_pools`` paginates to page 10 = 200 pools, covering roughly the last
  14 minutes of pool creation. Solana creates ~14 new pools/minute, so polling
  all 10 pages once a minute captures effectively the entire universe.
* ``ohlcv/minute`` returns at most 1000 candles per call and paginates
  arbitrarily far back via ``before_timestamp``.
* ``ohlcv/day`` is capped at ~181 candles regardless of ``limit``.
* ``trades`` returns the last 300 swaps *including* ``tx_from_address``, which
  gives free wallet-level attribution.
* The free tier allows ~30 requests/minute. Exceeding it, or sending a default
  Python User-Agent, returns ``403``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from alpha.http import HttpClient

log = logging.getLogger(__name__)

BASE = "https://api.geckoterminal.com/api/v2"
NETWORK = "solana"
WSOL = "So11111111111111111111111111111111111111112"

# GeckoTerminal free tier: 30 calls/min. We run at 25 to leave headroom for
# retries and any concurrent process.
FREE_TIER_RPM = 25.0

MAX_NEW_POOL_PAGES = 10
OHLCV_MAX_LIMIT = 1000


def _f(value: Any, default: float = 0.0) -> float:
    """Parse GeckoTerminal's stringly-typed numerics, tolerating None/''/junk."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    # Guard against NaN/inf leaking into feature vectors.
    return out if out == out and abs(out) != float("inf") else default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(slots=True)
class TimeframeStats:
    """Buy/sell/volume/price-change stats for one timeframe bucket."""

    buys: int = 0
    sells: int = 0
    buyers: int = 0
    sellers: int = 0
    volume_usd: float = 0.0
    price_change_pct: float = 0.0

    @property
    def txns(self) -> int:
        return self.buys + self.sells

    @property
    def buy_ratio(self) -> float:
        """Fraction of transactions that were buys; 0.5 when there is no data."""
        total = self.txns
        return self.buys / total if total else 0.5

    @property
    def unique_traders(self) -> int:
        return self.buyers + self.sellers


@dataclass(slots=True)
class Pool:
    """A normalised GeckoTerminal pool snapshot."""

    address: str
    name: str
    dex: str
    base_mint: str
    quote_mint: str
    created_at: datetime | None
    price_usd: float
    price_native: float
    fdv_usd: float
    market_cap_usd: float
    liquidity_usd: float
    timeframes: dict[str, TimeframeStats]
    observed_at: datetime

    TIMEFRAMES = ("m5", "m15", "m30", "h1", "h6", "h24")

    @property
    def age_minutes(self) -> float:
        if self.created_at is None:
            return float("nan")
        return (self.observed_at - self.created_at).total_seconds() / 60.0

    @property
    def is_sol_quoted(self) -> bool:
        """SOL-quoted pools are the memecoin universe; USDC pairs are majors."""
        return self.quote_mint == WSOL

    def tf(self, name: str) -> TimeframeStats:
        return self.timeframes.get(name, TimeframeStats())

    @classmethod
    def from_api(cls, item: dict[str, Any], observed_at: datetime | None = None) -> "Pool":
        attrs = item.get("attributes", {}) or {}
        rels = item.get("relationships", {}) or {}

        def rel_mint(key: str) -> str:
            data = (rels.get(key) or {}).get("data") or {}
            # ids look like "solana_<mint>"; strip the network prefix.
            raw = data.get("id", "")
            return raw.split("_", 1)[1] if "_" in raw else raw

        dex_data = (rels.get("dex") or {}).get("data") or {}
        txns = attrs.get("transactions") or {}
        vol = attrs.get("volume_usd") or {}
        chg = attrs.get("price_change_percentage") or {}

        timeframes: dict[str, TimeframeStats] = {}
        for name in cls.TIMEFRAMES:
            t = txns.get(name) or {}
            timeframes[name] = TimeframeStats(
                buys=_i(t.get("buys")),
                sells=_i(t.get("sells")),
                buyers=_i(t.get("buyers")),
                sellers=_i(t.get("sellers")),
                volume_usd=_f(vol.get(name)),
                price_change_pct=_f(chg.get(name)),
            )

        return cls(
            address=attrs.get("address", ""),
            name=attrs.get("name", ""),
            dex=dex_data.get("id", ""),
            base_mint=rel_mint("base_token"),
            quote_mint=rel_mint("quote_token"),
            created_at=parse_ts(attrs.get("pool_created_at")),
            price_usd=_f(attrs.get("base_token_price_usd")),
            price_native=_f(attrs.get("base_token_price_native_currency")),
            fdv_usd=_f(attrs.get("fdv_usd")),
            market_cap_usd=_f(attrs.get("market_cap_usd")),
            liquidity_usd=_f(attrs.get("reserve_in_usd")),
            timeframes=timeframes,
            observed_at=observed_at or datetime.now(timezone.utc),
        )

    def to_row(self) -> dict[str, Any]:
        """Flatten to a storage/feature row."""
        row: dict[str, Any] = {
            "pool": self.address,
            "name": self.name,
            "dex": self.dex,
            "base_mint": self.base_mint,
            "quote_mint": self.quote_mint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "observed_at": self.observed_at.isoformat(),
            "age_min": round(self.age_minutes, 3) if self.created_at else None,
            "price_usd": self.price_usd,
            "price_native": self.price_native,
            "fdv_usd": self.fdv_usd,
            "market_cap_usd": self.market_cap_usd,
            "liquidity_usd": self.liquidity_usd,
        }
        for name in self.TIMEFRAMES:
            t = self.tf(name)
            row[f"buys_{name}"] = t.buys
            row[f"sells_{name}"] = t.sells
            row[f"buyers_{name}"] = t.buyers
            row[f"sellers_{name}"] = t.sellers
            row[f"vol_{name}"] = t.volume_usd
            row[f"chg_{name}"] = t.price_change_pct
        return row


@dataclass(slots=True)
class Trade:
    """One swap against a pool, with the trader's wallet attached."""

    tx_hash: str
    block_number: int
    wallet: str
    timestamp: datetime | None
    kind: str  # "buy" or "sell" (from the base token's perspective)
    volume_usd: float
    price_usd: float
    from_amount: float
    to_amount: float

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> "Trade":
        a = item.get("attributes", {}) or {}
        kind = a.get("kind", "")
        # price_from/price_to refer to the swap direction, not the base token.
        # On a buy the base token is what we receive (to_token).
        price = _f(a.get("price_to_in_usd")) if kind == "buy" else _f(a.get("price_from_in_usd"))
        return cls(
            tx_hash=a.get("tx_hash", ""),
            block_number=_i(a.get("block_number")),
            wallet=a.get("tx_from_address", ""),
            timestamp=parse_ts(a.get("block_timestamp")),
            kind=kind,
            volume_usd=_f(a.get("volume_in_usd")),
            price_usd=price,
            from_amount=_f(a.get("from_token_amount")),
            to_amount=_f(a.get("to_token_amount")),
        )


@dataclass(slots=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume_usd: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc)


class GeckoTerminalClient:
    """Typed client for the GeckoTerminal v2 public API."""

    def __init__(self, http: HttpClient | None = None, network: str = NETWORK) -> None:
        self.http = http or HttpClient(requests_per_minute=FREE_TIER_RPM, burst=5)
        self.network = network

    # ---------------------------------------------------------------- discovery

    def new_pools(self, page: int = 1) -> list[Pool]:
        """Newest pools on the network. Page 1 is newest; page 10 is the limit."""
        data = self.http.get_json(f"{BASE}/networks/{self.network}/new_pools", params={"page": page})
        return self._pools(data)

    def all_new_pools(self, max_pages: int = MAX_NEW_POOL_PAGES) -> list[Pool]:
        """Sweep every discoverable new-pool page, de-duplicated by address.

        Pages overlap and reorder between calls because pools are being created
        continuously, so de-duplication is required rather than optional.
        """
        seen: dict[str, Pool] = {}
        for page in range(1, max_pages + 1):
            batch = self.new_pools(page)
            if not batch:
                break
            for pool in batch:
                if pool.address:
                    seen.setdefault(pool.address, pool)
        return list(seen.values())

    def trending_pools(self, page: int = 1, duration: str = "5m") -> list[Pool]:
        data = self.http.get_json(
            f"{BASE}/networks/{self.network}/trending_pools",
            params={"page": page, "duration": duration},
        )
        return self._pools(data)

    def search_pools(self, query: str, page: int = 1) -> list[Pool]:
        data = self.http.get_json(
            f"{BASE}/search/pools", params={"query": query, "network": self.network, "page": page}
        )
        return self._pools(data)

    # ------------------------------------------------------------------- detail

    def pool(self, address: str) -> Pool | None:
        data = self.http.get_json(f"{BASE}/networks/{self.network}/pools/{address}")
        item = (data or {}).get("data")
        return Pool.from_api(item) if item else None

    def pools_multi(self, addresses: list[str]) -> list[Pool]:
        """Batch pool lookup — up to 30 addresses in one request.

        This is the single most important efficiency lever in the system: it
        turns 30 rate-limited calls into one, which is what makes tracking
        thousands of live positions feasible on a 30 req/min budget.
        """
        out: list[Pool] = []
        for i in range(0, len(addresses), 30):
            chunk = [a for a in addresses[i : i + 30] if a]
            if not chunk:
                continue
            data = self.http.get_json(
                f"{BASE}/networks/{self.network}/pools/multi/{','.join(chunk)}"
            )
            out.extend(self._pools(data))
        return out

    def ohlcv(
        self,
        pool: str,
        timeframe: str = "minute",
        aggregate: int = 1,
        limit: int = OHLCV_MAX_LIMIT,
        before_timestamp: int | None = None,
        currency: str = "usd",
    ) -> list[Candle]:
        params: dict[str, Any] = {
            "aggregate": aggregate,
            "limit": min(limit, OHLCV_MAX_LIMIT),
            "currency": currency,
        }
        if before_timestamp is not None:
            params["before_timestamp"] = before_timestamp
        data = self.http.get_json(
            f"{BASE}/networks/{self.network}/pools/{pool}/ohlcv/{timeframe}", params=params
        )
        rows = ((data or {}).get("data", {}).get("attributes", {}) or {}).get("ohlcv_list") or []
        out: list[Candle] = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 6:
                continue
            out.append(
                Candle(
                    ts=_i(r[0]),
                    open=_f(r[1]),
                    high=_f(r[2]),
                    low=_f(r[3]),
                    close=_f(r[4]),
                    volume_usd=_f(r[5]),
                )
            )
        # API returns newest-first; ascending order is far easier to reason about.
        out.sort(key=lambda c: c.ts)
        return out

    def ohlcv_history(
        self, pool: str, *, timeframe: str = "minute", aggregate: int = 1, max_calls: int = 10
    ) -> list[Candle]:
        """Walk ``before_timestamp`` backwards to assemble a long history."""
        all_candles: dict[int, Candle] = {}
        before: int | None = None
        for _ in range(max_calls):
            batch = self.ohlcv(pool, timeframe, aggregate, before_timestamp=before)
            if not batch:
                break
            new = {c.ts: c for c in batch if c.ts not in all_candles}
            if not new:
                break
            all_candles.update(new)
            before = min(c.ts for c in batch)
            if len(batch) < OHLCV_MAX_LIMIT:
                break  # reached the beginning of the pool's history
        return sorted(all_candles.values(), key=lambda c: c.ts)

    def trades(self, pool: str, min_volume_usd: float = 0.0) -> list[Trade]:
        """Last ~300 swaps against a pool, each carrying the trader's wallet."""
        params = {"trade_volume_in_usd_greater_than": min_volume_usd} if min_volume_usd else None
        data = self.http.get_json(f"{BASE}/networks/{self.network}/pools/{pool}/trades", params=params)
        items = (data or {}).get("data") or []
        trades = [Trade.from_api(i) for i in items]
        trades.sort(key=lambda t: (t.timestamp or datetime.min.replace(tzinfo=timezone.utc)))
        return trades

    def token_info(self, mint: str) -> dict[str, Any] | None:
        data = self.http.get_json(f"{BASE}/networks/{self.network}/tokens/{mint}/info")
        item = (data or {}).get("data") or {}
        return item.get("attributes")

    def iter_new_pools(self, max_pages: int = MAX_NEW_POOL_PAGES) -> Iterator[Pool]:
        for page in range(1, max_pages + 1):
            batch = self.new_pools(page)
            if not batch:
                return
            yield from batch

    @staticmethod
    def _pools(data: Any) -> list[Pool]:
        items = (data or {}).get("data") or []
        if isinstance(items, dict):
            items = [items]
        out = []
        for i in items:
            try:
                out.append(Pool.from_api(i))
            except Exception as exc:  # a malformed pool must not kill a sweep
                log.debug("skipping malformed pool: %s", exc)
        return out
