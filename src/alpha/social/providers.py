"""Social data providers.

The architecture separates two things that are frequently and expensively
conflated:

* **Counting** — how many mentions, from whom, with what engagement, changing at
  what rate. This must be deterministic, reproducible and cheap, because it
  feeds thresholds and backtests. A strategy whose inputs are not reproducible
  cannot be backtested at all.
* **Judging** — is this narrative real, is this account credible, does this read
  as a coordinated campaign. This is genuinely hard and an LLM does it well.

Providers in this module do the counting. :mod:`alpha.social.judge` does the
judging, and it is wired so that it can only ever *veto* a trade, never size or
trigger one.

Every provider is optional. Absent an API key, each returns an explicitly empty
result and the system trades on on-chain data alone rather than silently
substituting a fabricated value.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from alpha.http import HttpClient

log = logging.getLogger(__name__)


@dataclass
class SocialSignal:
    """Deterministic social measurements for one token."""

    token: str
    available: bool = False
    mention_count: int = 0
    previous_count: int = 0
    change_percent: float = 0.0
    smart_mentions: int = 0
    posts: list[dict[str, Any]] = field(default_factory=list)
    source: str = ""
    error: str | None = None

    @property
    def velocity(self) -> float:
        """Growth in mentions versus the prior window."""
        if self.previous_count <= 0:
            return float(self.mention_count)
        return self.mention_count / self.previous_count


class SocialProvider:
    """Base class. Subclasses must degrade gracefully without credentials."""

    name = "none"

    @property
    def enabled(self) -> bool:
        return False

    def trending_contracts(self, window: str = "30m", min_mentions: int = 5) -> list[dict[str, Any]]:
        return []

    def token_signal(self, token: str, window: str = "1h") -> SocialSignal:
        return SocialSignal(token=token, available=False, source=self.name, error="provider disabled")


class ElfaProvider(SocialProvider):
    """Elfa AI — crypto-native social aggregation.

    Chosen as the primary counting source because it returns *contract
    addresses* rather than tickers. Ticker matching is hopeless for memecoins:
    symbols collide constantly and impersonation is routine, so a mention count
    keyed on "$PEPE" aggregates dozens of unrelated tokens. Keying on the mint
    sidesteps the problem entirely.
    """

    name = "elfa"
    BASE = "https://api.elfa.ai/v2"

    def __init__(self, api_key: str | None = None, http: HttpClient | None = None) -> None:
        self.api_key = api_key or os.environ.get("ELFA_API_KEY", "")
        self.http = http or HttpClient(requests_per_minute=55.0, burst=5)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"x-elfa-api-key": self.api_key, "Accept": "application/json"}

    def trending_contracts(self, window: str = "30m", min_mentions: int = 5) -> list[dict[str, Any]]:
        """Contract addresses whose mention counts are spiking."""
        if not self.enabled:
            return []
        data = self.http.get_json(
            f"{self.BASE}/aggregations/trending-cas/twitter",
            params={"timeWindow": window, "minMentions": min_mentions, "pageSize": 50},
            headers=self._headers(),
        )
        rows = (((data or {}).get("data") or {}).get("data")) or []
        return [r for r in rows if str(r.get("chain", "")).lower() == "solana"]

    def token_signal(self, token: str, window: str = "1h") -> SocialSignal:
        if not self.enabled:
            return SocialSignal(token=token, source=self.name, error="ELFA_API_KEY not set")
        data = self.http.get_json(
            f"{self.BASE}/aggregations/trending-tokens",
            params={"timeWindow": window, "minMentions": 1, "pageSize": 50},
            headers=self._headers(),
        )
        rows = (((data or {}).get("data") or {}).get("data")) or []
        needle = token.lstrip("$").lower()
        for row in rows:
            if str(row.get("token", "")).lstrip("$").lower() == needle:
                return SocialSignal(
                    token=token, available=True, source=self.name,
                    mention_count=int(row.get("current_count") or 0),
                    previous_count=int(row.get("previous_count") or 0),
                    change_percent=float(row.get("change_percent") or 0.0),
                )
        return SocialSignal(token=token, available=True, source=self.name, mention_count=0)

    def account_quality(self, username: str) -> dict[str, Any]:
        """Smart-follower statistics for one account.

        The smart-follower ratio is the most robust cheap authenticity signal
        available: followers can be bought, but followers who are themselves
        credible crypto accounts cannot be bought cheaply.
        """
        if not self.enabled:
            return {}
        data = self.http.get_json(
            f"{self.BASE}/account/smart-stats", params={"username": username}, headers=self._headers()
        )
        stats = (data or {}).get("data") or {}
        if stats:
            followers = float(stats.get("followerCount") or 0)
            smart = float(stats.get("smartFollowerCount") or 0)
            stats["smart_ratio"] = smart / followers if followers > 0 else 0.0
        return stats


class TwitterApiProvider(SocialProvider):
    """twitterapi.io — raw posts with full author metadata.

    This is the verification tier. It is the only source in the stack that
    returns author account-creation dates and follower counts, which are the
    fields the structural manipulation checks in
    :mod:`alpha.social.manipulation` actually depend on.
    """

    name = "twitterapi"
    BASE = "https://api.twitterapi.io"

    def __init__(self, api_key: str | None = None, http: HttpClient | None = None) -> None:
        self.api_key = api_key or os.environ.get("TWITTERAPI_KEY", "")
        self.http = http or HttpClient(requests_per_minute=60.0, burst=6)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, *, since_epoch: int | None = None, max_posts: int = 100) -> list[dict[str, Any]]:
        """Fetch posts matching ``query``, normalised to our internal shape."""
        if not self.enabled:
            return []
        full_query = f'"{query}"'
        if since_epoch:
            full_query += f" since_time:{since_epoch}"
        out: list[dict[str, Any]] = []
        cursor = ""
        while len(out) < max_posts:
            data = self.http.get_json(
                f"{self.BASE}/twitter/tweet/advanced_search",
                params={"query": full_query, "queryType": "Latest", "cursor": cursor},
                headers={"X-API-Key": self.api_key},
            )
            if not data:
                break
            tweets = data.get("tweets") or []
            if not tweets:
                break
            out.extend(self._normalise(t) for t in tweets)
            if not data.get("has_next_page"):
                break
            cursor = data.get("next_cursor") or ""
            if not cursor:
                break
        return out[:max_posts]

    def token_signal(self, token: str, window: str = "1h") -> SocialSignal:
        if not self.enabled:
            return SocialSignal(token=token, source=self.name, error="TWITTERAPI_KEY not set")
        posts = self.search(token)
        return SocialSignal(
            token=token, available=True, source=self.name,
            mention_count=len(posts), posts=posts,
        )

    @staticmethod
    def _normalise(tweet: dict[str, Any]) -> dict[str, Any]:
        author = tweet.get("author") or {}
        return {
            "id": tweet.get("id"),
            "url": tweet.get("url"),
            "text": tweet.get("text") or "",
            "created_at": tweet.get("createdAt"),
            "like_count": tweet.get("likeCount") or 0,
            "reply_count": tweet.get("replyCount") or 0,
            "retweet_count": tweet.get("retweetCount") or 0,
            "view_count": tweet.get("viewCount") or 0,
            "author": {
                "id": author.get("id"),
                "username": author.get("userName"),
                "created_at": author.get("createdAt"),
                "followers": author.get("followers") or 0,
                "is_verified": bool(author.get("isBlueVerified")),
            },
        }


def build_providers() -> list[SocialProvider]:
    """Instantiate whichever providers have credentials configured."""
    providers: list[SocialProvider] = []
    for cls in (ElfaProvider, TwitterApiProvider):
        provider = cls()
        if provider.enabled:
            providers.append(provider)
            log.info("social provider enabled: %s", provider.name)
        else:
            log.debug("social provider %s disabled (no API key)", provider.name)
    return providers
