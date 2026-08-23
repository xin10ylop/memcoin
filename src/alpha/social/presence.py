"""Static social-presence features from token metadata.

Published analysis finds that a token's *static, pre-launch social presence* —
does it have a website, a Telegram, its own X account, a real description — is a
stronger and far more reliably measurable predictor than mention *velocity*.
That is a fortunate result: presence is free, available the instant a token
mints, carries no latency, and cannot be manufactured by a bot swarm the way
mention counts can.

The module also draws a distinction that raw presence checks miss. Consider a
token whose metadata lists:

    "twitter": "https://x.com/elonmusk/status/2091476029860884897"

That is not the project's account. It is a link to somebody else's viral post,
used to borrow credibility the project has not earned. A naive "has twitter?"
check scores it identically to a project with a real, maintained account, when
the two are opposite signals. :func:`classify_social_link` separates *owned*
presence from *borrowed* presence.

Metadata usually lives on IPFS, and the most common gateway (``ipfs.io``) is
behind an interstitial that returns 403 to programmatic clients, so several
gateways are tried in turn.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from alpha.http import HttpClient

log = logging.getLogger(__name__)

# ipfs.io is listed last: it serves a Cloudflare interstitial to non-browsers.
IPFS_GATEWAYS = (
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://ipfs.io/ipfs/",
)

_STATUS_PATTERNS = (
    re.compile(r"/status(?:es)?/\d+", re.I),   # link to a specific post
    re.compile(r"/i/web/status/\d+", re.I),
    re.compile(r"/search\?", re.I),            # link to a search, not an account
    re.compile(r"/hashtag/", re.I),
)

_HANDLE_PATTERN = re.compile(r"(?:x\.com|twitter\.com)/(@?[A-Za-z0-9_]{1,15})/?$", re.I)

#: Accounts frequently linked to borrow credibility. Presence of one of these
#: as "the token's twitter" is a negative signal, not a positive one.
_BORROWED_HANDLES = frozenset(
    {"elonmusk", "realdonaldtrump", "cz_binance", "vitalikbuterin", "solana", "pumpdotfun"}
)


@dataclass
class SocialPresence:
    """Static presence signals for one token."""

    has_metadata: bool = False
    has_description: bool = False
    description_length: int = 0
    has_website: bool = False
    has_twitter: bool = False
    has_telegram: bool = False
    has_discord: bool = False
    #: Twitter link points at the project's own account rather than someone
    #: else's post.
    twitter_is_owned: bool = False
    twitter_handle: str = ""
    #: Link borrows a well-known account's credibility.
    twitter_is_borrowed: bool = False
    launchpad: str = ""
    channels: int = 0
    presence_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def as_features(self) -> dict[str, float]:
        return {
            "presence_has_metadata": float(self.has_metadata),
            "presence_has_description": float(self.has_description),
            "presence_description_len": float(min(self.description_length, 500)),
            "presence_has_website": float(self.has_website),
            "presence_has_twitter": float(self.has_twitter),
            "presence_twitter_owned": float(self.twitter_is_owned),
            "presence_twitter_borrowed": float(self.twitter_is_borrowed),
            "presence_has_telegram": float(self.has_telegram),
            "presence_channels": float(self.channels),
            "presence_score": self.presence_score,
        }


def classify_social_link(url: str) -> tuple[bool, str, bool]:
    """Classify an X/Twitter link.

    Returns ``(is_owned_account, handle, is_borrowed)``. A link to a specific
    post, a search, or a hashtag is not an account; a link to a famous account
    is borrowed credibility rather than the project's own presence.
    """
    if not url or not isinstance(url, str):
        return False, "", False
    url = url.strip().rstrip("/")
    if any(p.search(url) for p in _STATUS_PATTERNS):
        # Points at content, not at an account.
        owner = re.search(r"(?:x\.com|twitter\.com)/([A-Za-z0-9_]{1,15})/", url + "/")
        handle = (owner.group(1).lower() if owner else "")
        return False, handle, True
    match = _HANDLE_PATTERN.search(url)
    if not match:
        return False, "", False
    handle = match.group(1).lstrip("@").lower()
    if handle in _BORROWED_HANDLES:
        return False, handle, True
    return True, handle, False


def _ipfs_candidates(uri: str) -> list[str]:
    """Expand an IPFS URI into gateway URLs to try in order."""
    cid = ""
    if uri.startswith("ipfs://"):
        cid = uri[len("ipfs://"):]
    else:
        match = re.search(r"/ipfs/([A-Za-z0-9]+)", uri)
        if match:
            cid = match.group(1)
    if not cid:
        return [uri]
    return [gateway + cid for gateway in IPFS_GATEWAYS]


class PresenceFetcher:
    """Fetches token metadata and derives static presence features."""

    def __init__(self, http: HttpClient | None = None) -> None:
        # Metadata hosts are unrelated to the market-data APIs, so this client
        # has its own budget and does not consume the GeckoTerminal allowance.
        self.http = http or HttpClient(requests_per_minute=60.0, burst=8, max_retries=1)
        self._cache: dict[str, SocialPresence] = {}

    def fetch(self, metadata_uri: str, *, launchpad: str = "") -> SocialPresence:
        """Fetch and classify one token's metadata."""
        if not metadata_uri:
            return SocialPresence(launchpad=launchpad)
        if metadata_uri in self._cache:
            return self._cache[metadata_uri]

        data: dict[str, Any] | None = None
        for url in _ipfs_candidates(metadata_uri):
            result = self.http.get_json(url)
            if isinstance(result, dict):
                data = result
                break
        presence = self.from_metadata(data, launchpad=launchpad)
        if presence.has_metadata:
            self._cache[metadata_uri] = presence
        return presence

    @staticmethod
    def from_metadata(data: dict[str, Any] | None, *, launchpad: str = "") -> SocialPresence:
        """Derive presence features from an already-fetched metadata document."""
        presence = SocialPresence(launchpad=launchpad)
        if not isinstance(data, dict):
            presence.notes.append("metadata unavailable")
            return presence
        presence.has_metadata = True

        def field_value(*names: str) -> str:
            for name in names:
                value = data.get(name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        description = field_value("description")
        presence.has_description = bool(description)
        presence.description_length = len(description)

        website = field_value("website", "external_url")
        presence.has_website = bool(website)

        twitter = field_value("twitter", "x")
        presence.has_twitter = bool(twitter)
        if twitter:
            owned, handle, borrowed = classify_social_link(twitter)
            presence.twitter_is_owned = owned
            presence.twitter_handle = handle
            presence.twitter_is_borrowed = borrowed
            if borrowed:
                presence.notes.append(
                    f"twitter link borrows credibility rather than owning an account ({handle or 'post link'})"
                )

        telegram = field_value("telegram", "coin_community", "community")
        presence.has_telegram = bool(telegram) and "t.me" in telegram.lower()
        presence.has_discord = bool(field_value("discord"))

        presence.channels = sum(
            (presence.has_website, presence.twitter_is_owned, presence.has_telegram, presence.has_discord)
        )

        # Weighted so that an *owned* account counts and a borrowed link does
        # not, and a real description counts for something on its own.
        score = 0.0
        score += 0.30 if presence.twitter_is_owned else 0.0
        score += 0.20 if presence.has_telegram else 0.0
        score += 0.20 if presence.has_website else 0.0
        score += 0.10 if presence.has_discord else 0.0
        score += 0.20 if presence.description_length >= 40 else (
            0.10 if presence.description_length >= 10 else 0.0
        )
        if presence.twitter_is_borrowed:
            score -= 0.15
        presence.presence_score = max(0.0, min(1.0, score))
        return presence
