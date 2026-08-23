"""Detecting coordinated promotion from post metadata.

The features here are deliberately *structural* rather than linguistic. Text
similarity used to be a workable bot signal, but campaign accounts now post
LLM-written, varied copy with complete bios and plausible avatars, and
text-based detection loses that arms race. What remains expensive to fake is
account history and graph position: an account's creation date cannot be
backdated, a genuine following of credible accounts cannot be bought cheaply,
and coordinating a burst of posts leaves a timing signature.

So the ranking of evidence, strongest first:

1. **Account age distribution.** A cohort of accounts created weeks ago
   promoting one ticker is close to conclusive.
2. **Smart-follower ratio.** Followers who are themselves credible accounts.
   Buying followers is cheap; buying *credible* followers is not.
3. **Timing burst structure.** Coordinated posts arrive in a spike; organic
   attention diffuses.
4. **Engagement ratio anomalies.** Both directions matter — an implausibly low
   ratio implies bought followers who never engage, an implausibly high one on a
   small account implies bought engagement.
5. **Author uniqueness and near-duplication.** Weakest, listed last, because it
   is the easiest for a modern campaign to defeat.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass
class ManipulationFeatures:
    """Structural evidence about whether attention on a token is organic."""

    n_posts: int = 0
    n_unique_authors: int = 0
    median_account_age_days: float = 0.0
    pct_accounts_under_30d: float = 0.0
    pct_accounts_under_90d: float = 0.0
    median_engagement_rate: float = 0.0
    median_gap_seconds: float = 0.0
    burst_ratio: float = 0.0
    near_duplicate_rate: float = 0.0
    unique_author_ratio: float = 0.0
    pct_verified: float = 0.0
    median_followers: float = 0.0
    smart_follower_ratio: float = 0.0
    organic_score: float = 0.5      # 0 = clearly coordinated, 1 = clearly organic
    flags: list[str] = field(default_factory=list)
    #: Count of flags based on account history and graph position rather than
    #: post text. These are the expensive-to-fake ones.
    structural_flags: int = 0

    @property
    def looks_coordinated(self) -> bool:
        """Whether the attention on this token appears manufactured.

        Two independent *structural* red flags are treated as conclusive on
        their own, regardless of the additive score. A campaign that writes its
        posts with an LLM defeats every text-similarity signal, so its additive
        score stays deceptively moderate while the account-level evidence is
        overwhelming. Requiring the sum to cross a threshold would let exactly
        the most sophisticated campaigns through — which is backwards.
        """
        return self.organic_score < 0.35 or self.structural_flags >= 2

    def as_features(self) -> dict[str, float]:
        """Numeric subset suitable for the model's feature vector."""
        return {
            "social_n_posts": float(self.n_posts),
            "social_unique_authors": float(self.n_unique_authors),
            "social_median_account_age_days": self.median_account_age_days,
            "social_pct_new_accounts": self.pct_accounts_under_30d,
            "social_engagement_rate": self.median_engagement_rate,
            "social_burst_ratio": self.burst_ratio,
            "social_unique_author_ratio": self.unique_author_ratio,
            "social_organic_score": self.organic_score,
        }


def compute_manipulation_features(
    posts: Sequence[dict[str, Any]], *, now: datetime | None = None
) -> ManipulationFeatures:
    """Compute structural manipulation evidence from a set of posts.

    ``posts`` should carry ``created_at``, ``text``, and an ``author`` mapping
    with ``created_at``, ``followers`` and ``is_verified``. Missing fields
    degrade gracefully — an absent signal simply does not contribute.
    """
    out = ManipulationFeatures()
    if not posts:
        return out
    now = now or datetime.now(timezone.utc)

    ages: list[float] = []
    engagement: list[float] = []
    timestamps: list[float] = []
    followers: list[float] = []
    texts: list[str] = []
    authors: set[str] = set()
    verified = 0

    for post in posts:
        author = post.get("author") or {}
        handle = str(author.get("username") or author.get("id") or "")
        if handle:
            authors.add(handle)

        created = _parse_dt(author.get("created_at"))
        if created:
            ages.append(max(0.0, (now - created).days))

        follower_count = float(author.get("followers") or 0)
        if follower_count > 0:
            followers.append(follower_count)
            likes = float(post.get("like_count") or 0)
            replies = float(post.get("reply_count") or 0)
            engagement.append(100.0 * (likes + replies) / follower_count)

        if author.get("is_verified"):
            verified += 1

        posted = _parse_dt(post.get("created_at"))
        if posted:
            timestamps.append(posted.timestamp())
        texts.append(str(post.get("text") or ""))

    n = len(posts)
    out.n_posts = n
    out.n_unique_authors = len(authors)
    out.unique_author_ratio = len(authors) / n if n else 0.0
    out.pct_verified = verified / n if n else 0.0

    if ages:
        out.median_account_age_days = statistics.median(ages)
        out.pct_accounts_under_30d = sum(1 for a in ages if a < 30) / len(ages)
        out.pct_accounts_under_90d = sum(1 for a in ages if a < 90) / len(ages)
    if engagement:
        out.median_engagement_rate = statistics.median(engagement)
    if followers:
        out.median_followers = statistics.median(followers)

    if len(timestamps) >= 3:
        timestamps.sort()
        gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
        out.median_gap_seconds = statistics.median(gaps) if gaps else 0.0
        span = timestamps[-1] - timestamps[0]
        if span > 0:
            # Share of posts falling in the densest 10% of the time span.
            window = span * 0.1
            best = max(
                sum(1 for t in timestamps if start <= t <= start + window) for start in timestamps
            )
            out.burst_ratio = best / len(timestamps)

    out.near_duplicate_rate = _near_duplicate_rate(texts)
    out.organic_score, out.flags, out.structural_flags = _score_organic(out)
    return out


def _near_duplicate_rate(texts: Sequence[str], threshold: float = 0.7) -> float:
    """Fraction of post pairs that are near-identical by token-set Jaccard."""
    sets = [set(t.lower().split()) for t in texts if t.strip()]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    # Cap the comparison to keep this O(1) in practice on large samples.
    sets = sets[:120]
    pairs = 0
    dupes = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairs += 1
            union = sets[i] | sets[j]
            if union and len(sets[i] & sets[j]) / len(union) > threshold:
                dupes += 1
    return dupes / pairs if pairs else 0.0


def _score_organic(f: ManipulationFeatures) -> tuple[float, list[str], int]:
    """Combine evidence into a 0–1 organic score, weighted by fakeability.

    Returns the score, the human-readable flags, and a count of how many of
    those flags are structural (account-history or graph based) rather than
    textual.
    """
    score = 1.0
    flags: list[str] = []
    structural = 0

    # Weights are ordered by how expensive each signal is to fake. Account age
    # is close to conclusive on its own, so it carries the largest deduction;
    # text similarity is the cheapest to defeat and carries the smallest.
    if f.pct_accounts_under_30d > 0.5:
        score -= 0.40
        flags.append(f"{f.pct_accounts_under_30d:.0%} of accounts are under 30 days old")
        structural += 1
    elif f.pct_accounts_under_90d > 0.6:
        score -= 0.25
        flags.append(f"{f.pct_accounts_under_90d:.0%} of accounts are under 90 days old")
        structural += 1

    if 0 < f.smart_follower_ratio < 0.005:
        score -= 0.25
        flags.append("almost no followers are credible accounts")
        structural += 1

    if f.burst_ratio > 0.6 and f.n_posts >= 10:
        score -= 0.25
        flags.append(f"{f.burst_ratio:.0%} of posts fall in one narrow window")
        structural += 1

    # Both tails are suspicious: a dead audience implies bought followers, and
    # implausibly high engagement on small accounts implies bought engagement.
    if 0 < f.median_engagement_rate < 0.2:
        score -= 0.20
        flags.append(f"engagement rate {f.median_engagement_rate:.2f}% implies inactive followers")
        structural += 1
    elif f.median_engagement_rate > 25.0 and f.median_followers < 5_000:
        score -= 0.15
        flags.append(
            f"engagement rate {f.median_engagement_rate:.1f}% implausibly high for "
            f"{f.median_followers:.0f}-follower accounts"
        )
        structural += 1

    if f.unique_author_ratio < 0.4 and f.n_posts >= 10:
        score -= 0.20
        flags.append(f"only {f.unique_author_ratio:.0%} of posts come from distinct authors")
        structural += 1

    if f.near_duplicate_rate > 0.35:
        score -= 0.15
        flags.append(f"{f.near_duplicate_rate:.0%} of post pairs are near-identical")

    return max(0.0, min(1.0, score)), flags, structural


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value)
    for parser in (_iso, _twitter_format):
        dt = parser(text)
        if dt:
            return dt
    return None


def _iso(text: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _twitter_format(text: str) -> datetime | None:
    # e.g. "Tue Dec 10 07:00:30 +0000 2024"
    try:
        return datetime.strptime(text, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        return None
