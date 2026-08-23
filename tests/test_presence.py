"""Tests for static social-presence features."""
from __future__ import annotations

from alpha.social.presence import PresenceFetcher, classify_social_link


def test_own_account_is_classified_as_owned():
    owned, handle, borrowed = classify_social_link("https://x.com/myproject")
    assert owned and handle == "myproject" and not borrowed


def test_link_to_someone_elses_post_is_borrowed_not_owned():
    """Linking a viral tweet is borrowing credibility, not having presence."""
    owned, handle, borrowed = classify_social_link(
        "https://x.com/elonmusk/status/2091476029860884897"
    )
    assert not owned and borrowed and handle == "elonmusk"


def test_famous_account_link_is_borrowed():
    owned, _, borrowed = classify_social_link("https://x.com/elonmusk")
    assert not owned and borrowed


def test_search_link_is_not_an_account():
    owned, _, borrowed = classify_social_link("https://x.com/search?q=%24TICKER")
    assert not owned and borrowed


def test_empty_link_is_neither():
    assert classify_social_link("") == (False, "", False)


def test_full_presence_scores_high():
    presence = PresenceFetcher.from_metadata({
        "description": "A community token with a documented roadmap and an active team.",
        "website": "https://project.io",
        "twitter": "https://x.com/realproject",
        "telegram": "https://t.me/realproject",
    })
    assert presence.presence_score > 0.8
    assert presence.channels == 3
    assert presence.twitter_is_owned


def test_borrowed_only_scores_zero():
    presence = PresenceFetcher.from_metadata({
        "description": "", "website": "", "twitter": "https://x.com/elonmusk/status/1",
    })
    assert presence.presence_score == 0.0
    assert presence.twitter_is_borrowed
    assert presence.notes


def test_missing_metadata_is_handled():
    presence = PresenceFetcher.from_metadata(None)
    assert not presence.has_metadata
    assert presence.presence_score == 0.0


def test_features_are_numeric_and_complete():
    features = PresenceFetcher.from_metadata({"twitter": "https://x.com/p"}).as_features()
    assert all(isinstance(v, float) for v in features.values())
    assert "presence_twitter_owned" in features
