"""Social signal ingestion — deterministic counting, LLM judgment, never both confused."""

from alpha.social.manipulation import ManipulationFeatures, compute_manipulation_features
from alpha.social.providers import ElfaProvider, SocialSignal, TwitterApiProvider

__all__ = [
    "ElfaProvider", "ManipulationFeatures", "SocialSignal", "TwitterApiProvider",
    "compute_manipulation_features",
]
