"""Two-stage scoring: survive first, then win.

Measured on collected data, the outcome distribution for indiscriminate buying
is bimodal rather than continuous:

    45 wins        at  +71.9%
    57 small losses at   −9.9%
    47 total losses at  −100.0%     ← 31.5% of all trades

and the ordinary stop-loss fired on only 4% of trades, because **a stop cannot
protect against a token that stops trading**. There is no bid to sell into. That
single bucket drives expectancy from +26.2% (survivors only) to −13.6% (everything).

The consequence is that "which tokens go up" is the wrong primary question. The
right one is "which tokens will still be tradeable", and the two differ in how
precise a model must be to add value:

    filter removing 25% of deaths → −6.2% per trade
    filter removing 50% of deaths → +2.6% per trade   ← profitable
    filter removing 75% of deaths → +13.1% per trade

A survival filter only has to be *approximately* right, because removing part of
a −100% tail is worth more than ranking the survivors precisely. A return
forecaster has to be accurate before it is worth anything at all.

So the score is decomposed:

    P(win) = P(survive) · P(win | survive)

Each stage is trained, calibrated and gated independently. This matters
practically: the survival stage frequently clears its gate while the conditional
return stage does not, and in that case the system still has a usable edge. It
falls back to the empirical conditional base rate for the second term rather
than discarding the first stage's genuine signal — while saying clearly that it
has done so.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha.models.scorer import Scorer, TrainConfig, TrainReport

log = logging.getLogger(__name__)


@dataclass
class TwoStageReport:
    survival: TrainReport | None = None
    conditional: TrainReport | None = None
    survival_base_rate: float = 0.0
    conditional_base_rate: float = 0.0
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """The system is tradeable if *either* stage carries real signal.

        Survival alone is enough, because removing part of the total-loss tail
        is where most of the expectancy lives.
        """
        return bool(
            (self.survival and self.survival.beats_baseline)
            or (self.conditional and self.conditional.beats_baseline)
        )

    @property
    def mode(self) -> str:
        s = bool(self.survival and self.survival.beats_baseline)
        c = bool(self.conditional and self.conditional.beats_baseline)
        if s and c:
            return "both stages"
        if s:
            return "survival only"
        if c:
            return "conditional only"
        return "no edge"

    def summary(self) -> str:
        verdict = "USABLE" if self.usable else "NO EDGE — do not trade"
        parts = [f"[{verdict}] mode={self.mode} n={self.n_samples}"]
        if self.survival:
            parts.append(
                f"survival AUC={self.survival.auc_mean:.3f} "
                f"(base {self.survival_base_rate:.1%}, {'pass' if self.survival.beats_baseline else 'fail'})"
            )
        if self.conditional:
            parts.append(
                f"conditional AUC={self.conditional.auc_mean:.3f} "
                f"(base {self.conditional_base_rate:.1%}, "
                f"{'pass' if self.conditional.beats_baseline else 'fail'})"
            )
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "usable": self.usable,
            "mode": self.mode,
            "n_samples": self.n_samples,
            "survival_base_rate": self.survival_base_rate,
            "conditional_base_rate": self.conditional_base_rate,
            "survival": self.survival.to_dict() if self.survival else None,
            "conditional": self.conditional.to_dict() if self.conditional else None,
            "notes": self.notes,
        }


class TwoStageScorer:
    """Scores ``P(win) = P(survive) · P(win | survive)``."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.cfg = config or TrainConfig()
        self.survival = Scorer(self.cfg)
        self.conditional = Scorer(self.cfg)
        self.feature_names: list[str] = []
        self.survival_base_rate = 0.0
        self.conditional_base_rate = 0.0
        self.report: TwoStageReport | None = None
        self._use_survival_model = False
        self._use_conditional_model = False

    def fit(
        self,
        X: np.ndarray,
        y_survived: np.ndarray,
        y_win: np.ndarray,
        times: Sequence[float],
        groups: Sequence[str],
        sample_weight: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> TwoStageReport:
        X = np.asarray(X, dtype=float)
        y_survived = np.asarray(y_survived, dtype=int)
        y_win = np.asarray(y_win, dtype=int)
        self.feature_names = list(feature_names or [])
        notes: list[str] = []

        self.survival_base_rate = float(y_survived.mean()) if len(y_survived) else 0.0

        # Stage 1: survival, trained on everything.
        survival_report = self.survival.fit(
            X, y_survived, times, groups, sample_weight, feature_names
        )
        self._use_survival_model = survival_report.beats_baseline

        # Stage 2: win *conditional on surviving*, trained only on survivors.
        # Including deaths here would let the model learn "dead things don't
        # win", which is true, already captured by stage 1, and would make the
        # two stages multiply in the same information twice.
        mask = y_survived == 1
        conditional_report: TrainReport | None = None
        if mask.sum() >= 60:
            self.conditional_base_rate = float(y_win[mask].mean())
            weights = np.asarray(sample_weight, dtype=float)[mask] if sample_weight is not None else None
            conditional_report = self.conditional.fit(
                X[mask], y_win[mask],
                np.asarray(times, dtype=float)[mask],
                list(np.asarray(groups)[mask]),
                weights, feature_names,
            )
            self._use_conditional_model = conditional_report.beats_baseline
        else:
            self.conditional_base_rate = float(y_win[mask].mean()) if mask.sum() else 0.0
            notes.append(
                f"only {int(mask.sum())} surviving rows — too few to train the conditional stage; "
                f"using the empirical conditional base rate {self.conditional_base_rate:.1%}"
            )

        if self._use_survival_model and not self._use_conditional_model:
            notes.append(
                "survival stage carries signal but the conditional stage does not; "
                "using the conditional base rate for the second term. Most of the "
                "available expectancy comes from removing the total-loss tail, so "
                "this remains tradeable."
            )
        if not self._use_survival_model and self._use_conditional_model:
            notes.append(
                "conditional stage carries signal but survival does not; the "
                "total-loss tail is not being filtered, which is where most of "
                "the downside lives. Treat with caution."
            )

        self.report = TwoStageReport(
            survival=survival_report, conditional=conditional_report,
            survival_base_rate=self.survival_base_rate,
            conditional_base_rate=self.conditional_base_rate,
            n_samples=len(X), notes=notes,
        )
        return self.report

    # ------------------------------------------------------------------ score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = len(X)
        p_survive = (
            self.survival.predict_proba(X) if self._use_survival_model
            else np.full(n, self.survival_base_rate)
        )
        p_win_given = (
            self.conditional.predict_proba(X) if self._use_conditional_model
            else np.full(n, self.conditional_base_rate)
        )
        return np.clip(p_survive * p_win_given, 0.0, 1.0)

    def score_features(self, features: dict[str, float]) -> float:
        vec = np.array([[float(features.get(name, 0.0)) for name in self.feature_names]])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self.predict_proba(vec)[0])

    def survival_probability(self, features: dict[str, float]) -> float:
        """Stage-1 probability alone — useful as a standalone safety gate."""
        vec = np.array([[float(features.get(name, 0.0)) for name in self.feature_names]])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._use_survival_model:
            return self.survival_base_rate
        return float(self.survival.predict_proba(vec)[0])

    # --------------------------------------------------------------------- io

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "survival": self.survival, "conditional": self.conditional,
                    "feature_names": self.feature_names,
                    "survival_base_rate": self.survival_base_rate,
                    "conditional_base_rate": self.conditional_base_rate,
                    "use_survival_model": self._use_survival_model,
                    "use_conditional_model": self._use_conditional_model,
                    "report": self.report.to_dict() if self.report else None,
                },
                fh,
            )
        if self.report:
            path.with_suffix(".json").write_text(json.dumps(self.report.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TwoStageScorer":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        scorer = cls()
        scorer.survival = blob["survival"]
        scorer.conditional = blob["conditional"]
        scorer.feature_names = blob["feature_names"]
        scorer.survival_base_rate = blob["survival_base_rate"]
        scorer.conditional_base_rate = blob["conditional_base_rate"]
        scorer._use_survival_model = blob["use_survival_model"]
        scorer._use_conditional_model = blob["use_conditional_model"]
        return scorer
