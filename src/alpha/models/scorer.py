"""The scoring model: P(trade wins | features observed at decision time).

The output is consumed directly by Kelly sizing, which makes **calibration**
more important than discrimination. A model with excellent ranking power but
systematically overstated probabilities will size every position too large and
blow up; one that ranks slightly worse but reports honest probabilities will
not. Every model here is therefore wrapped in an isotonic calibrator fitted on
held-out folds.

The design deliberately favours conservatism:

* Gradient boosting is capped at shallow depth with strong regularisation.
  Memecoin features are noisy and the sample is small; a deep model memorises
  tokens rather than learning structure.
* A logistic-regression baseline is always trained alongside. If the boosted
  model cannot beat a linear model on purged out-of-sample folds, the extra
  complexity is not buying anything and the linear model is used instead.
* A **prior-only baseline** (predict the base rate for everything) is scored
  too. Any model that fails to beat it has no edge, and the system says so
  rather than trading it.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from alpha.features.build import FEATURE_NAMES
from alpha.models.validation import PurgedGroupTimeSplit

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    n_splits: int = 5
    purge_seconds: float = 45 * 60
    embargo_seconds: float = 15 * 60
    # Deliberately small trees: the sample is short and the signal is faint.
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.7
    reg_lambda: float = 5.0
    random_state: int = 7
    # --- honesty gates -------------------------------------------------------
    # A model must clear ALL of these to be marked usable. They are deliberately
    # strict: an earlier, laxer gate (mean AUC > 0.52 and any Brier improvement)
    # passed a model trained on randomly shuffled labels, which is precisely the
    # failure that leads to trading noise with real money.
    min_auc: float = 0.55
    # Lower bound of the AUC confidence interval must still beat a coin flip.
    auc_ci_z: float = 2.0
    min_auc_lower_bound: float = 0.52
    # Brier score must improve on the base-rate prediction by this *relative*
    # margin, not merely by any amount.
    min_brier_improvement_pct: float = 0.02
    # At least this share of folds must individually beat 0.5, so a single
    # lucky fold cannot carry the average.
    min_fold_win_rate: float = 0.60


@dataclass
class TrainReport:
    n_samples: int
    n_features: int
    n_pools: int
    base_rate: float
    auc_mean: float
    auc_std: float
    auc_folds: list[float]
    brier: float
    brier_baseline: float
    logloss: float
    logloss_baseline: float
    beats_baseline: bool
    model_kind: str
    feature_importance: dict[str, float] = field(default_factory=dict)
    calibration_bins: list[dict[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        verdict = "USABLE" if self.beats_baseline else "NO EDGE — do not trade"
        return (
            f"[{verdict}] {self.model_kind}: n={self.n_samples} pools={self.n_pools} "
            f"base_rate={self.base_rate:.1%} AUC={self.auc_mean:.3f}±{self.auc_std:.3f} "
            f"Brier={self.brier:.4f} (baseline {self.brier_baseline:.4f})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples, "n_features": self.n_features, "n_pools": self.n_pools,
            "base_rate": self.base_rate, "auc_mean": self.auc_mean, "auc_std": self.auc_std,
            "auc_folds": self.auc_folds, "brier": self.brier, "brier_baseline": self.brier_baseline,
            "logloss": self.logloss, "logloss_baseline": self.logloss_baseline,
            "beats_baseline": self.beats_baseline, "model_kind": self.model_kind,
            "feature_importance": self.feature_importance,
            "calibration_bins": self.calibration_bins, "notes": self.notes,
        }


class Scorer:
    """Trains, calibrates, persists and applies the win-probability model."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.cfg = config or TrainConfig()
        self.model: Any = None
        self.calibrator: Any = None
        #: (model, scaler, calibrator) per fold. Averaged at inference.
        self.ensemble: list[tuple[Any, Any, Any]] = []
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.base_rate: float = 0.0
        self.report: TrainReport | None = None

    # ------------------------------------------------------------------ train

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: Sequence[float],
        groups: Sequence[str],
        sample_weight: Sequence[float] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> TrainReport:
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if feature_names:
            self.feature_names = list(feature_names)
        n, n_feat = X.shape
        self.base_rate = float(y.mean()) if n else 0.0
        notes: list[str] = []

        if n < 60 or len(np.unique(y)) < 2:
            notes.append(f"insufficient data to train (n={n}, classes={len(np.unique(y))})")
            self.report = TrainReport(
                n_samples=n, n_features=n_feat, n_pools=len(set(groups)), base_rate=self.base_rate,
                auc_mean=0.5, auc_std=0.0, auc_folds=[], brier=1.0, brier_baseline=1.0,
                logloss=1.0, logloss_baseline=1.0, beats_baseline=False, model_kind="none",
                notes=notes,
            )
            return self.report

        weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else np.ones(n)
        cv = PurgedGroupTimeSplit(
            n_splits=self.cfg.n_splits, purge=self.cfg.purge_seconds, embargo=self.cfg.embargo_seconds
        )
        folds = list(cv.split(times, groups))
        if not folds:
            notes.append("purged CV produced no usable folds")

        candidates = self._build_candidates()
        results: dict[str, dict[str, Any]] = {}

        for name, factory in candidates.items():
            oof = np.full(n, np.nan)
            aucs: list[float] = []
            for train_idx, test_idx in folds:
                if len(np.unique(y[train_idx])) < 2:
                    continue
                model = factory()
                scaler = None
                Xtr, Xte = X[train_idx], X[test_idx]
                if name == "logistic":
                    scaler = StandardScaler().fit(Xtr)
                    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
                try:
                    model.fit(Xtr, y[train_idx], sample_weight=weights[train_idx])
                    pred = model.predict_proba(Xte)[:, 1]
                except Exception as exc:  # a failing fold must not kill training
                    log.debug("fold failed for %s: %s", name, exc)
                    continue
                oof[test_idx] = pred
                if len(np.unique(y[test_idx])) > 1:
                    aucs.append(float(roc_auc_score(y[test_idx], pred)))

            mask = ~np.isnan(oof)
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            results[name] = {
                "oof": oof, "mask": mask, "aucs": aucs,
                "auc": float(np.mean(aucs)) if aucs else 0.5,
                "brier": float(brier_score_loss(y[mask], oof[mask])),
            }

        if not results:
            notes.append("no model produced valid out-of-sample predictions")
            self.report = TrainReport(
                n_samples=n, n_features=n_feat, n_pools=len(set(groups)), base_rate=self.base_rate,
                auc_mean=0.5, auc_std=0.0, auc_folds=[], brier=1.0, brier_baseline=1.0,
                logloss=1.0, logloss_baseline=1.0, beats_baseline=False, model_kind="none", notes=notes,
            )
            return self.report

        # Pick on out-of-sample AUC, preferring the simpler model on a tie.
        best_name = max(results, key=lambda k: (results[k]["auc"], k == "logistic"))
        best = results[best_name]
        oof, mask, aucs = best["oof"], best["mask"], best["aucs"]

        # Calibrate on out-of-fold predictions so probabilities are honest.
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(oof[mask], y[mask])
        calibrated = np.clip(iso.predict(oof[mask]), 1e-6, 1 - 1e-6)

        baseline = np.full(mask.sum(), self.base_rate)
        brier = float(brier_score_loss(y[mask], calibrated))
        brier_base = float(brier_score_loss(y[mask], baseline))
        ll = float(log_loss(y[mask], calibrated, labels=[0, 1]))
        ll_base = float(log_loss(y[mask], np.clip(baseline, 1e-6, 1 - 1e-6), labels=[0, 1]))
        auc_mean = float(np.mean(aucs)) if aucs else 0.5

        beats, gate_notes = self._evaluate_gates(aucs, auc_mean, brier, brier_base)
        notes.extend(gate_notes)

        # Build the inference ensemble.
        #
        # A single model refit on all data CANNOT be used with a calibrator
        # fitted on out-of-fold predictions: the refit model is scoring its own
        # training data at inference time, so its output distribution is far
        # more confident than the out-of-fold distribution the isotonic map was
        # built from. With ``out_of_bounds="clip"`` every prediction then lands
        # above the calibrator's fitted range and clips to one value — the model
        # silently returns a constant while still reporting a healthy AUC.
        #
        # This was observed: a model reporting AUC 0.626 returned 0.5586 for
        # every row, destroying all ranking information without raising anything.
        #
        # The fix is the standard one: keep each fold's model, pair it with a
        # calibrator fitted on that fold's held-out predictions, and average the
        # calibrated outputs at inference. Every calibrator is then applied to
        # exactly the distribution it was built for.
        self.ensemble = []
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2 or len(test_idx) < 10:
                continue
            model = candidates[best_name]()
            scaler = None
            Xtr, Xte = X[train_idx], X[test_idx]
            if best_name == "logistic":
                scaler = StandardScaler().fit(Xtr)
                Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
            try:
                model.fit(Xtr, y[train_idx], sample_weight=weights[train_idx])
                held_out = model.predict_proba(Xte)[:, 1]
            except Exception as exc:
                log.debug("ensemble member failed: %s", exc)
                continue
            if len(np.unique(y[test_idx])) < 2:
                continue
            member_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            member_cal.fit(held_out, y[test_idx])
            self.ensemble.append((model, scaler, member_cal))

        if not self.ensemble:
            notes.append("no ensemble member could be built; predictions fall back to the base rate")
        self.model = self.ensemble[0][0] if self.ensemble else None
        self.calibrator = iso
        self._scaler = None

        self.report = TrainReport(
            n_samples=n, n_features=n_feat, n_pools=len(set(groups)), base_rate=self.base_rate,
            auc_mean=auc_mean, auc_std=float(np.std(aucs)) if aucs else 0.0,
            auc_folds=[round(a, 4) for a in aucs], brier=brier, brier_baseline=brier_base,
            logloss=ll, logloss_baseline=ll_base, beats_baseline=bool(beats), model_kind=best_name,
            feature_importance=self._ensemble_importance(best_name),
            calibration_bins=_calibration_table(y[mask], calibrated),
            notes=notes,
        )
        return self.report

    def _evaluate_gates(
        self, aucs: list[float], auc_mean: float, brier: float, brier_base: float
    ) -> tuple[bool, list[str]]:
        """Apply every honesty gate and explain any that fail.

        All conditions must hold. Each exists because a weaker version of it
        admitted a model fitted to shuffled labels during development.
        """
        cfg = self.cfg
        notes: list[str] = []
        checks: list[bool] = []

        ok = auc_mean >= cfg.min_auc
        checks.append(ok)
        if not ok:
            notes.append(f"mean AUC {auc_mean:.3f} below required {cfg.min_auc:.3f}")

        # Confidence bound on the mean AUC across folds.
        if len(aucs) >= 2:
            se = float(np.std(aucs, ddof=1)) / math.sqrt(len(aucs))
            lower = auc_mean - cfg.auc_ci_z * se
            ok = lower > cfg.min_auc_lower_bound
            checks.append(ok)
            if not ok:
                notes.append(
                    f"AUC lower bound {lower:.3f} (mean−{cfg.auc_ci_z}·SE) "
                    f"does not exceed {cfg.min_auc_lower_bound:.3f}"
                )
        else:
            checks.append(False)
            notes.append("too few folds to bound AUC")

        rel = (brier_base - brier) / brier_base if brier_base > 0 else 0.0
        ok = rel >= cfg.min_brier_improvement_pct
        checks.append(ok)
        if not ok:
            notes.append(
                f"Brier improvement {rel:.2%} below required {cfg.min_brier_improvement_pct:.0%} "
                f"({brier:.4f} vs baseline {brier_base:.4f})"
            )

        if aucs:
            share = sum(1 for a in aucs if a > 0.5) / len(aucs)
            ok = share >= cfg.min_fold_win_rate
            checks.append(ok)
            if not ok:
                notes.append(
                    f"only {share:.0%} of folds beat 0.5 (need {cfg.min_fold_win_rate:.0%})"
                )
        return all(checks), notes

    def permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        times: Sequence[float],
        groups: Sequence[str],
        n_permutations: int = 20,
        sample_weight: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Compare the real AUC against models fitted to shuffled labels.

        This is the definitive check. If the true AUC does not sit clearly above
        the distribution of AUCs obtained from destroyed labels, the apparent
        edge is an artefact of the fitting procedure and nothing else.
        """
        rng = np.random.default_rng(self.cfg.random_state)
        y = np.asarray(y, dtype=int)

        real = Scorer(self.cfg)
        real_report = real.fit(X, y, times, groups, sample_weight, self.feature_names)
        null_aucs: list[float] = []
        for _ in range(n_permutations):
            shuffled = rng.permutation(y)
            trial = Scorer(self.cfg)
            rep = trial.fit(X, shuffled, times, groups, sample_weight, self.feature_names)
            null_aucs.append(rep.auc_mean)

        arr = np.asarray(null_aucs, dtype=float)
        # p-value: share of null runs matching or beating the real AUC.
        p_value = float((arr >= real_report.auc_mean).sum() + 1) / (len(arr) + 1)
        return {
            "real_auc": real_report.auc_mean,
            "null_auc_mean": float(arr.mean()) if len(arr) else 0.5,
            "null_auc_max": float(arr.max()) if len(arr) else 0.5,
            "null_auc_p95": float(np.percentile(arr, 95)) if len(arr) else 0.5,
            "p_value": p_value,
            "significant": bool(p_value < 0.05 and real_report.auc_mean > arr.max()),
            "n_permutations": n_permutations,
        }

    def _build_candidates(self) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression

        cfg = self.cfg
        candidates: dict[str, Any] = {
            "logistic": lambda: LogisticRegression(
                max_iter=2000, C=0.5, class_weight="balanced", random_state=cfg.random_state
            )
        }
        try:
            import lightgbm as lgb

            candidates["lightgbm"] = lambda: lgb.LGBMClassifier(
                n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate,
                max_depth=cfg.max_depth, num_leaves=cfg.num_leaves,
                min_child_samples=cfg.min_child_samples, subsample=cfg.subsample,
                subsample_freq=1, colsample_bytree=cfg.colsample_bytree,
                reg_lambda=cfg.reg_lambda, random_state=cfg.random_state,
                n_jobs=2, verbose=-1,
            )
        except ImportError:
            log.info("lightgbm unavailable; using logistic regression only")
        return candidates

    def _ensemble_importance(self, kind: str) -> dict[str, float]:
        """Feature importance averaged across ensemble members.

        Averaging matters: a single fold can rank a feature highly by accident,
        and reporting that one fold's view would misdescribe what the model
        actually uses at inference, which is the average of all of them.
        """
        if not self.ensemble:
            return {}
        totals: dict[str, float] = {}
        for model, _scaler, _cal in self.ensemble:
            for name, weight in self._importance(model, kind).items():
                totals[name] = totals.get(name, 0.0) + weight
        n = float(len(self.ensemble))
        ranked = sorted(((k, v / n) for k, v in totals.items()), key=lambda kv: kv[1], reverse=True)
        return {k: round(v, 5) for k, v in ranked[:25]}

    def _importance(self, model: Any, kind: str) -> dict[str, float]:
        try:
            if kind == "logistic":
                raw = np.abs(model.coef_[0])
            else:
                raw = np.asarray(model.feature_importances_, dtype=float)
        except Exception:
            return {}
        total = raw.sum() or 1.0
        pairs = sorted(
            zip(self.feature_names, (raw / total).tolist()), key=lambda kv: kv[1], reverse=True
        )
        return {k: round(v, 5) for k, v in pairs[:25]}

    # ------------------------------------------------------------------ score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probabilities, averaged over the fold ensemble.

        Falls back to the base rate when no ensemble was built, which is an
        honest "no opinion" rather than a fabricated one.
        """
        X = np.asarray(X, dtype=float)
        ensemble = getattr(self, "ensemble", None)
        if not ensemble:
            return np.full(len(X), self.base_rate)

        predictions = []
        for model, scaler, calibrator in ensemble:
            Xm = scaler.transform(X) if scaler is not None else X
            try:
                raw = model.predict_proba(Xm)[:, 1]
            except Exception:
                continue
            predictions.append(calibrator.predict(raw))
        if not predictions:
            return np.full(len(X), self.base_rate)
        return np.clip(np.mean(predictions, axis=0), 0.0, 1.0)

    def score_features(self, features: dict[str, float]) -> float:
        """Score a single feature dict, in the canonical feature order."""
        vec = np.array([[float(features.get(name, 0.0)) for name in self.feature_names]])
        if not np.all(np.isfinite(vec)):
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self.predict_proba(vec)[0])

    # ------------------------------------------------------------------- io

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model": self.model, "calibrator": self.calibrator,
                    "ensemble": self.ensemble,
                    "scaler": getattr(self, "_scaler", None),
                    "feature_names": self.feature_names, "base_rate": self.base_rate,
                    "report": self.report.to_dict() if self.report else None,
                },
                fh,
            )
        if self.report:
            path.with_suffix(".json").write_text(json.dumps(self.report.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Scorer":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        scorer = cls()
        scorer.model = blob["model"]
        scorer.calibrator = blob["calibrator"]
        scorer.ensemble = blob.get("ensemble") or []
        scorer._scaler = blob.get("scaler")
        scorer.feature_names = blob["feature_names"]
        scorer.base_rate = blob["base_rate"]
        return scorer


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    """Predicted vs realised win rate per probability decile."""
    out: list[dict[str, float]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        out.append(
            {
                "bin_low": round(float(edges[i]), 3),
                "bin_high": round(float(edges[i + 1]), 3),
                "n": int(mask.sum()),
                "predicted": round(float(p[mask].mean()), 4),
                "actual": round(float(y[mask].mean()), 4),
            }
        )
    return out
