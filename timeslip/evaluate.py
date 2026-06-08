"""
Mature evaluation, calibration, learning curves and persistence for Time Slip.

A predictive model is only as trustworthy as the way it is tested. We evaluate
the next-``SLIP_HORIZON_MIN``-minute risk model under two honest regimes:

  COLD-START (new person)   train on a set of people, test on people the model
                            has *never seen*. The hard, generalisation case.

  PERSONALISED (future days) train on everyone's earlier days, test on their
                            later days. The model knows the person but not the
                            future. This is the Obsidian use case, and it is
                            legitimately more accurate.

We then calibrate probabilities (isotonic on a held-out slice), draw a learning
curve (accuracy vs. amount of training data), and persist a production model +
its calibrator + feature schema so it can score new self-logged data.

Honesty note: the per-minute slip event is partly irreducibly stochastic, so
windowed ROC in the high 0.7s-0.8s with good calibration is a *strong* result,
not a disappointing one. Anything near 1.0 would indicate leakage.
"""

from __future__ import annotations
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)
from sklearn.isotonic import IsotonicRegression

from . import config as C

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:                                   # pragma: no cover
    _HAVE_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def _make_model(spw: float):
    if _HAVE_XGB:
        return XGBClassifier(
            n_estimators=700, max_depth=6, learning_rate=0.045, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0, gamma=0.4,
            scale_pos_weight=spw, tree_method="hist", eval_metric="logloss",
            early_stopping_rounds=40, random_state=C.GLOBAL_SEED, n_jobs=0)
    return HistGradientBoostingClassifier(                       # pragma: no cover
        max_depth=6, learning_rate=0.045, max_iter=700,
        early_stopping=True, class_weight="balanced", random_state=C.GLOBAL_SEED)


def _fit(model, Xtr, ytr, Xval, yval):
    if _HAVE_XGB:
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    else:                                                        # pragma: no cover
        model.fit(Xtr, ytr)
    return model


def _spw(y) -> float:
    pos = max(1, int(np.sum(y))); neg = int(np.sum(y == 0)); return neg / pos


def _metrics(y, p) -> Dict:
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "base_rate": float(np.mean(y))}


def _per_person_auc(meta_sub: pd.DataFrame, y, p) -> pd.Series:
    out = {}
    for pid, idx in meta_sub.groupby("pid").groups.items():
        ii = meta_sub.index.get_indexer(idx)
        yy, pp = y[ii], p[ii]
        if 0 < yy.sum() < len(yy):
            out[pid] = float(roc_auc_score(yy, pp))
    return pd.Series(out)


# --------------------------------------------------------------------------- #
def regime_cold_start(fb: Dict) -> Dict:
    """Train on a subset of people; test on entirely unseen people."""
    X, y, meta = fb["X_real"], fb["y_window"], fb["meta"]
    pids = meta["pid"].unique()
    rng = np.random.default_rng(C.GLOBAL_SEED)
    pids = rng.permutation(pids)
    n = len(pids)
    test_p = set(pids[: max(2, n // 6)])
    val_p = set(pids[max(2, n // 6): max(4, n // 3)])
    train_p = set(pids[max(4, n // 3):])

    pid = meta["pid"].to_numpy()
    tr = np.isin(pid, list(train_p)); va = np.isin(pid, list(val_p)); te = np.isin(pid, list(test_p))
    model = _make_model(_spw(y[tr]))
    _fit(model, X[tr], y[tr], X[va], y[va])
    p_te = model.predict_proba(X[te])[:, 1]
    m = _metrics(y[te], p_te)
    ppa = _per_person_auc(meta[te].reset_index(drop=True), y[te], p_te)
    return dict(metrics=m, per_person_auc=ppa, n_train_people=len(train_p),
                n_test_people=len(test_p), y=y[te], p=p_te,
                model=model, val=(X[va], y[va]))


def regime_personalized(fb: Dict) -> Dict:
    """Train on everyone's earlier days; test on their later days."""
    X, y, meta = fb["X_real"], fb["y_window"], fb["meta"]
    tr_all, te = fb["train_mask"], fb["test_mask"]
    # carve the latest training day as a validation slice for early stopping
    day = meta["day"].to_numpy()
    cut = day[tr_all].max()
    va = tr_all & (day == cut)
    tr = tr_all & (day < cut)
    model = _make_model(_spw(y[tr]))
    _fit(model, X[tr], y[tr], X[va], y[va])
    p_te = model.predict_proba(X[te])[:, 1]
    m = _metrics(y[te], p_te)
    ppa = _per_person_auc(meta[te].reset_index(drop=True), y[te], p_te)
    return dict(metrics=m, per_person_auc=ppa, y=y[te], p=p_te,
                model=model, val=(X[va], y[va]))


def calibrate(p_val, y_val, p_test, y_test) -> Dict:
    """Isotonic calibration fit on validation; report Brier before/after."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y_val)
    p_cal = iso.transform(p_test)
    return dict(iso=iso, brier_before=float(brier_score_loss(y_test, p_test)),
                brier_after=float(brier_score_loss(y_test, p_cal)),
                p_cal=p_cal)


def learning_curve(fb: Dict, points: List[int] = None) -> pd.DataFrame:
    """Cold-start ROC as the number of TRAINING PEOPLE grows (fixed test set)."""
    X, y, meta = fb["X_real"], fb["y_window"], fb["meta"]
    pids = list(np.random.default_rng(C.GLOBAL_SEED).permutation(meta["pid"].unique()))
    pid = meta["pid"].to_numpy()
    test_p = pids[: max(2, len(pids) // 6)]
    pool = pids[max(2, len(pids) // 6):]
    te = np.isin(pid, test_p)
    if points is None:
        points = sorted(set([max(2, len(pool) // 8), len(pool) // 4,
                             len(pool) // 2, 3 * len(pool) // 4, len(pool)]))
    rows = []
    for k in points:
        train_p = pool[:k]
        tr = np.isin(pid, train_p)
        va = np.isin(pid, pool[max(0, k - 2):k]) if k >= 4 else tr
        model = _make_model(_spw(y[tr]))
        _fit(model, X[tr], y[tr], X[va], y[va])
        auc = roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])
        rows.append({"n_train_people": int(k), "roc_auc": float(auc)})
    return pd.DataFrame(rows)


def train_production_model(fb: Dict) -> Dict:
    """Final model on ALL data + isotonic calibrator + feature schema, for deploy."""
    X, y, meta = fb["X_real"], fb["y_window"], fb["meta"]
    day = meta["day"].to_numpy()
    cut = day.max()
    va = (day == cut)                                # last day as validation
    tr = ~va
    model = _make_model(_spw(y[tr]))
    _fit(model, X[tr], y[tr], X[va], y[va])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(X[va])[:, 1], y[va])
    return dict(model=model, iso=iso, columns=list(X.columns),
                horizon=fb["horizon_min"])


def save_artifacts(prod: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(prod, path)


def load_artifacts(path: str) -> Dict:
    return joblib.load(path)


def predict_risk(prod: Dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated next-horizon slip risk for feature rows aligned to schema."""
    Xa = X.reindex(columns=prod["columns"], fill_value=0.0)
    raw = prod["model"].predict_proba(Xa)[:, 1]
    return prod["iso"].transform(raw)


if __name__ == "__main__":
    from .simulate import simulate_all
    from .features import build_features
    m, e, pt = simulate_all()
    fb = build_features(m, pt)
    cs = regime_cold_start(fb)
    ps = regime_personalized(fb)
    print("COLD-START (unseen people): ", {k: round(v, 3) for k, v in cs["metrics"].items()})
    print("  per-person AUC: median %.3f  (IQR %.3f-%.3f)" % (
        cs["per_person_auc"].median(), cs["per_person_auc"].quantile(.25),
        cs["per_person_auc"].quantile(.75)))
    print("PERSONALISED (future days):  ", {k: round(v, 3) for k, v in ps["metrics"].items()})
    cal = calibrate(ps["model"].predict_proba(ps["val"][0])[:, 1], ps["val"][1],
                    ps["p"], ps["y"])
    print("  calibration Brier %.4f -> %.4f" % (cal["brier_before"], cal["brier_after"]))
    print("learning curve:\n", learning_curve(fb).to_string(index=False))
