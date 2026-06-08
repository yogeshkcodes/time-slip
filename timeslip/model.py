"""
Predictive + inferential models for *Time Slip*.

Three things happen here:

1. ``train_slip_model`` fits a gradient-boosted classifier (XGBoost) that
   predicts the probability of a slip onset in the next minute from the
   REALISTIC (self-loggable) features, and compares it against two transparent
   baselines (an L2 logistic regression and a notifications-only score). We
   report discrimination (ROC-AUC, PR-AUC) and calibration (Brier score) on the
   held-out days.

2. ``recover_coefficients`` fits a logistic regression on the ORACLE features
   (the exact hazard inputs) and compares the recovered coefficients with the
   ground-truth ``config.HAZARD`` values. This is the validation that the
   approach measures *causes*, not just correlations: if the pipeline cannot
   recover known coefficients from simulated data, we should not trust it on
   real data. Per-person intercepts are included to absorb the random effects
   used in the simulator and are excluded from the comparison.

3. ``fit_discrete_hazard`` exposes the same logistic hazard on the realistic
   features as an interpretable, deployable model with odds-ratio outputs.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

from . import config as C

try:
    from xgboost import XGBClassifier
    _HAVE_XGB = True
except Exception:                                   # pragma: no cover
    _HAVE_XGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier


def train_slip_model(fb: Dict) -> Dict:
    """Deployable model: predict whether a slip occurs in the next horizon.

    The target is ``y_window`` (any slip onset within the next
    ``config.SLIP_HORIZON_MIN`` minutes), which is both more learnable and more
    practically useful than the exact-minute label -- a real assistant would
    warn you that the *next stretch* is high-risk, not that minute 12:34:00
    specifically. Per-minute onset remains the target for the mechanistic
    recovery and survival analyses.
    """
    X = fb["X_real"]
    y = fb["y_window"]
    tr, te = fb["train_mask"], fb["test_mask"]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    pos = max(1, int(ytr.sum()))
    neg = int((ytr == 0).sum())
    spw = neg / pos

    if _HAVE_XGB:
        model = XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=4,
            reg_lambda=1.5, gamma=0.5, scale_pos_weight=spw,
            tree_method="hist", eval_metric="logloss",
            random_state=C.GLOBAL_SEED, n_jobs=0,
        )
        model.fit(Xtr, ytr)
        family = "XGBoost"
    else:                                           # pragma: no cover
        model = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=450,
            class_weight="balanced", random_state=C.GLOBAL_SEED)
        model.fit(Xtr, ytr)
        family = "HistGBDT"

    p_te = model.predict_proba(Xte)[:, 1]

    # ---- baselines ----
    logit = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(max_iter=2000,
                                                class_weight="balanced"))])
    logit.fit(Xtr, ytr)
    p_logit = logit.predict_proba(Xte)[:, 1]

    # notifications-only score (does "being pinged" alone explain slips?)
    notif_score = Xte["notif_15"].to_numpy() + 0.1 * Xte["notif"].to_numpy()

    metrics = {
        "model_family": family,
        "target": f"slip within next {fb['horizon_min']} min",
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "test_positives": int(yte.sum()),
        "roc_auc":  float(roc_auc_score(yte, p_te)),
        "pr_auc":   float(average_precision_score(yte, p_te)),
        "brier":    float(brier_score_loss(yte, p_te)),
        "base_rate": float(yte.mean()),
        "baseline_logit_roc":  float(roc_auc_score(yte, p_logit)),
        "baseline_logit_pr":   float(average_precision_score(yte, p_logit)),
        "baseline_notif_roc":  float(roc_auc_score(yte, notif_score)),
        "baseline_notif_pr":   float(average_precision_score(yte, notif_score)),
    }
    return dict(model=model, logit=logit, metrics=metrics,
                p_test=p_te, p_logit=p_logit, p_notif=notif_score,
                y_test=yte, X_test=Xte)


def train_oracle_model(fb: Dict) -> Dict:
    """Same risk model but trained on the latent ORACLE features.

    Used by explain.py to show that the counterfactual-attribution method
    recovers the true per-person drivers when given clean inputs -- isolating
    'is the method valid?' from 'how noisy is self-logged data?'.
    """
    X = fb["X_oracle"]
    y = fb["y_window"]
    tr, te = fb["train_mask"], fb["test_mask"]
    pos = max(1, int(y[tr].sum())); neg = int((y[tr] == 0).sum())
    if _HAVE_XGB:
        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.06, subsample=0.85,
            colsample_bytree=0.9, min_child_weight=4, reg_lambda=1.5,
            scale_pos_weight=neg / pos, tree_method="hist",
            eval_metric="logloss", random_state=C.GLOBAL_SEED, n_jobs=0)
    else:                                            # pragma: no cover
        model = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=400,
            class_weight="balanced", random_state=C.GLOBAL_SEED)
    model.fit(X[tr], y[tr])
    auc = float(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))
    return dict(model=model, oracle_test_auc=auc)


def train_attribution_model(fb: Dict, space: str = "real"):
    """Additive (logistic) surrogate used ONLY for counterfactual attribution.

    Prediction and explanation deliberately use different models. A gradient-
    boosted tree gives the best discrimination, but tree-based one-feature
    counterfactual ablation is unreliable for attribution (non-additive,
    saturating). Because the true slip process is additive on the logit scale, a
    standardised logistic regression yields faithful, monotonic per-feature
    counterfactuals -- and recovers the true per-person cause ranking far better
    (per-person Spearman ~0.75 vs ~0.25 for the tree). This is the model fed to
    ``explain.counterfactual_attribution``; the XGBoost model remains the
    predictor for the headline accuracy metrics.
    """
    X = fb["X_oracle"] if space == "oracle" else fb["X_real"]
    y = fb["y_window"]
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=4000,
                                               class_weight="balanced"))])
    pipe.fit(X, y)
    return pipe


def recover_coefficients(fb: Dict) -> Dict:
    """Fit logistic on oracle features; compare to ground-truth HAZARD."""
    Xo = fb["X_oracle"].copy()
    y = fb["y"]
    at_risk = fb["at_risk"]

    # per-person dummies to soak up the simulator's random intercepts
    pid_dummies = pd.get_dummies(at_risk["pid"], prefix="pid", dtype=float)
    design = pd.concat([Xo.reset_index(drop=True),
                        pid_dummies.reset_index(drop=True)], axis=1)

    # near-unregularised logistic so coefficients are directly comparable
    lr = LogisticRegression(C=1e9, max_iter=5000, solver="lbfgs")
    lr.fit(design.to_numpy(), y)

    coefs = dict(zip(design.columns, lr.coef_.ravel()))
    feat_names = list(Xo.columns)
    recovered = np.array([coefs[f] for f in feat_names])
    true = np.array([fb["true_beta"][f] for f in feat_names])

    rho, _ = spearmanr(recovered, true)
    pear = float(np.corrcoef(recovered, true)[0, 1])
    sign_agree = float(np.mean(np.sign(recovered) == np.sign(true)))

    table = pd.DataFrame({
        "feature": feat_names,
        "true_beta": true,
        "recovered_beta": recovered,
        "true_rank": pd.Series(np.abs(true)).rank(ascending=False).astype(int).values,
        "recovered_rank": pd.Series(np.abs(recovered)).rank(ascending=False).astype(int).values,
    }).sort_values("true_rank")

    return dict(table=table, spearman=float(rho), pearson=pear,
                sign_agreement=sign_agree, model=lr,
                feature_names=feat_names)


def fit_discrete_hazard(fb: Dict) -> Dict:
    """Interpretable logistic hazard on realistic features -> odds ratios."""
    X = fb["X_real"]
    y = fb["y"]
    tr = fb["train_mask"]
    sc = StandardScaler()
    Xz = sc.fit_transform(X[tr])
    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    lr.fit(Xz, y[tr])
    odds = pd.DataFrame({
        "feature": X.columns,
        "coef_per_sd": lr.coef_.ravel(),
        "odds_ratio_per_sd": np.exp(lr.coef_.ravel()),
    }).sort_values("coef_per_sd", key=np.abs, ascending=False)
    return dict(scaler=sc, model=lr, odds_ratios=odds)


if __name__ == "__main__":
    from .simulate import simulate_all
    from .features import build_features
    m, e, pt = simulate_all()
    fb = build_features(m, pt)
    res = train_slip_model(fb)
    for k, v in res["metrics"].items():
        print(f"{k:>22}: {v}")
    print("\n--- coefficient recovery ---")
    rec = recover_coefficients(fb)
    print(rec["table"].to_string(index=False))
    print(f"\nSpearman(recovered, true) = {rec['spearman']:.3f} | "
          f"sign agreement = {rec['sign_agreement']:.2f}")
