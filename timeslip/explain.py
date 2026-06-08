"""
Explanation & causal-attribution engine for *Time Slip*.

This is where the project earns its name: we go beyond "phone use correlates
with boredom" to a per-person, per-slip account of *what was driving the risk*
and *by how much*.

Three layers:

1. SHAP (TreeExplainer on the deployed gradient-boosted risk model) gives the
   global ranking of features and, aggregated per person, an individual
   importance profile.

2. Counterfactual attribution. For every minute at which the person actually
   slipped, we ask: had this one cause been at the person's own *calm* baseline
   (their 10th/90th percentile, i.e. a realistically achievable better state),
   how much would the model's risk have dropped? The normalised drops form a
   "slip fingerprint" -- the share of the person's lapses attributable to
   boredom vs. fatigue vs. the phone pull vs. task aversiveness, etc. This is a
   leave-one-cause-out counterfactual on the fitted model, reported alongside
   the absolute risk reduction.

3. Ground-truth validation. Because the simulator's hazard is additive on the
   logit scale, we can compute each cause's *true* contribution at each slip and
   check that the counterfactual fingerprint recovers it (per-person Spearman).
   This tells us how much to trust the fingerprints on real data.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

from . import config as C

try:
    import shap
    _HAVE_SHAP = True
except Exception:                                   # pragma: no cover
    _HAVE_SHAP = False


# Human cause -> how to push its REALISTIC features toward a calmer state.
#   "low"  : set to the person's 10th percentile  (erosive feature, lower=calmer)
#   "high" : set to the person's 90th percentile  (protective feature)
#   "zero" : set to 0                              (notifications / phone away)
CAUSES: Dict[str, Dict[str, List[str]]] = {
    "Boredom":                  {"low": ["boredom_obs", "boredom_obs_roll20"]},
    "Stress":                   {"low": ["stress_obs", "stress_obs_roll20"]},
    "Fatigue":                  {"high": ["alertness_obs"]},
    "Hunger":                   {"low": ["hunger_obs", "since_meal"]},
    "Task aversiveness":        {"low": ["aversive", "deadline"]},
    "Low intrinsic motivation": {"high": ["intrinsic"]},
    "Time-on-task (vigilance)": {"low": ["time_on_task", "vigilance"]},
    "Phone pull":               {"zero": ["notif", "notif_15", "phone_in_reach"]},
}

# Oracle-space version: causes map straight onto the latent hazard inputs, so
# this measures whether the *method* recovers truth given clean inputs.
ORACLE_CAUSES: Dict[str, Dict[str, List[str]]] = {
    "Boredom":                  {"low": ["boredom"]},
    "Stress":                   {"low": ["stress"]},
    "Fatigue":                  {"low": ["fatigue"]},
    "Hunger":                   {"low": ["hunger"]},
    "Task aversiveness":        {"low": ["aversive"]},
    "Low intrinsic motivation": {"low": ["low_intrinsic"]},
    "Time-on-task (vigilance)": {"low": ["vigilance"]},
    "Phone pull":               {"low": ["urge"]},
}

# Map each cause to the ground-truth hazard term(s) for validation.
_CAUSE_TO_TRUE = {
    "Boredom":                  [("boredom", "boredom")],
    "Stress":                   [("stress", "stress")],
    "Fatigue":                  [("fatigue", "fatigue")],
    "Hunger":                   [("hunger", "hunger")],
    "Task aversiveness":        [("aversive", "aversive")],
    "Low intrinsic motivation": [("low_intrinsic", "low_intrinsic")],
    "Time-on-task (vigilance)": [("vigilance", "vigilance")],
    "Phone pull":               [("urge", "urge_eff")],
}


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def counterfactual_attribution(model, fb: Dict, feature_space: str = "real") -> Dict:
    """Per-person slip fingerprints from leave-one-cause-out counterfactuals.

    ``feature_space`` selects the design matrix and cause->column mapping:
      'real'   -> deployed self-loggable model (the practical scenario)
      'oracle' -> latent hazard inputs (tests method validity on clean data)
    """
    if feature_space == "oracle":
        X, causes = fb["X_oracle"], ORACLE_CAUSES
    else:
        X, causes = fb["X_real"], CAUSES
    meta = fb["meta"]
    cols = list(X.columns)

    onset_idx = np.where(fb["at_risk"]["slip_onset"].to_numpy() == 1)[0]
    Xo = X.iloc[onset_idx].reset_index(drop=True)
    pid = meta.iloc[onset_idx]["pid"].reset_index(drop=True)

    # per-person calm baselines
    q10 = X.groupby(meta["pid"].to_numpy()).quantile(0.10)
    q90 = X.groupby(meta["pid"].to_numpy()).quantile(0.90)

    base_risk = _predict(model, Xo)

    rows = []
    for cause, ops in causes.items():
        Xc = Xo.copy()
        for c in ops.get("low", []):
            if c in cols:
                Xc[c] = q10.loc[pid.values, c].to_numpy()
        for c in ops.get("high", []):
            if c in cols:
                Xc[c] = q90.loc[pid.values, c].to_numpy()
        for c in ops.get("zero", []):
            if c in cols:
                Xc[c] = 0.0
        risk_c = _predict(model, Xc)
        drop = np.clip(base_risk - risk_c, 0.0, None)
        rows.append(pd.DataFrame({"pid": pid.values, "cause": cause,
                                  "drop": drop}))
    long = pd.concat(rows, ignore_index=True)

    # per-person mean absolute reduction and normalised share
    per = (long.groupby(["pid", "cause"])["drop"].mean()
           .reset_index().rename(columns={"drop": "mean_risk_reduction"}))
    tot = per.groupby("pid")["mean_risk_reduction"].transform("sum").replace(0, np.nan)
    per["share"] = per["mean_risk_reduction"] / tot

    # overall (population) fingerprint
    overall = (per.groupby("cause")["mean_risk_reduction"].mean()
               .reset_index().sort_values("mean_risk_reduction", ascending=False))
    overall["share"] = overall["mean_risk_reduction"] / overall["mean_risk_reduction"].sum()

    # dominant cause per person
    dom = (per.sort_values("share", ascending=False)
           .groupby("pid").head(1).set_index("pid")["cause"].to_dict())

    return dict(per_person=per, overall=overall, dominant=dom,
                onset_index=onset_idx, base_risk=base_risk)


def _latent_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The exact latent inputs to the generative hazard, for the given rows."""
    L = pd.DataFrame(index=df.index)
    L["boredom"]       = df["boredom"].to_numpy()
    L["fatigue"]       = df["fatigue"].to_numpy()
    L["stress"]        = df["stress"].to_numpy()
    L["aversive"]      = df["aversive"].to_numpy()
    L["vigilance"]     = df["vigilance"].clip(upper=C.VIGILANCE_CAP).to_numpy()
    L["hunger"]        = df["hunger"].to_numpy()
    L["urge"]          = df["urge_eff"].to_numpy()
    L["low_intrinsic"] = (1.0 - df["intrinsic"]).to_numpy()
    L["low_mood"]      = (1.0 - df["mood"]).to_numpy()
    L["focus_reserve"] = df["focus_reserve"].to_numpy()
    return L


def _true_logit(L: pd.DataFrame, intercepts: np.ndarray) -> np.ndarray:
    H = C.HAZARD
    return (H["intercept"] + intercepts
            + H["boredom"] * L["boredom"] + H["fatigue"] * L["fatigue"]
            + H["stress"] * L["stress"] + H["aversive"] * L["aversive"]
            + H["vigilance"] * L["vigilance"] + H["hunger"] * L["hunger"]
            + H["urge"] * L["urge"] + H["low_intrinsic"] * L["low_intrinsic"]
            + H["low_mood"] * L["low_mood"]
            - H["self_control"] * L["focus_reserve"]).to_numpy()


def ground_truth_attribution(fb: Dict) -> pd.DataFrame:
    """True slip fingerprint: the *same* leave-one-cause-out counterfactual, but
    evaluated on the real generative hazard (config.HAZARD), so it is directly
    comparable to the model-based fingerprints."""
    ar = fb["at_risk"]
    personas = fb["personas"]
    inter = personas.set_index("pid")["intercept"]

    onset = ar[ar["slip_onset"] == 1].copy().reset_index(drop=True)
    L = _latent_frame(onset)
    pid = onset["pid"].reset_index(drop=True)
    inter_onset = pid.map(inter).to_numpy()

    Lall = _latent_frame(ar)
    q10 = Lall.groupby(ar["pid"].to_numpy()).quantile(0.10)

    base = 1.0 / (1.0 + np.exp(-_true_logit(L, inter_onset)))
    rows = []
    for cause, ops in ORACLE_CAUSES.items():
        Lc = L.copy()
        for col in ops["low"]:
            Lc[col] = q10.loc[pid.values, col].to_numpy()
        pc = 1.0 / (1.0 + np.exp(-_true_logit(Lc, inter_onset)))
        rows.append(pd.DataFrame({"pid": pid.values, "cause": cause,
                                  "drop": np.clip(base - pc, 0.0, None)}))
    long = pd.concat(rows, ignore_index=True)
    g = long.groupby(["pid", "cause"])["drop"].mean().reset_index()
    g["share"] = g["drop"] / g.groupby("pid")["drop"].transform("sum").replace(0, np.nan)
    return g.pivot(index="pid", columns="cause", values="share").fillna(0.0)


def validate_attribution(cf: Dict, truth: pd.DataFrame) -> Dict:
    """Per-person Spearman between counterfactual shares and true shares."""
    from scipy.stats import spearmanr
    causes = list(_CAUSE_TO_TRUE.keys())
    pivot = (cf["per_person"].pivot(index="pid", columns="cause", values="share")
             .reindex(columns=causes).fillna(0.0))
    truth = truth.reindex(columns=causes).reindex(index=pivot.index).fillna(0.0)
    per = {}
    for pid in pivot.index:
        rho, _ = spearmanr(pivot.loc[pid].to_numpy(), truth.loc[pid].to_numpy())
        per[pid] = float(rho)
    # overall: stack all (pid, cause) shares
    rho_all, _ = spearmanr(pivot.to_numpy().ravel(), truth.to_numpy().ravel())
    return dict(per_person=per, overall=float(rho_all),
                cf_shares=pivot, true_shares=truth)


def shap_analysis(model, fb: Dict, n_sample: int = 4000) -> Dict:
    """Global + per-person SHAP importances on the deployed risk model."""
    if not _HAVE_SHAP:                               # pragma: no cover
        return dict(available=False)
    X = fb["X_real"]
    te = fb["test_mask"]
    Xte = X[te].reset_index(drop=True)
    pid_te = fb["meta"][te]["pid"].reset_index(drop=True)

    rng = np.random.default_rng(C.GLOBAL_SEED)
    n = min(n_sample, len(Xte))
    samp = rng.choice(len(Xte), size=n, replace=False)
    Xs = Xte.iloc[samp]
    pid_s = pid_te.iloc[samp].to_numpy()

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):                         # some versions return [neg,pos]
        sv = sv[1]
    sv = np.asarray(sv)

    glob = (pd.DataFrame({"feature": X.columns,
                          "mean_abs_shap": np.abs(sv).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))

    # per-person mean |shap| profile (top features)
    per_person = {}
    for pid in np.unique(pid_s):
        m = pid_s == pid
        prof = pd.Series(np.abs(sv[m]).mean(axis=0), index=X.columns)
        per_person[pid] = prof.sort_values(ascending=False)

    return dict(available=True, global_importance=glob,
                shap_values=sv, X_sample=Xs, per_person=per_person)


if __name__ == "__main__":
    from .simulate import simulate_all
    from .features import build_features
    from .model import train_slip_model, train_oracle_model
    m, e, pt = simulate_all()
    fb = build_features(m, pt)
    res = train_slip_model(fb)
    ores = train_oracle_model(fb)
    truth = ground_truth_attribution(fb)

    cf = counterfactual_attribution(res["model"], fb, "real")
    cfo = counterfactual_attribution(ores["model"], fb, "oracle")
    print("=== population slip fingerprint (realistic / self-logged) ===")
    print(cf["overall"].round(4).to_string(index=False))
    print("\n=== dominant cause per person (realistic) ===")
    for pid, c in cf["dominant"].items():
        print(f"  {pid}: {c}")
    val = validate_attribution(cf, truth)
    valo = validate_attribution(cfo, truth)
    print(f"\nmethod validity  (oracle inputs)  overall Spearman={valo['overall']:.3f}")
    print(f"practical result (self-logged)    overall Spearman={val['overall']:.3f}")
    print("per-person (self-logged):",
          {k: round(v, 2) for k, v in val["per_person"].items()})
    sh = shap_analysis(res["model"], fb)
    if sh["available"]:
        print("\ntop-10 SHAP features:")
        print(sh["global_importance"].head(10).to_string(index=False))
