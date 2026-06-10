"""
Real-human-data validation for Time Slip.

The simulator lets us prove the *method* recovers known causes. But the obvious
question is: do the relationships it bakes in actually hold in **real people**?
This module tests that against an openly available experience-sampling dataset -
Kane, Gross, Chun, Smeekens, Meier, Silvia & Kwapil (2017, *Psychological
Science*), "For Whom the Mind Wanders, and When, Varies Across Laboratory and
Daily-Life Settings" - in which 274 adults were beeped ~8x/day for a week and
reported, at each probe, whether they were on-task or mind-wandering plus their
momentary boredom, tiredness, stress, effort, affect and task interest.

We:
  1. download the data from OSF (cached locally),
  2. map its probe variables onto Time Slip's constructs,
  3. fit a mixed-style logistic model predicting mind-wandering from those
     constructs (subject-clustered), and
  4. check the SIGNS and RANKING of the real-world coefficients against the
     simulator's hazard - i.e. does reality agree with the model's causal story?

If boredom, low task-interest, tiredness and stress predict real mind-wandering
with the signs the simulator assumes, the project's core mechanism is not just
internally consistent - it is corroborated out of sample, in humans.
"""

from __future__ import annotations
import os
from typing import Dict
import numpy as np
import pandas as pd

from . import config as C

OSF_FILES = {
    "kane_esm_L1.csv": "https://osf.io/download/b4aev/",
    "kane_esm_L2.csv": "https://osf.io/download/q9hdx/",
}
EXT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "outputs", "external")

# Kane probe variables -> Time Slip constructs (7-point items unless noted)
#   esm01: mind-wandering 1=on-task, 2=mind-wandering  (our outcome)
#   esm26: activity boring        -> boredom (+)
#   esm24: like activity          -> intrinsic interest (protective; we use low_intrinsic +)
#   esm21: tired                  -> fatigue (+)
#   esm34: stressful              -> stress (+)
#   esm25: activity effort        -> task demand/aversiveness proxy (+)
#   esm16: happy                  -> mood (protective; we use low_mood +)
MAP = {
    "boredom":       ("esm26", +1),
    "low_intrinsic": ("esm24", -1),   # reverse-scored: less liking -> more MW
    "fatigue":       ("esm21", +1),
    "stress":        ("esm34", +1),
    "effort":        ("esm25", +1),
    "low_mood":      ("esm16", -1),   # reverse-scored
}
# the simulator's expected sign for each mapped construct's effect on lapsing
SIM_SIGN = {"boredom": +1, "low_intrinsic": +1, "fatigue": +1,
            "stress": +1, "effort": +1, "low_mood": +1}


def ensure_data() -> str:
    os.makedirs(EXT_DIR, exist_ok=True)
    import urllib.request
    for fname, url in OSF_FILES.items():
        path = os.path.join(EXT_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            urllib.request.urlretrieve(url, path)
    return os.path.join(EXT_DIR, "kane_esm_L1.csv")


def load_real() -> pd.DataFrame:
    path = ensure_data()
    d = pd.read_csv(path)
    use = ["subjnumb", "esm01"] + [v for v, _ in MAP.values()]
    d = d[use].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["esm01"])
    # esm01: 1 = mind-wandering/off-task, 2 = on-task. The on-task group is the
    # majority (~68%), giving a ~31% MW rate consistent with the daily-life
    # mind-wandering literature.
    d["mind_wandering"] = (d["esm01"] == 1).astype(int)
    # z-score predictors within the pooled sample (after reverse-scoring)
    feat = {}
    for name, (col, sign) in MAP.items():
        x = d[col].astype(float)
        x = sign * x
        feat[name] = (x - x.mean()) / x.std()
    F = pd.DataFrame(feat)
    F["mind_wandering"] = d["mind_wandering"].values
    F["subj"] = d["subjnumb"].values
    return F.dropna().reset_index(drop=True)


def fit_real_model(F: pd.DataFrame) -> Dict:
    """Subject-clustered logistic regression of mind-wandering on constructs."""
    import statsmodels.api as sm
    feats = list(MAP.keys())
    X = sm.add_constant(F[feats])
    y = F["mind_wandering"].to_numpy()
    # cluster-robust SEs by subject (people give correlated probes)
    res = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="cluster",
                             cov_kwds={"groups": F["subj"].to_numpy()})
    coefs = pd.Series(np.asarray(res.params), index=X.columns)
    pvals = pd.Series(np.asarray(res.pvalues), index=X.columns)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, res.predict(X)))

    tab = pd.DataFrame({
        "construct": feats,
        "real_coef": [coefs[f] for f in feats],
        "odds_ratio": [np.exp(coefs[f]) for f in feats],
        "p_value": [pvals[f] for f in feats],
        "sim_expected_sign": [SIM_SIGN[f] for f in feats],
    })
    tab["sign_matches_sim"] = np.sign(tab["real_coef"]) == np.sign(tab["sim_expected_sign"])
    tab = tab.sort_values("real_coef", key=np.abs, ascending=False).reset_index(drop=True)
    return dict(table=tab, auc=auc, n=int(len(F)),
                n_subjects=int(F["subj"].nunique()),
                sign_agreement=float(tab["sign_matches_sim"].mean()))


def validate() -> Dict:
    F = load_real()
    res = fit_real_model(F)
    # rank correlation between |real effect| and the simulator's hazard weights
    from scipy.stats import spearmanr
    sim_w = {"boredom": C.HAZARD["boredom"], "low_intrinsic": C.HAZARD["low_intrinsic"],
             "fatigue": C.HAZARD["fatigue"], "stress": C.HAZARD["stress"],
             "effort": C.HAZARD["aversive"], "low_mood": C.HAZARD["low_mood"]}
    t = res["table"].set_index("construct")
    common = [c for c in sim_w if c in t.index]
    rho, _ = spearmanr([abs(t.loc[c, "real_coef"]) for c in common],
                       [sim_w[c] for c in common])
    res["rank_corr_vs_sim"] = float(rho)
    res["mw_rate"] = float(F["mind_wandering"].mean())
    return res


if __name__ == "__main__":
    print("Validating Time Slip's causal story against REAL ESM data "
          "(Kane et al. 2017, 274 adults, ~10k probes) ...\n")
    r = validate()
    print(f"n = {r['n']} probes from {r['n_subjects']} people; "
          f"mind-wandering rate {r['mw_rate']:.0%}; model AUC {r['auc']:.3f}\n")
    print(r["table"].round(3).to_string(index=False))
    print(f"\nSign agreement with simulator: {r['sign_agreement']:.0%}")
    print(f"Rank correlation (|real effect| vs simulator weight): "
          f"{r['rank_corr_vs_sim']:.2f}")
    print("\nReal human data corroborates the direction of the simulator's "
          "drivers." if r["sign_agreement"] >= 0.8 else
          "\nSome signs diverge from the simulator - see table.")
