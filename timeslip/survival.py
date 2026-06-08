"""
Survival / time-to-lapse analysis for *Time Slip*.

Two complementary, statistically-correct views of *when* attention gives way:

  Kaplan-Meier      The probability of still being on-task after t minutes of
                    continuous focused work ("attention survival curve"), split
                    by the internal state present when the work began.

  Discrete-time     A pooled logistic hazard over every at-risk focus minute,
  hazard            with time-on-task entered explicitly (linear + quadratic) to
                    capture the vigilance decrement. exp(coefficient) is a
                    hazard ratio. This is the correct survival model for
                    minute-resolution data and -- unlike a Cox model fed
                    spell-*mean* covariates -- it is not biased by the fact that
                    states such as boredom themselves rise with time-on-task.

We also expose the raw empirical hazard-vs-time-on-task curve.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

try:
    from lifelines import KaplanMeierFitter
    _HAVE_LIFELINES = True
except Exception:                                   # pragma: no cover
    _HAVE_LIFELINES = False

try:
    import statsmodels.api as sm
    _HAVE_SM = True
except Exception:                                   # pragma: no cover
    _HAVE_SM = False

from . import config as C


# covariates for the discrete-time hazard (latent states = mechanistic view)
_HAZ_COVARS = ["boredom", "fatigue", "stress", "aversive", "hunger",
               "urge_eff", "low_intrinsic", "low_mood", "focus_reserve"]


def build_spells(at_risk: pd.DataFrame) -> pd.DataFrame:
    """Focused-work spells: time-on-task until a slip (event) or reset (censor).

    Records the internal state at the *start* of the spell (for unbiased
    grouping in Kaplan-Meier), never the mean over the spell.
    """
    df = at_risk[at_risk["focus"] == 1].copy()
    df = df.sort_values(["pid", "day", "clock_min"])
    spells: List[dict] = []
    for (pid, day), g in df.groupby(["pid", "day"], sort=False):
        g = g.reset_index(drop=True)
        start, tot = 0, len(g)
        for i in range(tot):
            is_last = (i == tot - 1)
            broke = (not is_last) and (g.loc[i + 1, "time_on_task"] <= g.loc[i, "time_on_task"])
            event = int(g.loc[i, "slip_onset"] == 1)
            if event or broke or is_last:
                seg = g.iloc[start:i + 1]
                spells.append(dict(
                    pid=pid, day=int(day),
                    duration=max(1, int(seg["time_on_task"].iloc[-1]) + 1),
                    event=event, activity=seg["activity"].iloc[-1],
                    boredom_start=float(seg["boredom"].iloc[0]),
                    aversive_start=float(seg["aversive"].iloc[0]),
                    fatigue_start=float(seg["fatigue"].iloc[0]),
                ))
                start = i + 1
    return pd.DataFrame(spells)


def km_curves(spells: pd.DataFrame, split: str = "aversive_start") -> Dict:
    """Kaplan-Meier attention-survival curves, overall and by a tertile split."""
    if not _HAVE_LIFELINES or spells.empty:          # pragma: no cover
        return dict(available=False)
    out = {"available": True, "groups": {}}
    kmf = KaplanMeierFitter()
    kmf.fit(spells["duration"], spells["event"], label="overall")
    out["overall"] = kmf.survival_function_.reset_index()
    out["median_overall"] = float(kmf.median_survival_time_)

    q1, q2 = spells[split].quantile([0.33, 0.66])
    bins = pd.cut(spells[split], [-np.inf, q1, q2, np.inf],
                  labels=["low", "mid", "high"])
    for lab in ["low", "high"]:
        sub = spells[bins == lab]
        if len(sub) < 20:
            continue
        k = KaplanMeierFitter()
        k.fit(sub["duration"], sub["event"], label=f"{split}={lab}")
        out["groups"][lab] = dict(
            curve=k.survival_function_.reset_index(),
            median=float(k.median_survival_time_), n=int(len(sub)))
    out["split_on"] = split
    return out


def fit_discrete_time_hazard(at_risk: pd.DataFrame) -> Dict:
    """Pooled logistic hazard over focus minutes -> hazard ratios with CIs."""
    f = at_risk[at_risk["focus"] == 1].copy()
    f["low_intrinsic"] = 1.0 - f["intrinsic"]
    f["low_mood"] = 1.0 - f["mood"]
    f["tot"] = f["time_on_task"].clip(upper=120) / 60.0          # hours on task
    f["tot2"] = f["tot"] ** 2

    cov = _HAZ_COVARS + ["tot", "tot2"]
    X = f[cov].copy()
    means, sds = X.mean(), X.std().replace(0, 1.0)
    Xz = (X - means) / sds
    y = f["slip_onset"].astype(int).to_numpy()

    if not _HAVE_SM:                                 # pragma: no cover
        return dict(available=False)

    Xc = sm.add_constant(Xz)
    res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
    ci = res.conf_int()
    tab = pd.DataFrame({
        "covariate": Xc.columns,
        "coef": res.params.values,
        "hazard_ratio": np.exp(res.params.values),
        "hr_lo": np.exp(ci[0].values),
        "hr_hi": np.exp(ci[1].values),
        "p": res.pvalues.values,
    })
    tab = tab[tab["covariate"] != "const"].sort_values("hazard_ratio", ascending=False)
    # pseudo R2 and discrimination
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, res.predict(Xc)))
    return dict(available=True, summary=tab.reset_index(drop=True),
                pseudo_r2=float(res.prsquared), auc=auc,
                n=int(len(y)), n_events=int(y.sum()))


def vigilance_curve(at_risk: pd.DataFrame, max_min: int = 80, step: int = 5) -> pd.DataFrame:
    """Empirical per-minute slip hazard as a function of time-on-task (focus)."""
    f = at_risk[at_risk["focus"] == 1].copy()
    f["tot_bin"] = (f["time_on_task"] // step * step).clip(upper=max_min)
    out = (f.groupby("tot_bin")
             .agg(hazard=("slip_onset", "mean"), n=("slip_onset", "size"))
             .reset_index())
    return out[out["n"] >= 30]


if __name__ == "__main__":
    from .simulate import simulate_all
    from .features import build_features
    m, e, pt = simulate_all()
    fb = build_features(m, pt)
    sp = build_spells(fb["at_risk"])
    print("focus spells:", sp.shape, "| events:", int(sp["event"].sum()))
    haz = fit_discrete_time_hazard(fb["at_risk"])
    if haz["available"]:
        print(f"\ndiscrete-time hazard  AUC={haz['auc']:.3f}  "
              f"pseudoR2={haz['pseudo_r2']:.3f}  "
              f"({haz['n_events']}/{haz['n']} minute-events)")
        print(haz["summary"].round(3).to_string(index=False))
    km = km_curves(sp)
    if km["available"]:
        print(f"\nmedian time-on-task to first slip: {km['median_overall']:.0f} min")
        for lab, d in km["groups"].items():
            print(f"  boredom_start={lab}: median={d['median']:.0f} min (n={d['n']})")
