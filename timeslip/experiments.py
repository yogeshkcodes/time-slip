"""
N-of-1 experiment engine for Time Slip.

This is what turns a model *claim* ("phone-away would cut your slips ~40%") into
*your own measured evidence*. The Attention Account prescribes one change; you
register it, keep logging, and this engine measures the effect on YOUR data with
the rigour of a single-subject experiment:

  * weekday-controlled permutation test  - behaviour varies by day-of-week, so we
    remove each weekday's mean before permuting the arm labels (10k permutations).
    No distributional assumptions; the p-value is exact to the permutation set.
  * bootstrap 95% CI on the effect        - so you see the range, not just a point.
  * minimum detectable effect (MDE)        - the honesty layer: given how noisy
    your days are and how many you've logged, what is the smallest change you
    *could* have detected? Stops us declaring "no effect" when we're just
    under-powered, and tells you how many more days you'd need.

A verdict is only "improved"/"worsened" when the test clears p<0.05 AND both
arms have enough days. Otherwise it says "no detectable change" or "keep logging
(need ~N more days)". Results are stored in a small JSON registry so each run
reports the running tally.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

# metric name -> how to reduce one day's log rows to a single number
METRICS = {
    "slips_per_day":    lambda g: float(g["slip"].sum()),
    "min_lost_per_day": lambda g: float(pd.to_numeric(g.get("slip_minutes", 0),
                                                       errors="coerce").fillna(0).sum()),
    "phone_min_per_day": lambda g: float(pd.to_numeric(
        g.loc[g.get("slip_channel", "") == "phone", "slip_minutes"],
        errors="coerce").fillna(0).sum()) if "slip_channel" in g else 0.0,
}
METRIC_LABEL = {"slips_per_day": "slips/day",
                "min_lost_per_day": "minutes lost/day",
                "phone_min_per_day": "phone minutes/day"}

MIN_DAYS_PER_ARM = 5          # below this, never declare a result
PERM_N = 10000


@dataclass
class Experiment:
    id: str
    cause: str                # the fingerprint cause it targets
    change: str               # the one behaviour change, in plain English
    metric: str               # key in METRICS
    start_date: str           # ISO date the intervention began
    measure: str = ""         # how success is judged (free text)
    created: str = ""


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def load_registry(path: str) -> List[Experiment]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [Experiment(**e) for e in json.load(f)]


def save_registry(path: str, exps: List[Experiment]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in exps], f, indent=2)


def start_experiment(path: str, cause: str, change: str, metric: str,
                     measure: str = "", start: Optional[str] = None) -> Experiment:
    exps = load_registry(path)
    start = start or date.today().isoformat()
    eid = f"{metric}_{start}_{len(exps)+1}"
    exp = Experiment(id=eid, cause=cause, change=change, metric=metric,
                     start_date=start, measure=measure,
                     created=date.today().isoformat())
    exps.append(exp)
    save_registry(path, exps)
    return exp


# --------------------------------------------------------------------------- #
# the statistics
# --------------------------------------------------------------------------- #
def _daily(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per logged day: the metric value + weekday."""
    fn = METRICS[metric]
    rows = []
    for d, g in df.groupby("date"):
        rows.append({"date": str(d), "value": fn(g),
                     "wd": pd.Timestamp(d).dayofweek})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def _weekday_resid(vals: np.ndarray, wd: np.ndarray) -> np.ndarray:
    """Subtract each weekday's mean so permutation isn't confounded by day mix."""
    resid = vals.astype(float).copy()
    for w in np.unique(wd):
        m = wd == w
        resid[m] = resid[m] - resid[m].mean()
    return resid


def _perm_p(resid: np.ndarray, arm: np.ndarray, obs: float,
            rng: np.random.Generator, n: int = PERM_N) -> float:
    count = 0
    for _ in range(n):
        perm = rng.permutation(arm)
        diff = resid[perm == 1].mean() - resid[perm == 0].mean()
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n + 1)


def _bootstrap_ci(base: np.ndarray, interv: np.ndarray,
                  rng: np.random.Generator, n: int = 5000):
    diffs = np.empty(n)
    for i in range(n):
        b = rng.choice(base, size=len(base), replace=True)
        v = rng.choice(interv, size=len(interv), replace=True)
        diffs[i] = v.mean() - b.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _mde(base: np.ndarray, interv: np.ndarray) -> float:
    """Minimum detectable effect (abs units) at 80% power, alpha .05 two-sided."""
    n0, n1 = len(base), len(interv)
    if n0 < 2 or n1 < 2:
        return float("inf")
    sp = np.sqrt(((n0 - 1) * base.var(ddof=1) + (n1 - 1) * interv.var(ddof=1))
                 / max(1, n0 + n1 - 2))
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    return float(z * sp * np.sqrt(1.0 / n0 + 1.0 / n1))


def evaluate_experiment(df: pd.DataFrame, exp: Experiment,
                        seed: int = 12345) -> Dict:
    """Measure an experiment's effect on the person's own logged data."""
    if "date" not in df.columns or "slip" not in df.columns:
        return dict(ok=False, reason="log needs 'date' and 'slip' columns")
    daily = _daily(df, exp.metric)
    daily["arm"] = (daily["date"] >= exp.start_date).astype(int)
    base = daily.loc[daily["arm"] == 0, "value"].to_numpy()
    interv = daily.loc[daily["arm"] == 1, "value"].to_numpy()
    n0, n1 = len(base), len(interv)
    res = dict(ok=True, id=exp.id, cause=exp.cause, change=exp.change,
               metric=exp.metric, metric_label=METRIC_LABEL.get(exp.metric, exp.metric),
               start_date=exp.start_date, n_baseline=n0, n_intervention=n1,
               baseline_mean=float(base.mean()) if n0 else float("nan"),
               intervention_mean=float(interv.mean()) if n1 else float("nan"))

    if n0 < MIN_DAYS_PER_ARM or n1 < MIN_DAYS_PER_ARM:
        need = max(0, MIN_DAYS_PER_ARM - min(n0, n1))
        res.update(verdict="keep logging",
                   detail=f"need ~{max(need, MIN_DAYS_PER_ARM - n1)} more "
                          f"intervention day(s) (have {n1}, baseline {n0}).")
        return res

    rng = np.random.default_rng(seed)
    diff = float(interv.mean() - base.mean())
    pct = 100 * diff / base.mean() if base.mean() else float("nan")
    resid = _weekday_resid(daily["value"].to_numpy(), daily["wd"].to_numpy())
    obs_resid = resid[daily["arm"].to_numpy() == 1].mean() - \
        resid[daily["arm"].to_numpy() == 0].mean()
    p = _perm_p(resid, daily["arm"].to_numpy(), obs_resid, rng)
    lo, hi = _bootstrap_ci(base, interv, rng)
    mde = _mde(base, interv)
    mde_pct = 100 * mde / base.mean() if base.mean() else float("inf")

    if p < 0.05 and diff < 0:
        verdict = "improved"
    elif p < 0.05 and diff > 0:
        verdict = "worsened"
    elif abs(diff) < mde:
        verdict = "no detectable change"
    else:
        verdict = "inconclusive"
    res.update(diff=diff, pct_change=pct, p_value=float(p),
               ci_lo=lo, ci_hi=hi, mde=mde, mde_pct=mde_pct, verdict=verdict)
    return res


def narrative_line(res: Dict) -> str:
    """One plain-English sentence summarising an evaluated experiment."""
    if not res.get("ok"):
        return f"Experiment could not be scored: {res.get('reason')}."
    lbl = res["metric_label"]
    if res["verdict"] == "keep logging":
        return (f"**Experiment in progress** ({res['change']}): {res['detail']} "
                "Result will appear once there's enough data.")
    direction = "down" if res["diff"] < 0 else "up"
    sig = (f"p={res['p_value']:.3f}" if res["p_value"] >= 0.001 else "p<0.001")
    pct = abs(res["pct_change"])
    ci = (f"95% CI [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}] {lbl}")
    if res["verdict"] in ("improved", "worsened"):
        word = "cut" if res["verdict"] == "improved" else "raised"
        return (f"**Your experiment worked: it {word} your {lbl} by "
                f"{pct:.0f}%** ({res['baseline_mean']:.1f} -> "
                f"{res['intervention_mean']:.1f}; {sig}, {ci}). "
                "This is YOUR data, not a model prediction.")
    if res["verdict"] == "no detectable change":
        return (f"No detectable change in {lbl} ({res['baseline_mean']:.1f} -> "
                f"{res['intervention_mean']:.1f}; {sig}). With this many days you "
                f"could only have detected a change of >={res['mde_pct']:.0f}% - "
                "log more to resolve smaller effects.")
    return (f"Trend ({direction} {pct:.0f}% in {lbl}) but not significant yet "
            f"({sig}); keep logging.")


def evaluate_active(df: pd.DataFrame, registry_path: str) -> List[Dict]:
    return [evaluate_experiment(df, e) for e in load_registry(registry_path)]


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # self-test: the engine must (a) detect a real effect and (b) NOT
    # hallucinate one when there is none.
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-04-01", periods=28).astype(str)

    def make_log(effect):
        rows = []
        for i, d in enumerate(days):
            arm = i >= 14
            lam = 30 * (1 - effect) if arm else 30
            nslip = rng.poisson(lam)
            for _ in range(nslip):
                rows.append({"date": d, "slip": 1, "slip_minutes": 5,
                             "slip_channel": "phone"})
            rows.append({"date": d, "slip": 0, "slip_minutes": 0, "slip_channel": ""})
        return pd.DataFrame(rows)

    exp = Experiment(id="t", cause="Phone pull", change="phone in other room",
                     metric="slips_per_day", start_date=str(days[14]))
    print("REAL 35% effect ->", narrative_line(evaluate_experiment(make_log(0.35), exp)))
    print("\nNULL 0% effect  ->", narrative_line(evaluate_experiment(make_log(0.0), exp)))
