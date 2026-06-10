"""
Real-data pipeline for *Time Slip* — analyse a person's own logged routine.

This is the bridge from "validated method" to "use it on yourself". Given a
filled-in self-log (see ``schema.py`` / the CSV template), it produces:

  Level 1 — DESCRIPTIVE (always, if you logged your slips): where and when your
            slips actually happen, the channel mix, time lost, and which felt
            states are elevated in the moments you slip vs. the moments you don't.
            This is just a faithful summary of YOUR data — no model, no guessing.

  Level 2 — PERSONAL MODEL (only if you have enough logged slips): a logistic
            risk model fit on *your* rows, with a counterfactual "slip
            fingerprint" — the share of your slips attributable to each cause,
            using the same machinery validated on the simulation.

If there isn't enough data for Level 2, it says so plainly and tells you how much
more to log, rather than inventing a fragile result.

CLI:  python analyze_me.py [path/to/my_log.csv]
      (with no path it generates a realistic example log and analyses that)
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from . import config as C
from . import schema as S

sns.set_theme(style="whitegrid", context="talk")

# minimum data before we trust a personal model
MIN_ROWS_FOR_MODEL = 30
MIN_SLIPS_FOR_MODEL = 6

CHANNEL_COLORS = {
    "phone": "#e4572e", "mind_wandering": "#76b041", "task_switch": "#2e86ab",
    "snack": "#f0a202", "social": "#9b5de5",
}

# cause -> how to nudge the person's features toward a calmer baseline
CAUSES_REAL: Dict[str, Dict[str, List[str]]] = {
    "Boredom":                  {"low": ["boredom_obs"]},
    "Stress":                   {"low": ["stress_obs"]},
    "Fatigue":                  {"high": ["alertness_obs", "energy_obs"]},
    "Hunger":                   {"low": ["hunger_obs"]},
    "Task aversiveness":        {"low": ["aversive", "deadline"]},
    "Low intrinsic motivation": {"high": ["intrinsic"]},
    "Phone pull":               {"zero": ["notif", "phone_in_reach"]},
    "Time-on-task (vigilance)": {"low": ["time_on_task"]},
}

# states whose felt level we contrast between slip and non-slip moments (Level 1)
_CONTRAST = ["boredom_obs", "stress_obs", "alertness_obs", "energy_obs",
             "hunger_obs", "aversive", "intrinsic", "deadline", "difficulty"]
_PRETTY = {
    "boredom_obs": "boredom", "stress_obs": "stress", "alertness_obs": "alertness",
    "energy_obs": "energy", "hunger_obs": "hunger", "aversive": "task aversiveness",
    "intrinsic": "intrinsic interest", "deadline": "deadline pressure",
    "difficulty": "task difficulty", "time_on_task": "time-on-task",
}


# --------------------------------------------------------------------------- #
def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    problems = S.validate_self_log(df)
    if problems:
        raise ValueError("Your log has issues:\n  - " + "\n  - ".join(problems))
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validated self-log -> model-ready features (self-loggable only)."""
    d = S.to_model_frame(df).copy()           # rescales Likert 1-5 -> 0-1, renames
    d = d.sort_values(["date", "clock_min"]).reset_index(drop=True)
    d["hour"] = d["clock_min"] / 60.0
    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24.0)

    # within-day temporal features (vectorised; day boundaries reset everything)
    d["minutes_awake"] = d["clock_min"] - d.groupby("date")["clock_min"].transform("first")
    # time-on-task: minutes since task_type changed (a new day also starts a run)
    change = (d["task_type"] != d["task_type"].shift()) | (d["date"] != d["date"].shift())
    runs = change.cumsum()
    d["time_on_task"] = (d["clock_min"]
                         - d.groupby(runs)["clock_min"].transform("first")).fillna(0.0)
    # minutes since last meal, within the day
    meal = d["task_type"].astype(str).str.contains("meal", case=False, na=False)
    last_meal = d["clock_min"].where(meal).groupby(d["date"]).ffill()
    d["since_meal"] = (d["clock_min"] - last_meal).fillna(d["minutes_awake"])
    return d


def _feature_matrix(d: pd.DataFrame) -> pd.DataFrame:
    numeric = ["boredom_obs", "stress_obs", "energy_obs", "hunger_obs",
               "alertness_obs", "difficulty", "intrinsic", "aversive", "deadline",
               "open_tasks", "phone_in_reach", "notif", "time_on_task",
               "minutes_awake", "since_meal", "hour_sin", "hour_cos"]
    numeric = [c for c in numeric if c in d.columns]
    X = d[numeric].astype(float).copy()
    if "task_type" in d.columns:
        cat = pd.get_dummies(d["task_type"], prefix="task", dtype=float)
        X = pd.concat([X.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    return X.fillna(0.0)


# --------------------------------------------------------------------------- #
def descriptive(df: pd.DataFrame, d: pd.DataFrame) -> Dict:
    """Level 1: where/when/why-ish, straight from the person's own labels."""
    has_labels = "slip" in df.columns and df["slip"].notna().any()
    out: Dict = {"has_labels": bool(has_labels), "n_rows": int(len(df)),
                 "n_days": int(df["date"].nunique())}
    if not has_labels:
        return out

    s = df["slip"].fillna(0).astype(int).to_numpy()
    out["n_slips"] = int(s.sum())
    out["slip_rate"] = float(s.mean())

    # WHEN: by hour of day (keep counts so we can ignore sparsely-logged hours)
    by_hour = (pd.DataFrame({"hour": (df["clock_min"] // 60).astype(int), "slip": s})
               .groupby("hour")["slip"].agg(["mean", "count"]))
    out["by_hour"] = by_hour

    # WHERE: by activity / task_type
    out["by_task"] = (pd.DataFrame({"task": df["task_type"], "slip": s})
                      .groupby("task")["slip"].agg(["mean", "count"])
                      .sort_values("mean", ascending=False))

    # channel mix + time lost
    if "slip_channel" in df.columns:
        out["channels"] = df.loc[s == 1, "slip_channel"].value_counts(normalize=True)
    if "slip_minutes" in df.columns:
        out["time_lost_per_day"] = float(df.loc[s == 1, "slip_minutes"].sum()
                                         / max(1, out["n_days"]))

    # WHY (model-free): standardised difference in felt state, slip vs non-slip
    contrast = {}
    for c in _CONTRAST:
        if c in d.columns:
            a, b = d.loc[s == 1, c], d.loc[s == 0, c]
            sd = d[c].std()
            if sd > 1e-9 and len(a) and len(b):
                contrast[_PRETTY.get(c, c)] = float((a.mean() - b.mean()) / sd)
    out["state_contrast"] = (pd.Series(contrast).sort_values(ascending=False)
                             if contrast else pd.Series(dtype=float))
    return out


def personal_model(d: pd.DataFrame, df: pd.DataFrame) -> Optional[Dict]:
    """Level 2: fit a logistic risk model on the person's own data."""
    if "slip" not in df.columns:
        return None
    y = df["slip"].fillna(0).astype(int).to_numpy()
    if len(y) < MIN_ROWS_FOR_MODEL or y.sum() < MIN_SLIPS_FOR_MODEL or y.sum() == len(y):
        return dict(enough=False, n=int(len(y)), n_slips=int(y.sum()),
                    need_rows=MIN_ROWS_FOR_MODEL, need_slips=MIN_SLIPS_FOR_MODEL)

    X = _feature_matrix(d)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    # cross-validated discrimination (honest, out-of-sample)
    auc = np.nan
    try:
        k = min(5, int(y.sum()), int((y == 0).sum()))
        if k >= 2:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=C.GLOBAL_SEED)
            auc = float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")))
    except Exception:
        pass
    pipe.fit(X, y)
    return dict(enough=True, model=pipe, X=X, y=y, auc=auc,
                n=int(len(y)), n_slips=int(y.sum()))


SHRINKAGE_K = 25      # pseudo-slips: weight of the population prior


def _population_prior() -> Optional[pd.DataFrame]:
    """Population fingerprint stashed in the trained artifact, if available."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "model", "timeslip_model.joblib")
    if not os.path.exists(path):
        return None
    try:
        import joblib
        prod = joblib.load(path)
        return prod.get("population_fingerprint")
    except Exception:
        return None


def personal_fingerprint(model_res: Dict, d: pd.DataFrame) -> pd.DataFrame:
    """Counterfactual cause shares from the person's fitted model.

    With few logged slips a purely personal estimate is noisy, so we shrink it
    toward the population fingerprint (empirical-Bayes style): the personal
    weight is n_slips/(n_slips+K). With ~25 slips you are weighted 50/50; with
    hundreds, the prior barely matters. The report shows the blend weight.
    """
    pipe, X, y = model_res["model"], model_res["X"], model_res["y"]
    slip_rows = X[y == 1]
    if slip_rows.empty:
        return pd.DataFrame(columns=["cause", "share"])
    base = pipe.predict_proba(slip_rows)[:, 1]
    q10, q90 = X.quantile(0.10), X.quantile(0.90)
    rows = []
    for cause, ops in CAUSES_REAL.items():
        Xc = slip_rows.copy()
        touched = False
        for c in ops.get("low", []):
            if c in Xc.columns:
                Xc[c] = q10[c]; touched = True
        for c in ops.get("high", []):
            if c in Xc.columns:
                Xc[c] = q90[c]; touched = True
        for c in ops.get("zero", []):
            if c in Xc.columns:
                Xc[c] = 0.0; touched = True
        if not touched:
            continue
        drop = np.clip(base - pipe.predict_proba(Xc)[:, 1], 0, None).mean()
        rows.append({"cause": cause, "reduction": float(drop)})
    fp = pd.DataFrame(rows)
    tot = fp["reduction"].sum()
    fp["share"] = fp["reduction"] / tot if tot > 0 else 0.0

    # empirical-Bayes shrinkage toward the population prior
    prior = _population_prior()
    n_slips = int(y.sum())
    alpha = n_slips / (n_slips + SHRINKAGE_K)
    fp["share_personal"] = fp["share"]
    if prior is not None and not prior.empty:
        pr = prior.set_index("cause")["share"]
        fp["share"] = (alpha * fp["share"]
                       + (1 - alpha) * fp["cause"].map(pr).fillna(0.0))
        fp["share"] = fp["share"] / fp["share"].sum()
    fp = fp.sort_values("share", ascending=False).reset_index(drop=True)
    fp.attrs["alpha"] = alpha if (prior is not None and not prior.empty) else 1.0
    return fp


# --------------------------------------------------------------------------- #
def _fig_when(desc: Dict, path: str):
    if not desc.get("has_labels"):
        return
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(16, 6))
    bh = desc["by_hour"]
    a0.bar(bh.index, bh["mean"], color="#9b5de5")
    a0.set_xlabel("hour of day"); a0.set_ylabel("slip rate")
    a0.set_title("WHEN you slip")
    bt = desc["by_task"].head(10).iloc[::-1]
    a1.barh(bt.index, bt["mean"], color="#2e86ab")
    a1.set_xlabel("slip rate"); a1.set_title("WHERE you slip (by activity)")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_why(desc: Dict, path: str):
    sc = desc.get("state_contrast")
    if sc is None or sc.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e4572e" if v > 0 else "#2e86ab" for v in sc.values[::-1]]
    ax.barh(sc.index[::-1], sc.values[::-1], color=colors)
    ax.axvline(0, color="gray")
    ax.set_xlabel("how elevated when you slip  (SD vs. focused moments)")
    ax.set_title("WHY (model-free): states elevated at the moments you slip")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_fingerprint(fp: pd.DataFrame, auc, path: str):
    if fp.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    f = fp.iloc[::-1]
    ax.barh(f["cause"], f["share"], color="#f0a202")
    ax.set_xlabel("share of reducible slip risk")
    t = "Your personal slip fingerprint"
    if auc == auc:                                  # not NaN
        t += f"  (personal model CV-AUC {auc:.2f})"
    ax.set_title(t)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def write_report(desc: Dict, model_res, fp, fig_dir: str, rep_path: str):
    L = ["# Your slip report", ""]
    L.append(f"- Logged **{desc['n_rows']} intervals** across {desc['n_days']} days.")
    if not desc.get("has_labels"):
        L += ["",
              "> No `slip` labels found in your log, so this is limited to the data "
              "you entered. Mark which intervals you slipped (and the channel) to "
              "unlock the where/when/why analysis and your personal fingerprint."]
        open(rep_path, "w", encoding="utf-8").write("\n".join(L))
        return

    L.append(f"- **{desc['n_slips']} slips** "
             f"(~{desc['n_slips']/max(1,desc['n_days']):.1f}/day).")
    if "time_lost_per_day" in desc:
        L.append(f"- **~{desc['time_lost_per_day']:.0f} min/day** off-task in slips.")
    if "channels" in desc:
        L.append("- Channel mix: " +
                 ", ".join(f"{k} {v:.0%}" for k, v in desc["channels"].items()))

    L += ["", "## WHEN — your highest-risk hours"]
    bh = desc["by_hour"]
    bh = bh[bh["count"] >= 3].sort_values("mean", ascending=False).head(3)
    L.append("- " + ", ".join(f"{int(h):02d}:00 ({r['mean']:.0%})"
                              for h, r in bh.iterrows()))

    L += ["", "## WHERE — activities you slip in most",
          "| Activity | slip rate | logged |", "|---|---|---|"]
    for task, r in desc["by_task"].head(6).iterrows():
        L.append(f"| {task} | {r['mean']:.0%} | {int(r['count'])} |")

    sc = desc.get("state_contrast")
    if sc is not None and not sc.empty:
        L += ["", "## WHY — what's elevated when you slip (model-free)",
              "Standardised gap between slip moments and focused moments "
              "(positive = higher when you slip):"]
        for name, v in sc.head(5).items():
            L.append(f"- **{name}**: {v:+.2f} SD")

    L += ["", "## Your personal fingerprint (model-based)"]
    if model_res is None or not model_res.get("enough", False):
        n = model_res["n"] if model_res else desc["n_rows"]
        ns = model_res["n_slips"] if model_res else desc.get("n_slips", 0)
        L.append(f"- Not enough data yet for a stable personal model "
                 f"(you have {n} rows / {ns} slips; aim for at least "
                 f"{MIN_ROWS_FOR_MODEL} rows and {MIN_SLIPS_FOR_MODEL} slips). "
                 "Keep logging — the WHEN/WHERE/WHY above already works.")
    else:
        L.append(f"- Personal risk model trained on your data "
                 f"(cross-validated AUC "
                 f"{model_res['auc']:.2f} — 0.50 is chance).")
        alpha = fp.attrs.get("alpha", 1.0) if hasattr(fp, "attrs") else 1.0
        if alpha < 1.0:
            L.append(f"- Shares are shrunk toward the population fingerprint "
                     f"(personal weight {alpha:.0%} — grows as you log more slips).")
        L.append("")
        L.append("| Cause | Share of your reducible slip risk |")
        L.append("|---|---|")
        for _, r in fp.iterrows():
            L.append(f"| {r['cause']} | {r['share']:.0%} |")
        if not fp.empty:
            L += ["", f"**Your dominant trigger: {fp.iloc[0]['cause']}.**"]
        L += ["", "> Reading the two 'why' sections together: the WHY list shows "
              "what is *present* when you slip (often hard or aversive tasks — "
              "that's simply when focus is demanded). The fingerprint shows what "
              "is most *reducible* — the lever that would cut your slips the most. "
              "They answer different questions, so they can differ."]

    L += ["", "---",
          "*Figures: `me_when.png`, `me_why.png`, `me_fingerprint.png` in "
          f"`{os.path.relpath(fig_dir)}`.*"]
    open(rep_path, "w", encoding="utf-8").write("\n".join(L))


def analyze(path: str, out_root: str) -> Dict:
    fig_dir = os.path.join(out_root, "me")
    os.makedirs(fig_dir, exist_ok=True)
    df = load_log(path)
    # sort BEFORE prepare so features (built on the sorted frame) stay aligned
    # with labels taken from df
    df = df.sort_values(["date", "clock_min"]).reset_index(drop=True)
    d = prepare(df)
    desc = descriptive(df, d)
    mres = personal_model(d, df)
    fp = pd.DataFrame(columns=["cause", "share"])
    if mres and mres.get("enough"):
        fp = personal_fingerprint(mres, d)

    _fig_when(desc, os.path.join(fig_dir, "me_when.png"))
    _fig_why(desc, os.path.join(fig_dir, "me_why.png"))
    auc = mres["auc"] if (mres and mres.get("enough")) else float("nan")
    _fig_fingerprint(fp, auc, os.path.join(fig_dir, "me_fingerprint.png"))
    rep_path = os.path.join(fig_dir, "report_me.md")
    write_report(desc, mres, fp, fig_dir, rep_path)

    # plain-English layer: per-slip autopsies + the weekly Attention Account
    from . import narrative as N
    autopsy = N.per_slip_attribution(mres, df, d)
    account = N.attention_account(df, d, desc, mres, fp, autopsy)
    acc_path = os.path.join(fig_dir, "attention_account.md")
    with open(acc_path, "w", encoding="utf-8") as f:
        f.write(account)

    return dict(desc=desc, model=mres, fingerprint=fp, report=rep_path,
                account=acc_path, autopsy=autopsy, fig_dir=fig_dir)


# --------------------------------------------------------------------------- #
def make_example_log(pid: str = "P01", interval: int = 20,
                     minutes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build a realistic *filled* self-log from the simulator, to test/demo with.

    Downsamples one simulated person to one row every ``interval`` minutes and
    converts latent states to 1-5 Likert self-reports, mimicking what a real
    person would actually type into the template. Pass ``minutes`` to reuse an
    existing simulation instead of re-running it.
    """
    if minutes is None:
        from .simulate import simulate_all
        minutes, _, _ = simulate_all()
    pm = minutes[minutes["pid"] == pid].sort_values(["day", "clock_min"]).copy()

    def lk(x):                                   # 0-1 latent -> 1-5 Likert
        return int(np.clip(round(1 + 4 * float(x)), 1, 5))

    base = pd.Timestamp("2026-05-01")
    rows = []
    for day, g in pm.groupby("day"):
        g = g.reset_index(drop=True)
        for i in range(0, len(g), interval):
            w = g.iloc[i:i + interval]
            r = g.iloc[i]
            slipped = int(w["slip_onset"].max() == 1)
            ch = ""
            if slipped:
                cs = w.loc[w["slip_onset"] == 1, "slip_channel"]
                ch = cs.iloc[0] if len(cs) else ""
            rows.append({
                "date": (base + pd.Timedelta(days=int(day))).strftime("%Y-%m-%d"),
                "clock_min": int(r["clock_min"]),
                "activity": r["activity"], "task_type": r["task_type"],
                "difficulty": lk(r["difficulty"]), "intrinsic": lk(r["intrinsic"]),
                "aversive": lk(r["aversive"]), "boredom": lk(r["boredom_obs"]),
                "stress": lk(r["stress_obs"]), "energy": lk(r["energy_obs"]),
                "hunger": lk(r["hunger_obs"]), "alertness": lk(r["alertness_obs"]),
                "location": r["location"], "social": r["social"],
                "phone_in_reach": int(r["phone_in_reach"]),
                "notif": int(w["notif"].sum()), "deadline": lk(r["deadline"]),
                "open_tasks": int(r["open_tasks"]),
                "slip": slipped, "slip_channel": ch,
                "slip_minutes": int(w["in_slip"].sum()),
            })
    return pd.DataFrame(rows)
