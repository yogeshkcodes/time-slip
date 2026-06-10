"""
Narrative reporting for Time Slip - reports written for humans, not reviewers.

Two deliverables:

  PER-SLIP AUTOPSY   Every slip gets one readable line with its own cause
                     breakdown, computed by running the leave-one-cause-out
                     counterfactual on *that specific moment* (not the average):
                     "Tue 14:20 - during deep_work (47 min in), slipped to
                      phone for 12 min. Drivers: task aversiveness 38%,
                      phone pull 31%, boredom 14%."

  ATTENTION ACCOUNT  A weekly statement that reads like a bank statement:
                     balance (time lost), where it went, the week's worst
                     slips as one-line stories, your cause split in plain
                     words, patterns worth knowing, and ONE prescribed
                     experiment for next week with a measurement plan.

No charts are needed to understand either. The numbers come from the person's
own log; the cause split uses their personal model (with the shrinkage weight
shown when their data is still small).
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from .realdata import CAUSES_REAL

DAYNAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# plain-English one-liners for each cause, used in the statement
CAUSE_MEANING = {
    "Phone pull":               "the phone + notifications themselves, not your willpower",
    "Task aversiveness":        "dread of the task - you flee what you avoid starting",
    "Low intrinsic motivation": "the task doesn't matter to you, so attention leaks",
    "Stress":                   "arousal is up; focus narrows and then snaps",
    "Boredom":                  "under-stimulation - the mind goes shopping",
    "Fatigue":                  "low alertness (sleep, circadian dip)",
    "Hunger":                   "a body signal outbidding the task",
    "Time-on-task (vigilance)": "focus fatigue from long unbroken stretches",
}

# dominant cause -> (experiment to run next week, how it will be measured)
EXPERIMENTS = {
    "Phone pull": (
        "Phone in another room (not pocket, not desk) during your two longest "
        "focus blocks each day; notifications batched to 3 fixed times.",
        "phone-channel slips/day and minutes lost vs this week"),
    "Task aversiveness": (
        "Every morning, break your most-avoided task into a 10-minute first "
        "step and do it in your best hour - before email.",
        "slips during that task and time-to-start vs this week"),
    "Low intrinsic motivation": (
        "Attach each low-interest block to a stake you care about (tell "
        "someone, time-box it, or trade it for a reward block).",
        "slips during low-interest blocks vs this week"),
    "Stress": (
        "90 seconds of slow breathing before each focus block; park open "
        "loops in a visible list before starting.",
        "slips on high-deadline days vs this week"),
    "Boredom": (
        "Raise the challenge: tighten deadlines on monotonous work, or batch "
        "it into short timed sprints with a visible score.",
        "slips during monotonous blocks vs this week"),
    "Fatigue": (
        "Protect a fixed wake time and move your hardest block into your "
        "alertness peak; 10-minute walk at the post-lunch dip.",
        "afternoon slips/day vs this week"),
    "Hunger": (
        "Fixed meal/snack times; never enter a 90-minute focus block more "
        "than 3 hours after eating.",
        "late-morning and pre-dinner slips vs this week"),
    "Time-on-task (vigilance)": (
        "A deliberate 5-minute break every N minutes, where N is just under "
        "your median focus stretch (see Patterns below).",
        "slips after the 30-minute mark of any block vs this week"),
}


def _hhmm(clock_min) -> str:
    m = int(clock_min)
    return f"{m // 60:02d}:{m % 60:02d}"


def _dayname(date_str) -> str:
    try:
        return DAYNAMES[pd.Timestamp(date_str).dayofweek]
    except Exception:
        return str(date_str)


# --------------------------------------------------------------------------- #
# per-slip attribution
# --------------------------------------------------------------------------- #
def per_slip_attribution(model_res: Dict, df: pd.DataFrame,
                         d: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One row per slip with its own counterfactual cause shares.

    Same leave-one-cause-out machinery as the fingerprint, but the drops are
    kept per-slip instead of averaged, so each incident gets its own story.
    Returns None when there is no usable personal model.
    """
    if not (model_res and model_res.get("enough")):
        return None
    pipe, X, y = model_res["model"], model_res["X"], model_res["y"]
    idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return None
    Xs = X.iloc[idx]
    base = pipe.predict_proba(Xs)[:, 1]
    q10, q90 = X.quantile(0.10), X.quantile(0.90)

    drops = {}
    for cause, ops in CAUSES_REAL.items():
        Xc = Xs.copy(); touched = False
        for c in ops.get("low", []):
            if c in Xc.columns: Xc[c] = q10[c]; touched = True
        for c in ops.get("high", []):
            if c in Xc.columns: Xc[c] = q90[c]; touched = True
        for c in ops.get("zero", []):
            if c in Xc.columns: Xc[c] = 0.0; touched = True
        if not touched:
            continue
        drops[cause] = np.clip(base - pipe.predict_proba(Xc)[:, 1], 0, None)

    D = pd.DataFrame(drops, index=idx)
    tot = D.sum(axis=1).replace(0, np.nan)
    shares = D.div(tot, axis=0).fillna(0.0)

    meta = df.iloc[idx].reset_index(drop=True)
    out = shares.reset_index(drop=True)
    out["date"] = meta.get("date", "").values
    out["clock_min"] = meta.get("clock_min", 0).values
    out["activity"] = meta.get("activity", meta.get("task_type", "")).values
    out["channel"] = meta.get("slip_channel", "").fillna("").values
    out["minutes"] = pd.to_numeric(meta.get("slip_minutes", 0),
                                   errors="coerce").fillna(0).values
    out["time_on_task"] = (d.iloc[idx]["time_on_task"].values
                           if "time_on_task" in d.columns else 0)
    return out


def slip_story(row: pd.Series, causes: List[str]) -> str:
    """Render one slip as a single readable line."""
    shares = [(c, row[c]) for c in causes if c in row and row[c] >= 0.10]
    shares.sort(key=lambda t: t[1], reverse=True)
    drivers = ", ".join(f"{c.lower()} {s:.0%}" for c, s in shares[:3]) \
        or "no single dominant driver"
    ch = f" to {row['channel']}" if row.get("channel") else ""
    dur = f" for {row['minutes']:.0f} min" if row.get("minutes", 0) > 0 else ""
    tot = (f", {row['time_on_task']:.0f} min into the task"
           if row.get("time_on_task", 0) >= 10 else "")
    return (f"**{_dayname(row['date'])} {_hhmm(row['clock_min'])}** - during "
            f"{row['activity']}{tot}: slipped{ch}{dur}. Drivers: {drivers}.")


# --------------------------------------------------------------------------- #
# the Attention Account statement (self-log flavour)
# --------------------------------------------------------------------------- #
def _week_over_week(df: pd.DataFrame) -> Optional[str]:
    """Compare the most recent 7 logged days against the 7 before them."""
    if "date" not in df.columns or "slip" not in df.columns:
        return None
    days = sorted(df["date"].unique())
    if len(days) < 10:
        return None
    recent, prior = set(days[-7:]), set(days[-14:-7])
    s = df.groupby("date")["slip"].sum()
    m = (df.groupby("date")["slip_minutes"].sum()
         if "slip_minutes" in df.columns else None)
    r_s = s[s.index.isin(recent)].mean(); p_s = s[s.index.isin(prior)].mean()
    line = None
    if p_s > 0:
        d = 100 * (r_s / p_s - 1)
        arrow = "down" if d < 0 else "up"
        line = f"Slips/day are **{arrow} {abs(d):.0f}%** vs the previous week"
        if m is not None:
            r_m = m[m.index.isin(recent)].mean(); p_m = m[m.index.isin(prior)].mean()
            if p_m > 0:
                dm = 100 * (r_m / p_m - 1)
                line += (f"; time lost is "
                         f"**{'down' if dm < 0 else 'up'} {abs(dm):.0f}%**.")
            else:
                line += "."
    return line


def attention_account(df: pd.DataFrame, d: pd.DataFrame, desc: Dict,
                      model_res, fp: pd.DataFrame,
                      autopsy: Optional[pd.DataFrame]) -> str:
    """Build the plain-English weekly statement (markdown)."""
    days = desc.get("n_days", df["date"].nunique() if "date" in df else 1)
    n_slips = int(desc.get("n_slips", 0))
    lost = desc.get("time_lost_per_day")

    dates = sorted(df["date"].astype(str).unique()) if "date" in df else []
    span = f"{dates[0]} to {dates[-1]}" if dates else "your log"

    L = [f"# Attention Account - {span}", ""]

    # ---- balance ----
    L.append("## Balance")
    bal = f"- **{n_slips} slips** over {days} day(s) (~{n_slips/max(1,days):.1f}/day)."
    L.append(bal)
    if lost is not None:
        L.append(f"- **~{lost:.0f} min/day lost** to slips - about "
                 f"**{lost*7/60:.1f} hours/week**.")
    wow = _week_over_week(df)
    if wow:
        L.append(f"- {wow}")

    # ---- where it went ----
    if "channels" in desc and len(desc["channels"]):
        L += ["", "## Where it went"]
        L.append("- " + ", ".join(f"{k} {v:.0%}" for k, v in desc["channels"].items()))
    bt = desc.get("by_task")
    if bt is not None and len(bt):
        worst = bt[bt["count"] >= 3].head(3)
        if len(worst):
            L.append("- Most slippery activities: " +
                     ", ".join(f"{t} ({r['mean']:.0%})" for t, r in worst.iterrows()))

    # ---- worst slips as stories ----
    if autopsy is not None and len(autopsy):
        causes = [c for c in CAUSES_REAL if c in autopsy.columns]
        worst = autopsy.sort_values("minutes", ascending=False).head(5)
        L += ["", "## The week's worst slips"]
        for _, r in worst.iterrows():
            L.append("- " + slip_story(r, causes))
        L.append("- *(Drivers explain why attention broke at that moment; the "
                 "channel is just where it went afterwards.)*")
    elif n_slips:
        L += ["", "## The week's worst slips",
              "- Log a few more days (30+ rows, 6+ slips) to unlock per-slip "
              "cause breakdowns."]

    # ---- cause split ----
    L += ["", "## Why you slip (your cause split)"]
    if fp is not None and len(fp):
        for _, r in fp.iterrows():
            if r["share"] < 0.03:
                continue
            meaning = CAUSE_MEANING.get(r["cause"], "")
            L.append(f"- **{r['cause']}: {r['share']:.0%}** - {meaning}")
        alpha = fp.attrs.get("alpha", 1.0) if hasattr(fp, "attrs") else 1.0
        if alpha < 1.0:
            L.append(f"- *(personal weight {alpha:.0%}; the rest comes from the "
                     "population prior until you log more slips)*")
    else:
        L.append("- Not enough logged slips yet for a stable split - the "
                 "balance and stories above are already real.")

    # ---- patterns ----
    L += ["", "## Patterns worth knowing"]
    bh = desc.get("by_hour")
    if bh is not None and len(bh):
        solid = bh[bh["count"] >= 3]
        if len(solid):
            top = solid["mean"].idxmax()
            L.append(f"- Your riskiest logged hour is **{int(top):02d}:00** "
                     f"({solid.loc[top, 'mean']:.0%} of intervals slip).")
    if autopsy is not None and len(autopsy) >= 5:
        med_tot = autopsy["time_on_task"].median()
        if med_tot >= 5:
            L.append(f"- Slips typically arrive **~{med_tot:.0f} minutes into a "
                     "task** - plan a deliberate break just before that mark.")
    sc = desc.get("state_contrast")
    if sc is not None and len(sc):
        top_state = sc.index[0]
        L.append(f"- The state most elevated at your slip moments: "
                 f"**{top_state}** (+{sc.iloc[0]:.1f} SD vs focused moments).")

    # ---- experiment ----
    if fp is not None and len(fp):
        dom = fp.iloc[0]["cause"]
        exp, measure = EXPERIMENTS.get(dom, (None, None))
        if exp:
            L += ["", "## Next week's experiment (one change only)",
                  f"- Your dominant cause is **{dom}**, so: {exp}",
                  f"- **How we'll know it worked:** {measure}.",
                  "- Change nothing else, keep logging, and next week's "
                  "statement will show the before/after."]

    L += ["", "---",
          "*Plain-English by design. Every number comes from your own log; "
          "the cause split is computed by your personal model.*"]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# tracker flavour (behavioural data, no felt states)
# --------------------------------------------------------------------------- #
def attention_account_tracker(summary: Dict, slips: pd.DataFrame,
                              spells: pd.DataFrame) -> str:
    L = ["# Attention Account (behavioural) - from your tracked computer use", ""]
    L.append("## Balance")
    L.append(f"- **{summary['hours_tracked']:.1f} hours tracked** over "
             f"{summary['n_days']} day(s).")
    L.append(f"- **{summary['n_slips']} behavioural slips** "
             f"(~{summary['slips_per_day']:.1f}/day), including "
             f"**{summary['rabbit_holes']} rabbit holes**; "
             f"~**{summary['time_lost_per_day']:.0f} min/day lost**.")
    if summary.get("median_focus_min") == summary.get("median_focus_min"):
        L.append(f"- Median focus stretch: **{summary['median_focus_min']:.0f} min** "
                 f"(longest {summary['longest_focus_min']:.0f}); "
                 f"{summary['switches_per_hour']:.0f} context switches/hour.")

    if summary.get("top_distractors"):
        L += ["", "## Where it went"]
        for app, mins in list(summary["top_distractors"].items())[:5]:
            L.append(f"- {app}: ~{mins:.0f} min/day")

    if len(slips):
        L += ["", "## The worst rabbit holes"]
        worst = slips.sort_values("dwell_min", ascending=False).head(5)
        for _, r in worst.iterrows():
            L.append(f"- **{_dayname(r['date'])} {str(r['time'])[11:16]}** - after "
                     f"{r['after_work_min']:.0f} min of work, {r['app']} for "
                     f"**{r['dwell_min']:.0f} min**"
                     + (" (rabbit hole)" if r.get("rabbit_hole") else "") + ".")

        L += ["", "## Patterns worth knowing"]
        hr = slips.groupby(slips["hour"].astype(int)).size()
        if len(hr):
            L.append(f"- Most slips start around **{int(hr.idxmax()):02d}:00**.")
        med_aw = slips["after_work_min"].median()
        L.append(f"- A slip typically follows **~{med_aw:.0f} min** of continuous "
                 "work - schedule a deliberate break just before that point.")
        L += ["", "## Next week's experiment (one change only)",
              "- Your slips are detected behaviourally (work -> distractor), so "
              "the cleanest lever is access: block or log out of your top "
              "distractor during your two longest work blocks.",
              "- **How we'll know it worked:** rabbit holes/day and min/day "
              "lost vs this week."]

    L += ["", "---",
          "*Behavioural data only - no self-report involved. Pair with "
          "`analyze_me.py` to add the felt-state causes (boredom, stress, "
          "fatigue) to this picture.*"]
    return "\n".join(L)
