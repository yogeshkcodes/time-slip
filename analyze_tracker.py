"""
Time Slip - analyse your REAL tracked computer behaviour.

    python track_me.py                  # first, collect data (run it while you work)
    python analyze_tracker.py           # then analyse outputs/tracker/
    python analyze_tracker.py my.csv    # or a specific tracker CSV
    python analyze_tracker.py --demo    # no data yet? analyse a synthetic demo file

Writes a behavioural report + figures to outputs/tracker_report/:
real focus-survival, a real vigilance curve, slip timing, and rabbit-hole stats
- computed from your actual app usage, no self-report required.
"""

from __future__ import annotations
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from timeslip import tracker as T

sns.set_theme(style="whitegrid", context="talk")
ROOT = os.path.dirname(os.path.abspath(__file__))
TRACK_DIR = os.path.join(ROOT, "outputs", "tracker")
REPORT_DIR = os.path.join(ROOT, "outputs", "tracker_report")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_figures(df, segs, slips, spells):
    # 1. time by category per day
    days = max(1, df["date"].nunique())
    by_cat = (df.groupby("category")["dur_s"].sum() / 60.0 / days).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"work": "#2e86ab", "distractor": "#e4572e", "communication": "#f0a202",
              "browse": "#9b5de5", "other": "#999999", "away": "#cccccc"}
    ax.barh(by_cat.index, by_cat.values,
            color=[colors.get(c, "#999") for c in by_cat.index])
    ax.set_xlabel("minutes per day")
    ax.set_title("Where your screen time actually goes")
    _save(fig, "real_time_by_category.png")

    # 2. slips by hour
    if len(slips):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(slips["hour"], bins=range(6, 24), color="#e4572e", rwidth=0.9)
        ax.set_xlabel("hour of day"); ax.set_ylabel("behavioural slips")
        ax.set_title("When you actually slip (work -> distractor)")
        _save(fig, "real_slips_by_hour.png")

    # 3. real focus survival
    if len(spells) >= 10:
        import numpy as np
        d = np.sort(spells["dur_min"].to_numpy())
        surv = 1.0 - np.arange(1, len(d) + 1) / len(d)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.step(d, surv, where="post", color="#2e86ab", lw=2.5)
        med = float(spells["dur_min"].median())
        ax.axvline(med, ls="--", color="gray", label=f"median {med:.0f} min")
        ax.set_xlabel("minutes of continuous focused work")
        ax.set_ylabel("P(still focused)")
        ax.set_xlim(0, min(90, d.max()))
        ax.set_title("Your REAL attention-survival curve")
        ax.legend()
        _save(fig, "real_focus_survival.png")

    # 4. real vigilance curve
    vig = T.vigilance_real(spells)
    if len(vig) >= 3:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(vig["dur_bin"], vig["p_distract"], "o-", color="#e4572e")
        ax.set_xlabel("focus-spell length (min)")
        ax.set_ylabel("P(spell ends in a distractor)")
        ax.set_title("Your REAL vigilance pattern")
        _save(fig, "real_vigilance.png")


def write_report(summary, slips, path):
    L = ["# Your real behaviour report (tracker data)",
         f"*{summary['hours_tracked']:.1f} hours tracked over "
         f"{summary['n_days']} day(s). All data local; labels are behavioural "
         "(work -> distractor transitions), no self-report needed.*", ""]
    mpd = summary["min_per_day"]
    L.append("## Where the time goes (min/day)")
    for cat in sorted(mpd, key=mpd.get, reverse=True):
        L.append(f"- **{cat}**: {mpd[cat]:.0f} min")
    L += ["", "## Slips (work -> distractor)",
          f"- **{summary['n_slips']} slips** (~{summary['slips_per_day']:.1f}/day), "
          f"of which **{summary['rabbit_holes']} rabbit holes** (>= "
          f"{T.RABBIT_HOLE_MIN} min).",
          f"- **~{summary['time_lost_per_day']:.0f} min/day** lost to slip dwells.",
          f"- Median focus spell: **{summary['median_focus_min']:.0f} min** "
          f"(longest {summary['longest_focus_min']:.0f} min); "
          f"{summary['switches_per_hour']:.0f} app-context switches/hour."]
    if summary["top_distractors"]:
        L += ["", "## Top distractors (min/day)"]
        for app, mins in summary["top_distractors"].items():
            L.append(f"- {app}: {mins:.0f} min")
    if len(slips):
        worst = slips.sort_values("dwell_min", ascending=False).head(5)
        L += ["", "## Worst rabbit holes"]
        for _, r in worst.iterrows():
            L.append(f"- {r['date']} {str(r['time'])[11:16]} - **{r['dwell_min']:.0f} min** "
                     f"on {r['app']} after {r['after_work_min']:.0f} min of work")
    L += ["", "---",
          "*Figures in this folder: real_time_by_category, real_slips_by_hour, "
          "real_focus_survival, real_vigilance. Pair with `analyze_me.py` "
          "(self-logged states) to add the WHY layer to this WHAT layer.*"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--demo" in sys.argv:
        # kept OUT of outputs/tracker/ so it never mixes with your real data
        src = os.path.join(ROOT, "outputs", "tracker_demo", "track_demo.csv")
        T.make_demo_tracker(src)
        print(f"(demo mode: synthesised {src})")
    elif args:
        src = args[0]
    else:
        src = TRACK_DIR

    df = T.load_tracker(src)
    segs = T.build_segments(df)
    slips = T.detect_slips(segs)
    spells = T.focus_spells(segs)
    summary = T.summarize(df, segs, slips)

    make_figures(df, segs, slips, spells)
    rep = os.path.join(REPORT_DIR, "report_tracker.md")
    write_report(summary, slips, rep)

    # plain-English statement (no charts needed)
    from timeslip import narrative as N
    acc = os.path.join(REPORT_DIR, "attention_account.md")
    with open(acc, "w", encoding="utf-8") as f:
        f.write(N.attention_account_tracker(summary, slips, spells))

    print(f"Tracked {summary['hours_tracked']:.1f} h over {summary['n_days']} day(s): "
          f"{summary['n_slips']} behavioural slips "
          f"(~{summary['time_lost_per_day']:.0f} min/day lost), "
          f"median focus {summary['median_focus_min']:.0f} min.")
    print(f"Report    -> {rep}")
    print(f"Statement -> {acc}   (plain English, start here)")


if __name__ == "__main__":
    main()
