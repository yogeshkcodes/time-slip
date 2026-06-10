"""
Time Slip - live "what-if" engine.

Type in how you feel RIGHT NOW and what you're doing; get your calibrated
slip risk for the next 10 minutes, an honest uncertainty interval, and the
single best lever to pull - computed from the trained production model.

    python whatif.py --boredom 4 --stress 3 --energy 2 --task deep_work \
                     --aversive 4 --phone 1 --notifs 3 --tot 35

    python whatif.py --policy      # simulate week-long interventions instead
                                   # (phone away / DND / more sleep) on the SCM

All "feel" arguments are 1-5 scales (like the log template). Defaults are
neutral, so you can give only what you know. Requires the trained model
(outputs/model/timeslip_model.joblib) - run `python run_all.py` once first.
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "outputs", "model", "timeslip_model.joblib")

FOCUS_TASKS = {"deep_work", "study", "admin", "meeting", "shift_work"}


def lk(x: float) -> float:
    """1-5 Likert -> 0-1 model scale."""
    return (float(x) - 1.0) / 4.0


def build_row(a, columns) -> pd.DataFrame:
    focus = 1 if a.task in FOCUS_TASKS else 0
    tot = float(a.tot)
    vig = min(tot / 60.0, 2.0) if focus else min(0.5, tot / 120.0)
    hour = a.hour
    row = {
        "boredom_obs": lk(a.boredom), "stress_obs": lk(a.stress),
        "energy_obs": lk(a.energy), "hunger_obs": lk(a.hunger),
        "alertness_obs": lk(a.alertness),
        "boredom_obs_roll20": lk(a.boredom), "stress_obs_roll20": lk(a.stress),
        "difficulty": lk(a.difficulty), "intrinsic": lk(a.intrinsic),
        "aversive": lk(a.aversive), "deadline": lk(a.deadline),
        "open_tasks": a.open_tasks, "focus": focus, "is_meal": 0, "is_break": 0,
        "phone_in_reach": a.phone, "minutes_awake": max(0.0, (hour - 7.0) * 60),
        "time_on_task": tot, "vigilance": vig, "notif": min(a.notifs, 2),
        "notif_15": a.notifs, "slips_today": a.slips_today, "since_slip": 600.0,
        "since_meal": a.since_meal, "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24), "sex_M": 0.5,
        "trait_self_control": 0.5, "neuroticism": 0.5, "conscientiousness": 0.5,
        "habit_strength": 0.5, "chronotype": 0.5, "caffeine_use": 0.5,
        f"task_type_{a.task}": 1.0, "location_office": 1.0, "social_alone": 1.0,
    }
    X = pd.DataFrame([row]).reindex(columns=columns, fill_value=0.0)
    return X


LEVERS = {
    "Phone away + notifications off": {"phone_in_reach": 0.0, "notif": 0.0,
                                       "notif_15": 0.0},
    "Take a break now (reset time-on-task)": {"time_on_task": 0.0, "vigilance": 0.0},
    "Eat something": {"hunger_obs": 0.25, "since_meal": 0.0},
    "Switch to / reframe as an engaging task": {"intrinsic": 0.75,
                                                "aversive": 0.25},
    "Down-regulate stress (breathing, walk)": {"stress_obs": 0.25,
                                               "stress_obs_roll20": 0.25},
    "Boost alertness (rest, light, caffeine)": {"alertness_obs": 0.75,
                                                "energy_obs": 0.75},
}


def main():
    ap = argparse.ArgumentParser(description="Live slip-risk what-if")
    for name, default, hint in [
            ("boredom", 3, "1-5"), ("stress", 3, "1-5"), ("energy", 3, "1-5"),
            ("hunger", 2, "1-5"), ("alertness", 3, "1-5"), ("difficulty", 3, "1-5"),
            ("intrinsic", 3, "1-5"), ("aversive", 3, "1-5"), ("deadline", 3, "1-5")]:
        ap.add_argument(f"--{name}", type=float, default=default,
                        help=f"how it feels right now ({hint})")
    ap.add_argument("--task", default="deep_work",
                    help="deep_work/study/admin/meeting/leisure/chores/...")
    ap.add_argument("--phone", type=int, default=1, help="phone in reach? 0/1")
    ap.add_argument("--notifs", type=float, default=2,
                    help="notifications in the last 15 min")
    ap.add_argument("--tot", type=float, default=20,
                    help="minutes continuously on this task")
    ap.add_argument("--hour", type=float,
                    default=float(pd.Timestamp.now().hour)
                    + pd.Timestamp.now().minute / 60.0)
    ap.add_argument("--open-tasks", dest="open_tasks", type=float, default=2)
    ap.add_argument("--slips-today", dest="slips_today", type=float, default=3)
    ap.add_argument("--since-meal", dest="since_meal", type=float, default=120)
    ap.add_argument("--policy", action="store_true",
                    help="simulate week-long interventions on the causal model")
    a = ap.parse_args()

    if a.policy:
        from timeslip.interventions import run_policies
        print("Simulating intervention policies (paired seeds, 8 archetypes) ...")
        print(run_policies().to_string())
        print("\nModel-implied effects under the simulator's assumptions - "
              "a hypothesis generator for a real experiment.")
        return

    if not os.path.exists(MODEL_PATH):
        sys.exit("No trained model found. Run `python run_all.py` once first.")
    import joblib
    prod = joblib.load(MODEL_PATH)

    X = build_row(a, prod["columns"])
    raw = float(prod["model"].predict_proba(X)[:, 1][0])
    risk = float(prod["iso"].transform([raw])[0])

    interval = ""
    if "cal_scores" in prod:
        from timeslip.evaluate import venn_abers_interval
        lo, hi = venn_abers_interval(prod["cal_scores"], prod["cal_y"], raw)
        interval = f"  (Venn-Abers interval {lo:.0%}-{hi:.0%})"

    print(f"\nRisk of an attention slip in the next {prod['horizon']} min: "
          f"**{risk:.0%}**{interval}")

    # levers
    results = []
    for name, mods in LEVERS.items():
        Xc = X.copy()
        for col, val in mods.items():
            if col in Xc.columns:
                Xc[col] = val
        raw_c = float(prod["model"].predict_proba(Xc)[:, 1][0])
        risk_c = float(prod["iso"].transform([raw_c])[0])
        results.append((name, risk_c, risk - risk_c))
    results.sort(key=lambda r: r[2], reverse=True)

    print("\nWhat would help most right now:")
    for name, risk_c, drop in results:
        bar = "#" * max(0, int(round(drop * 40)))
        print(f"  {name:<42} -> {risk_c:>4.0%}  (-{max(0,drop):.0%}) {bar}")
    print("\n(Calibrated on the simulation cohort; personalise by logging your "
          "own data with analyze_me.py / Obsidian.)")


if __name__ == "__main__":
    main()
