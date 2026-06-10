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
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "outputs", "model", "timeslip_model.joblib")


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
    from timeslip.api import TimeSlip
    ts = TimeSlip(MODEL_PATH)
    r = ts.risk(boredom=a.boredom, stress=a.stress, energy=a.energy,
                hunger=a.hunger, alertness=a.alertness, difficulty=a.difficulty,
                intrinsic=a.intrinsic, aversive=a.aversive, deadline=a.deadline,
                task=a.task, phone=a.phone, notifs=a.notifs, tot=a.tot,
                hour=a.hour, open_tasks=a.open_tasks, slips_today=a.slips_today,
                since_meal=a.since_meal)

    interval = ""
    if r["interval"]:
        lo, hi = r["interval"]
        interval = f"  (Venn-Abers interval {lo:.0%}-{hi:.0%})"
    print(f"\nRisk of an attention slip in the next {r['horizon_min']} min: "
          f"**{r['risk']:.0%}**{interval}")

    print("\nWhat would help most right now:")
    for lev in r["levers"]:
        bar = "#" * max(0, int(round(lev["reduction"] * 40)))
        print(f"  {lev['name']:<42} -> {lev['new_risk']:>4.0%}  "
              f"(-{lev['reduction']:.0%}) {bar}")
    print("\n(Calibrated on the benchmark cohort; personalise by logging your "
          "own data with analyze_me.py / Obsidian.)")


if __name__ == "__main__":
    main()
