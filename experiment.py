"""
Time Slip - run an N-of-1 experiment on yourself.

The Attention Account prescribes ONE change. Register it, keep logging, and the
engine measures the real effect on YOUR data (weekday-controlled permutation
test + bootstrap CI + minimum-detectable-effect honesty).

    python experiment.py accept my_log.csv        # auto-register the prescribed change
    python experiment.py start  --cause "Phone pull" --metric phone_min_per_day \
                                --change "phone in another room during focus blocks"
    python experiment.py status my_log.csv        # measure progress / result
    python experiment.py list                     # show registered experiments

Use --vault "C:/path/to/Vault" to keep the registry inside an Obsidian vault
(otherwise it lives in outputs/experiments.json).
"""

from __future__ import annotations
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def registry_path(vault: str | None) -> str:
    if vault:
        return os.path.join(vault, "TimeSlip", "experiments.json")
    return os.path.join(ROOT, "outputs", "experiments.json")


def main():
    ap = argparse.ArgumentParser(description="N-of-1 experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("accept", help="register the change the report prescribed")
    pa.add_argument("log")
    pa.add_argument("--vault", default=None)

    ps = sub.add_parser("start", help="register a custom experiment")
    ps.add_argument("--cause", required=True)
    ps.add_argument("--metric", default="slips_per_day",
                    choices=["slips_per_day", "min_lost_per_day", "phone_min_per_day"])
    ps.add_argument("--change", required=True)
    ps.add_argument("--measure", default="")
    ps.add_argument("--start", default=None, help="ISO date intervention began")
    ps.add_argument("--vault", default=None)

    pt = sub.add_parser("status", help="measure registered experiments on a log")
    pt.add_argument("log")
    pt.add_argument("--vault", default=None)

    pl = sub.add_parser("list", help="list registered experiments")
    pl.add_argument("--vault", default=None)

    a = ap.parse_args()
    from timeslip import experiments as E
    reg = registry_path(getattr(a, "vault", None))

    if a.cmd == "list":
        exps = E.load_registry(reg)
        if not exps:
            print(f"No experiments registered ({reg}).")
            return
        for e in exps:
            print(f"- [{e.id}] {e.cause}: \"{e.change}\" "
                  f"(metric {e.metric}, started {e.start_date})")
        return

    if a.cmd == "start":
        e = E.start_experiment(reg, a.cause, a.change, a.metric, a.measure, a.start)
        print(f"Registered experiment [{e.id}] starting {e.start_date}.")
        print("Keep logging, then: python experiment.py status <log>")
        return

    if a.cmd == "accept":
        from timeslip import realdata as R
        df = R.load_log(a.log).sort_values(["date", "clock_min"]).reset_index(drop=True)
        d = R.prepare(df)
        mres = R.personal_model(d, df)
        if not (mres and mres.get("enough")):
            sys.exit("Not enough logged data yet to identify your dominant cause. "
                     "Log more, or use `start` to set one manually.")
        fp = R.personal_fingerprint(mres, d)
        from timeslip import narrative as N
        dom = fp.iloc[0]["cause"]
        change, measure = N.EXPERIMENTS.get(dom, ("the prescribed change", ""))
        metric = N.CAUSE_METRIC.get(dom, "slips_per_day")
        e = E.start_experiment(reg, dom, change, metric, measure)
        print(f"Your dominant cause is {dom}. Registered experiment [{e.id}] "
              f"starting today ({e.start_date}):")
        print(f"  {change}")
        print("From today, apply that change and keep logging. "
              "Run `python experiment.py status <log>` to see the measured effect.")
        return

    if a.cmd == "status":
        from timeslip import realdata as R
        df = R.load_log(a.log).sort_values(["date", "clock_min"]).reset_index(drop=True)
        results = E.evaluate_active(df, reg)
        if not results:
            print(f"No experiments registered ({reg}). Use `accept` or `start`.")
            return
        for res in results:
            print("- " + E.narrative_line(res))


if __name__ == "__main__":
    main()
