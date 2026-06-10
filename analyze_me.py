"""
Time Slip - analyse YOUR own routine.

    python analyze_me.py path/to/my_log.csv     # analyse your filled-in log
    python analyze_me.py                         # no file? build + analyse an
                                                 # example log so you can see the
                                                 # exact output format

Get a blank template to fill in with:  python -m timeslip.schema
Outputs land in ./outputs/me/  (start with report_me.md).
"""

from __future__ import annotations
import os
import sys
import pandas as pd

from timeslip import realdata as R

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)
    else:
        # no file given -> generate a realistic example so the format is obvious
        print("No log given - generating a realistic EXAMPLE log to analyse "
              "(this is what your own data would look like) ...")
        ex = R.make_example_log()
        path = os.path.join(OUT, "data", "example_filled_log.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ex.to_csv(path, index=False)
        print(f"  wrote example log: {path}  ({len(ex)} rows)")

    print(f"\nAnalysing {path} ...")
    res = R.analyze(path, OUT)

    d = res["desc"]
    print(f"\n  logged {d['n_rows']} intervals over {d['n_days']} days")
    if d.get("has_labels"):
        print(f"  {d['n_slips']} slips logged")
        if not res["fingerprint"].empty:
            top = res["fingerprint"].iloc[0]
            print(f"  dominant trigger: {top['cause']} ({top['share']:.0%})")
    print(f"\nDone.")
    print(f"  Statement -> {res.get('account', '')}   (plain English, start here)")
    print(f"  Report    -> {res['report']}")
    print(f"  Figures   -> {res['fig_dir']}  (me_when / me_why / me_fingerprint)")


if __name__ == "__main__":
    main()
