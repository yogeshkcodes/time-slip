"""
Time Slip <-> Obsidian.

    python obsidian_sync.py "C:/path/to/Vault" --init   # create the TimeSlip folder
    python obsidian_sync.py "C:/path/to/Vault"          # parse logs -> write report

Log your routine as a markdown table in <Vault>/TimeSlip/logs/*.md, then run the
second command. It writes "<Vault>/TimeSlip/Time Slip Report.md" (with charts).
See <Vault>/TimeSlip/README.md after --init for the column format.
"""

from __future__ import annotations
import os
import sys
from timeslip import obsidian as O


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__)
        sys.exit(1)
    vault = args[0]
    if not os.path.isdir(vault):
        print(f"Vault folder not found: {vault}")
        sys.exit(1)

    if "--init" in flags:
        base = O.make_vault(vault)
        print(f"Created {base}")
        print("  - README.md, Daily Log Template.md, logs/2026-05-01.md (example)")
        print("Log your days in logs/, then re-run without --init.")
        return

    print(f"Reading logs from {os.path.join(vault, O.DIRNAME, 'logs')} ...")
    res = O.sync(vault)
    d = res["desc"]
    print(f"  parsed {res['n_rows']} intervals over {d['n_days']} days")
    if d.get("has_labels"):
        print(f"  {d.get('n_slips', 0)} slips")
        if not res["fingerprint"].empty:
            top = res["fingerprint"].iloc[0]
            print(f"  dominant trigger: {top['cause']} ({top['share']:.0%})")
    print(f"\nWrote report -> {res['report']}")
    print("Open it in Obsidian (Reading view shows the charts).")


if __name__ == "__main__":
    main()
