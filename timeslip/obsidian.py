"""
Obsidian integration for Time Slip.

Keep a folder ``TimeSlip/`` inside your Obsidian vault, log your routine in
daily notes as a simple markdown table, and this turns those notes into a
report *inside the vault* (with embedded figures) — so the whole loop lives in
Obsidian.

Layout created inside the vault:

    <vault>/TimeSlip/
        README.md            how to log
        Daily Log Template.md a table you can copy into each day's note
        logs/                your daily logs (one .md per day, a table inside)
            2026-05-01.md    (an example is written for you)
        figures/             generated charts (written by sync)
        Time Slip Report.md  generated report (written by sync)

CLI:
    python obsidian_sync.py "C:/path/to/Vault" --init    # scaffold the folder
    python obsidian_sync.py "C:/path/to/Vault"           # parse logs -> report
"""

from __future__ import annotations
import os
import glob
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from . import realdata as R
from . import schema as S

DIRNAME = "TimeSlip"

# columns we expect in a log table (friendly: uses HH:MM "time" instead of clock_min)
_LOG_COLUMNS = ["time", "activity", "task_type", "difficulty", "intrinsic",
                "aversive", "boredom", "stress", "energy", "hunger", "alertness",
                "location", "social", "phone_in_reach", "notif", "deadline",
                "open_tasks", "slip", "slip_channel", "slip_minutes"]

_DEFAULTS = {"notif": 0, "open_tasks": 1, "deadline": 3, "phone_in_reach": 1,
             "slip": 0, "slip_channel": "", "slip_minutes": 0,
             "difficulty": 3, "intrinsic": 3, "aversive": 3, "boredom": 2,
             "stress": 2, "energy": 3, "hunger": 2, "alertness": 3,
             "location": "home", "social": "alone"}

_EXAMPLE = """\
## Routine log

| time | activity | task_type | difficulty | intrinsic | aversive | boredom | stress | energy | hunger | alertness | location | social | phone_in_reach | notif | deadline | open_tasks | slip | slip_channel | slip_minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 08:30 | morning email | admin | 3 | 2 | 4 | 3 | 3 | 3 | 2 | 3 | office | alone | 1 | 4 | 4 | 3 | 1 | phone | 8 |
| 09:30 | deep work | deep_work | 4 | 4 | 3 | 2 | 3 | 4 | 2 | 4 | office | alone | 0 | 1 | 4 | 3 | 0 |  | 0 |
| 11:00 | deep work | deep_work | 4 | 3 | 4 | 4 | 3 | 3 | 3 | 3 | office | alone | 1 | 3 | 4 | 3 | 1 | mind_wandering | 4 |
| 12:30 | lunch | meal | 1 | 4 | 1 | 1 | 2 | 3 | 4 | 3 | cafe | colleagues | 1 | 5 | 1 | 0 | 1 | social | 10 |
| 14:00 | meeting | meeting | 3 | 3 | 4 | 4 | 4 | 2 | 2 | 2 | office | colleagues | 1 | 6 | 4 | 4 | 1 | phone | 6 |
"""

_README = """\
# Time Slip — log your routine here

Log your day in a **markdown table** inside `logs/` (one note per day, named
like `2026-05-01.md`). Copy the table from *Daily Log Template* and add a row
every ~15–30 minutes, or once per activity block.

Columns (most are 1–5 scales; leave blanks and they default sensibly):
- **time** — HH:MM at the start of the interval
- **activity / task_type** — what you were doing (task_type categories:
  deep_work, meeting, admin, study, meal, break, commute, chores, childcare,
  exercise, leisure, social…)
- **difficulty, intrinsic, aversive** — how hard / interesting / dreaded (1–5)
- **boredom, stress, energy, hunger, alertness** — how you felt (1–5)
- **location, social** — where, and who with (alone/colleagues/family/friends)
- **phone_in_reach** — 1 if your phone was within reach, else 0
- **notif** — notifications received in the interval
- **deadline** — deadline pressure right now (1–5); **open_tasks** — count
- **slip** — 1 if your attention slipped in this interval, else 0
- **slip_channel** — if it slipped: phone / mind_wandering / task_switch / snack / social
- **slip_minutes** — roughly how long you were off-task

Then run, from the project folder:

```
python obsidian_sync.py "<path to this vault>"
```

It writes **Time Slip Report.md** (with charts) back into this folder.
"""


# --------------------------------------------------------------------------- #
def _ts_dir(vault: str) -> str:
    return os.path.join(vault, DIRNAME)


def make_vault(vault: str) -> str:
    """Scaffold the TimeSlip folder inside an existing Obsidian vault."""
    base = _ts_dir(vault)
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base, "figures"), exist_ok=True)
    with open(os.path.join(base, "README.md"), "w", encoding="utf-8") as f:
        f.write(_README)
    with open(os.path.join(base, "Daily Log Template.md"), "w", encoding="utf-8") as f:
        f.write("# {{date}}\n\n" + _EXAMPLE)
    example_path = os.path.join(base, "logs", "2026-05-01.md")
    if not os.path.exists(example_path):
        with open(example_path, "w", encoding="utf-8") as f:
            f.write("# 2026-05-01 (example — replace with your own)\n\n" + _EXAMPLE)
    return base


# --------------------------------------------------------------------------- #
def _parse_tables(text: str) -> List[Dict]:
    """Extract rows from all GitHub-flavoured markdown tables in a note."""
    rows: List[Dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        is_table = line.startswith("|") and i + 1 < len(lines)
        if is_table:
            sep = lines[i + 1].strip()
            sep_ok = set(sep.replace("|", "").replace(":", "").replace(" ", "")) <= set("-") \
                and "-" in sep
            if sep_ok:
                header = [c.strip() for c in line.strip().strip("|").split("|")]
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if len(cells) == len(header):
                        rows.append(dict(zip(header, cells)))
                    j += 1
                i = j
                continue
        i += 1
    return rows


def parse_vault(vault: str) -> pd.DataFrame:
    """Read every daily log in TimeSlip/logs/ into a self-log DataFrame."""
    base = _ts_dir(vault)
    log_dir = os.path.join(base, "logs")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"No logs folder at {log_dir}. Run with --init first.")
    rows: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(path))[0]
        # date from filename (YYYY-MM-DD...) if the table omits it
        file_date = stem[:10] if len(stem) >= 10 and stem[4] == "-" else stem
        for r in _parse_tables(text):
            if "example" in str(r.get("activity", "")).lower():
                pass  # keep example rows too; they are valid data
            r.setdefault("date", file_date)
            if not r.get("date"):
                r["date"] = file_date
            rows.append(r)
    if not rows:
        raise ValueError(f"No log tables found under {log_dir}.")

    df = pd.DataFrame(rows)

    # time HH:MM -> clock_min (minutes from midnight)
    if "clock_min" not in df.columns and "time" in df.columns:
        def _to_min(t):
            try:
                h, m = str(t).split(":")[:2]
                return int(h) * 60 + int(m)
            except Exception:
                return np.nan
        df["clock_min"] = df["time"].map(_to_min)

    # fill missing optional columns with neutral defaults, coerce types
    for col, dv in _DEFAULTS.items():
        if col not in df.columns:
            df[col] = dv
    numeric = ["clock_min", "difficulty", "intrinsic", "aversive", "boredom",
               "stress", "energy", "hunger", "alertness", "phone_in_reach",
               "notif", "deadline", "open_tasks", "slip", "slip_minutes"]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["slip"] = df["slip"].fillna(0).astype(int)
    for c in ["difficulty", "intrinsic", "aversive", "boredom", "stress",
              "energy", "hunger", "alertness", "deadline"]:
        df[c] = df[c].fillna(_DEFAULTS[c]).clip(1, 5)
    df["phone_in_reach"] = df["phone_in_reach"].fillna(1).clip(0, 1).astype(int)
    df["notif"] = df["notif"].fillna(0).clip(lower=0)
    df["open_tasks"] = df["open_tasks"].fillna(0).clip(lower=0)
    df["slip_minutes"] = df["slip_minutes"].fillna(0)
    df = df.dropna(subset=["clock_min"]).reset_index(drop=True)
    df["clock_min"] = df["clock_min"].astype(int)
    return df


# --------------------------------------------------------------------------- #
def _report_markdown(desc: Dict, fp: pd.DataFrame, mres, fig_rel: str) -> str:
    L = ["# Time Slip Report",
         f"*Auto-generated from your logs. {desc['n_rows']} intervals "
         f"across {desc['n_days']} days.*", ""]
    if not desc.get("has_labels"):
        L += ["> Add a `slip` column (1 when your attention slipped) to your logs "
              "to unlock the where/when/why analysis and your fingerprint."]
        return "\n".join(L)

    L.append(f"- **{desc['n_slips']} slips** "
             f"(~{desc['n_slips']/max(1,desc['n_days']):.1f}/day).")
    if "time_lost_per_day" in desc:
        L.append(f"- **~{desc['time_lost_per_day']:.0f} min/day** off-task.")
    if "channels" in desc:
        L.append("- Channels: " +
                 ", ".join(f"{k} {v:.0%}" for k, v in desc["channels"].items()))

    L += ["", "## When & where", f"![]({fig_rel}/me_when.png)"]
    L += ["", "## Why — what's elevated when you slip", f"![]({fig_rel}/me_why.png)"]

    if mres and mres.get("enough") and not fp.empty:
        L += ["", "## Your fingerprint", f"![]({fig_rel}/me_fingerprint.png)", "",
              "| Cause | Share |", "|---|---|"]
        for _, r in fp.iterrows():
            L.append(f"| {r['cause']} | {r['share']:.0%} |")
        L += ["", f"**Dominant trigger: {fp.iloc[0]['cause']}.**",
              "", f"*(personal model cross-validated AUC "
              f"{mres['auc']:.2f}; 0.50 = chance.)*"]
    else:
        n = mres["n"] if mres else desc["n_rows"]
        ns = mres["n_slips"] if mres else desc.get("n_slips", 0)
        L += ["", "## Your fingerprint",
              f"Not enough data yet ({n} rows / {ns} slips). Aim for "
              f"{R.MIN_ROWS_FOR_MODEL}+ rows and {R.MIN_SLIPS_FOR_MODEL}+ slips."]
    L += ["", "---", "*Re-run `python obsidian_sync.py \"<vault>\"` after logging "
          "more to refresh this report.*"]
    return "\n".join(L)


def sync(vault: str) -> Dict:
    """Parse the vault's logs, analyse them, and write a report back into it."""
    base = _ts_dir(vault)
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    df = parse_vault(vault)
    problems = S.validate_self_log(df)
    if problems:
        raise ValueError("Log problems:\n  - " + "\n  - ".join(problems))

    d = R.prepare(df)
    desc = R.descriptive(df, d)
    mres = R.personal_model(d, df)
    fp = pd.DataFrame(columns=["cause", "share"])
    if mres and mres.get("enough"):
        fp = R.personal_fingerprint(mres, d)

    R._fig_when(desc, os.path.join(fig_dir, "me_when.png"))
    R._fig_why(desc, os.path.join(fig_dir, "me_why.png"))
    auc = mres["auc"] if (mres and mres.get("enough")) else float("nan")
    R._fig_fingerprint(fp, auc, os.path.join(fig_dir, "me_fingerprint.png"))

    md = _report_markdown(desc, fp, mres, "figures")
    report_path = os.path.join(base, "Time Slip Report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return dict(report=report_path, desc=desc, fingerprint=fp, n_rows=len(df))
