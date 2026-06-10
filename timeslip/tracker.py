"""
Analysis of real tracker data (from track_me.py) for Time Slip.

Takes the raw foreground-app samples and reconstructs the *behavioural* story of
your attention: focus spells, task switches, digital "slips" (work -> distractor
transitions), rabbit holes (long distractor dwells), and the real-data versions
of the project's core curves - the focus-survival curve and the vigilance
decrement - computed from YOUR actual computer use rather than simulation.

A "slip" here is detected behaviourally (you were on work apps, then dwelled on
a distractor), so no self-report is needed. That makes labels objective but
narrower than the simulator's five channels: phone pickups, snacks and
walk-aways show up only as idle gaps. Combine with the Obsidian/CSV self-log if
you want felt-state causes too.
"""

from __future__ import annotations
import glob
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# app / title categorisation
# ---------------------------------------------------------------------------
WORK_EXES = {
    "code.exe", "devenv.exe", "pycharm64.exe", "idea64.exe", "studio64.exe",
    "windowsterminal.exe", "powershell.exe", "cmd.exe", "wt.exe", "claude.exe",
    "winword.exe", "excel.exe", "powerpnt.exe", "onenote.exe", "obsidian.exe",
    "notion.exe", "acrobat.exe", "acrord32.exe", "python.exe", "jupyter.exe",
    "sublime_text.exe", "notepad++.exe", "notepad.exe", "rstudio.exe",
    "matlab.exe", "blender.exe", "photoshop.exe", "figma.exe",
}
COMM_EXES = {"outlook.exe", "olk.exe", "teams.exe", "ms-teams.exe", "slack.exe",
             "thunderbird.exe", "zoom.exe", "discord.exe", "whatsapp.exe",
             "telegram.exe", "signal.exe"}
GAME_EXES = {"steam.exe", "steamwebhelper.exe", "epicgameslauncher.exe",
             "league of legends.exe", "valorant.exe", "cs2.exe", "minecraft.exe",
             "robloxplayerbeta.exe", "fortniteclient-win64-shipping.exe"}
BROWSER_EXES = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
                "opera.exe", "opera_gx.exe", "arc.exe", "vivaldi.exe"}

DISTRACTOR_TITLES = ["youtube", "reddit", "twitter", "x.com", " / x", "instagram",
                     "facebook", "tiktok", "netflix", "twitch", "prime video",
                     "hotstar", "9gag", "buzzfeed", "imgur", "pinterest",
                     "hacker news", "9anime", "crunchyroll", "spotify"]
WORK_TITLES = ["github", "stack overflow", "stackoverflow", "docs.google",
               "overleaf", "jira", "confluence", "colab", "kaggle", "arxiv",
               "scholar", "wikipedia", "documentation", "pull request", "localhost",
               "chatgpt", "claude", "gemini", "jupyter", "papers", "pdf"]

IDLE_AWAY_S = 120        # idle longer than this = away from the computer
SLIP_MIN_WORK_S = 180    # must have been on work this long for a switch to count
SLIP_MIN_DWELL_S = 45    # must stay on the distractor this long to call it a slip
RABBIT_HOLE_MIN = 10     # distractor dwell (minutes) that counts as a rabbit hole


def categorize(exe: str, title: str, idle_s: float) -> str:
    if idle_s >= IDLE_AWAY_S:
        return "away"
    exe = (exe or "").lower(); t = (title or "").lower()
    if exe in GAME_EXES:
        return "distractor"
    if exe in BROWSER_EXES:
        if any(k in t for k in DISTRACTOR_TITLES):
            return "distractor"
        if any(k in t for k in WORK_TITLES):
            return "work"
        return "browse"
    if exe in WORK_EXES:
        return "work"
    if exe in COMM_EXES:
        return "communication"
    if any(k in t for k in DISTRACTOR_TITLES):
        return "distractor"
    return "other"


# ---------------------------------------------------------------------------
def load_tracker(dir_or_file: str) -> pd.DataFrame:
    """Load one tracker CSV or every track_*.csv in a directory."""
    if os.path.isdir(dir_or_file):
        paths = sorted(glob.glob(os.path.join(dir_or_file, "track_*.csv")))
    else:
        paths = [dir_or_file]
    if not paths:
        raise FileNotFoundError(f"No tracker CSVs in {dir_or_file}")
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["title"] = df["title"].fillna("")
    df["idle_s"] = pd.to_numeric(df["idle_s"], errors="coerce").fillna(0.0)
    df["category"] = [categorize(e, t, i) for e, t, i in
                      zip(df["exe"], df["title"], df["idle_s"])]
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    # sample spacing (seconds) -> duration each sample represents, capped at 60s
    dt = df["timestamp"].diff().dt.total_seconds().shift(-1)
    df["dur_s"] = dt.clip(upper=60).fillna(5.0)
    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse samples into contiguous same-category segments."""
    segs: List[dict] = []
    for date, g in df.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        runs = (g["category"] != g["category"].shift()).cumsum()
        for _, seg in g.groupby(runs):
            segs.append(dict(
                date=date,
                category=seg["category"].iloc[0],
                start=seg["timestamp"].iloc[0],
                end=seg["timestamp"].iloc[-1],
                dur_min=float(seg["dur_s"].sum() / 60.0),
                exe=seg["exe"].mode().iloc[0],
                title=seg["title"].iloc[0],
                hour=float(seg["hour"].iloc[0]),
            ))
    return pd.DataFrame(segs)


def detect_slips(segs: pd.DataFrame) -> pd.DataFrame:
    """Behavioural slips: work spell -> distractor dwell (objective labels)."""
    slips: List[dict] = []
    for date, g in segs.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        work_run_min = 0.0
        for i in range(len(g)):
            cat = g.loc[i, "category"]
            if cat == "work":
                work_run_min += g.loc[i, "dur_min"]
            elif cat in ("browse", "communication") and g.loc[i, "dur_min"] < 2:
                pass                                  # brief detour, keep the run
            elif cat == "distractor":
                if (work_run_min * 60 >= SLIP_MIN_WORK_S
                        and g.loc[i, "dur_min"] * 60 >= SLIP_MIN_DWELL_S):
                    slips.append(dict(
                        date=date, time=g.loc[i, "start"], hour=g.loc[i, "hour"],
                        dwell_min=float(g.loc[i, "dur_min"]),
                        after_work_min=float(work_run_min),
                        app=g.loc[i, "exe"], title=g.loc[i, "title"][:60],
                        rabbit_hole=bool(g.loc[i, "dur_min"] >= RABBIT_HOLE_MIN),
                    ))
                work_run_min = 0.0
            else:
                work_run_min = 0.0
    return pd.DataFrame(slips)


def focus_spells(segs: pd.DataFrame) -> pd.DataFrame:
    """Work spells with how they ended (distractor = event, else censored)."""
    rows: List[dict] = []
    for date, g in segs.groupby("date", sort=True):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            if g.loc[i, "category"] != "work":
                continue
            nxt = g.loc[i + 1, "category"] if i + 1 < len(g) else None
            rows.append(dict(
                date=date, dur_min=float(g.loc[i, "dur_min"]),
                hour=float(g.loc[i, "hour"]),
                ended_in_distraction=int(nxt == "distractor"),
            ))
    return pd.DataFrame(rows)


def vigilance_real(spells: pd.DataFrame, step: int = 5, cap: int = 60) -> pd.DataFrame:
    """Real vigilance decrement: P(spell ends in distraction) by spell length bin."""
    if spells.empty:
        return pd.DataFrame(columns=["dur_bin", "p_distract", "n"])
    s = spells.copy()
    s["dur_bin"] = (s["dur_min"] // step * step).clip(upper=cap)
    out = (s.groupby("dur_bin")
             .agg(p_distract=("ended_in_distraction", "mean"), n=("dur_min", "size"))
             .reset_index())
    return out[out["n"] >= 5]


def summarize(df: pd.DataFrame, segs: pd.DataFrame, slips: pd.DataFrame) -> Dict:
    days = max(1, df["date"].nunique())
    by_cat = (df.groupby("category")["dur_s"].sum() / 60.0 / days)
    top_distractors = (segs[segs["category"] == "distractor"]
                       .groupby("exe")["dur_min"].sum()
                       .sort_values(ascending=False).head(8) / days)
    sp = focus_spells(segs)
    return dict(
        n_days=days,
        hours_tracked=float(df["dur_s"].sum() / 3600.0),
        min_per_day=by_cat.to_dict(),
        n_slips=int(len(slips)),
        slips_per_day=float(len(slips) / days),
        rabbit_holes=int(slips["rabbit_hole"].sum()) if len(slips) else 0,
        time_lost_per_day=float(slips["dwell_min"].sum() / days) if len(slips) else 0.0,
        median_focus_min=float(sp["dur_min"].median()) if len(sp) else float("nan"),
        longest_focus_min=float(sp["dur_min"].max()) if len(sp) else float("nan"),
        top_distractors=top_distractors.to_dict(),
        switches_per_hour=float(len(segs) / max(0.1, df["dur_s"].sum() / 3600.0)),
    )


# ---------------------------------------------------------------------------
# demo generator: synthesises a plausible tracker file so the analyzer can be
# tested before you have your own data. Clearly labelled as a demo.
# ---------------------------------------------------------------------------
def make_demo_tracker(path: str, days: int = 5, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    apps_work = [("code.exe", "main.py - timeslip - Visual Studio Code"),
                 ("winword.exe", "draft.docx - Word"),
                 ("chrome.exe", "xgboost docs - Google Chrome")]
    apps_dis = [("chrome.exe", "YouTube - Google Chrome"),
                ("chrome.exe", "reddit: the front page - Google Chrome"),
                ("chrome.exe", "Instagram - Google Chrome")]
    apps_comm = [("olk.exe", "Inbox - Outlook"), ("slack.exe", "Slack - #general")]
    for d in range(days):
        base = pd.Timestamp("2026-06-01") + pd.Timedelta(days=d, hours=9)
        t = base
        end = base + pd.Timedelta(hours=8)
        while t < end:
            hrs = (t - base).total_seconds() / 3600.0
            p_dis = 0.10 + 0.10 * (1 if 4.5 < hrs < 6 else 0)   # post-lunch dip
            r = rng.random()
            if r < p_dis:
                exe, title = apps_dis[rng.integers(len(apps_dis))]
                dur = float(rng.lognormal(np.log(6), 0.7))       # heavy tail
            elif r < p_dis + 0.12:
                exe, title = apps_comm[rng.integers(len(apps_comm))]
                dur = float(rng.uniform(2, 8))
            else:
                exe, title = apps_work[rng.integers(len(apps_work))]
                dur = float(rng.lognormal(np.log(14), 0.6))
            dur = min(dur, 45.0)
            for k in range(int(dur * 60 // 15)):
                rows.append([(t + pd.Timedelta(seconds=15 * k))
                             .isoformat(timespec="seconds"), exe, title,
                             float(rng.uniform(0, 20))])
            t += pd.Timedelta(minutes=dur)
    df = pd.DataFrame(rows, columns=["timestamp", "exe", "title", "idle_s"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
