"""
Time Slip public API - the embeddable hook for other projects.

A small, stable facade so any app (a focus timer, a journaling tool, an Obsidian
plugin, a wearable companion, a study backend) can get attention-risk and cause
attribution without touching the internals.

    from timeslip.api import TimeSlip
    ts = TimeSlip()                                   # loads the trained model
    r = ts.risk(boredom=4, task="deep_work", tot=40, phone=1, notifs=3)
    print(r["risk"], r["interval"], r["best_lever"]["name"])

    fp = ts.fingerprint("my_log.csv")                 # personal cause breakdown
    beh = ts.tracker("outputs/tracker")               # behaviour from tracked data

Everything is local and returns plain dicts / DataFrames (JSON-friendly), so it
drops into a Flask/FastAPI route, a notebook, or another pipeline unchanged.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "outputs", "model", "timeslip_model.joblib")

FOCUS_TASKS = {"deep_work", "study", "admin", "meeting", "shift_work"}

# how each lever nudges the model inputs toward a calmer state (0-1 scale)
LEVERS = {
    "phone_away":     ("Phone away + notifications off",
                       {"phone_in_reach": 0.0, "notif": 0.0, "notif_15": 0.0}),
    "break":          ("Take a short break (reset time-on-task)",
                       {"time_on_task": 0.0, "vigilance": 0.0}),
    "eat":            ("Eat something",
                       {"hunger_obs": 0.25, "since_meal": 0.0}),
    "reframe":        ("Make the task engaging / reframe it",
                       {"intrinsic": 0.75, "aversive": 0.25}),
    "destress":       ("Down-regulate stress (breathing, walk)",
                       {"stress_obs": 0.25, "stress_obs_roll20": 0.25}),
    "energize":       ("Boost alertness (rest, light, caffeine)",
                       {"alertness_obs": 0.75, "energy_obs": 0.75}),
}


def _lk(x) -> float:
    return (float(x) - 1.0) / 4.0


class TimeSlip:
    """Loaded once; cheap to call repeatedly."""

    def __init__(self, model_path: str = DEFAULT_MODEL):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run `python run_all.py` once.")
        import joblib
        self._prod = joblib.load(model_path)
        self.columns: List[str] = self._prod["columns"]
        self.horizon: int = int(self._prod["horizon"])

    # ---- live risk + recommendation ---------------------------------------
    def _row(self, s: Dict) -> pd.DataFrame:
        g = lambda k, d: s.get(k, d)
        task = g("task", "deep_work")
        focus = 1 if task in FOCUS_TASKS else 0
        tot = float(g("tot", 20))
        vig = min(tot / 60.0, 2.0) if focus else min(0.5, tot / 120.0)
        hour = float(g("hour", 14.0))
        row = {
            "boredom_obs": _lk(g("boredom", 3)), "stress_obs": _lk(g("stress", 3)),
            "energy_obs": _lk(g("energy", 3)), "hunger_obs": _lk(g("hunger", 2)),
            "alertness_obs": _lk(g("alertness", 3)),
            "boredom_obs_roll20": _lk(g("boredom", 3)),
            "stress_obs_roll20": _lk(g("stress", 3)),
            "difficulty": _lk(g("difficulty", 3)), "intrinsic": _lk(g("intrinsic", 3)),
            "aversive": _lk(g("aversive", 3)), "deadline": _lk(g("deadline", 3)),
            "open_tasks": float(g("open_tasks", 2)), "focus": focus,
            "is_meal": 0, "is_break": 0, "phone_in_reach": int(g("phone", 1)),
            "minutes_awake": max(0.0, (hour - 7.0) * 60),
            "time_on_task": tot, "vigilance": vig,
            "notif": min(float(g("notifs", 2)), 2), "notif_15": float(g("notifs", 2)),
            "slips_today": float(g("slips_today", 3)), "since_slip": 600.0,
            "since_meal": float(g("since_meal", 120)),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24), "sex_M": 0.5,
            "trait_self_control": 0.5, "neuroticism": 0.5, "conscientiousness": 0.5,
            "habit_strength": 0.5, "chronotype": 0.5, "caffeine_use": 0.5,
            f"task_type_{task}": 1.0, "location_office": 1.0, "social_alone": 1.0,
        }
        return pd.DataFrame([row]).reindex(columns=self.columns, fill_value=0.0)

    def _raw(self, X) -> float:
        return float(self._prod["model"].predict_proba(X)[:, 1][0])

    def _cal(self, raw: float) -> float:
        return float(self._prod["iso"].transform([raw])[0])

    def risk(self, **state) -> Dict:
        """Calibrated next-horizon slip risk + interval + ranked levers."""
        X = self._row(state)
        raw = self._raw(X)
        risk = self._cal(raw)
        interval = None
        if "cal_scores" in self._prod:
            from .evaluate import venn_abers_interval
            lo, hi = venn_abers_interval(self._prod["cal_scores"],
                                         self._prod["cal_y"], raw)
            interval = [round(lo, 3), round(hi, 3)]
        levers = []
        for key, (name, mods) in LEVERS.items():
            Xc = X.copy()
            for col, val in mods.items():
                if col in Xc.columns:
                    Xc[col] = val
            raw_c = self._raw(Xc)
            risk_c = self._cal(raw_c)
            levers.append({"id": key, "name": name, "new_risk": round(risk_c, 3),
                           "reduction": round(max(0.0, risk - risk_c), 3),
                           "_raw_drop": raw - raw_c})
        # rank by the model's raw-score drop (isotonic calibration has flat steps
        # that would otherwise collapse distinct levers to the same value)
        levers.sort(key=lambda d: d["_raw_drop"], reverse=True)
        for d in levers:
            d.pop("_raw_drop", None)
        return {"risk": round(risk, 3), "horizon_min": self.horizon,
                "interval": interval, "best_lever": levers[0], "levers": levers}

    # ---- personal fingerprint from a log ----------------------------------
    def fingerprint(self, log) -> pd.DataFrame:
        """Cause-share breakdown from a self-log (path or DataFrame)."""
        from . import realdata as R
        df = R.load_log(log) if isinstance(log, str) else log
        df = df.sort_values(["date", "clock_min"]).reset_index(drop=True)
        d = R.prepare(df)
        mres = R.personal_model(d, df)
        if not (mres and mres.get("enough")):
            return self.population_fingerprint()
        return R.personal_fingerprint(mres, d)

    def population_fingerprint(self) -> pd.DataFrame:
        return self._prod.get("population_fingerprint",
                              pd.DataFrame(columns=["cause", "share"]))

    # ---- behaviour from tracked data --------------------------------------
    def tracker(self, path: str) -> Dict:
        from . import tracker as T
        df = T.load_tracker(path)
        segs = T.build_segments(df)
        slips = T.detect_slips(segs)
        return T.summarize(df, segs, slips)


# module-level convenience (lazy singleton)
_INSTANCE: Optional[TimeSlip] = None


def get() -> TimeSlip:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TimeSlip()
    return _INSTANCE


if __name__ == "__main__":
    ts = TimeSlip()
    import json
    print("API smoke test - risk for a bored, long-on-task deep-work moment:")
    print(json.dumps(ts.risk(boredom=4, stress=3, task="deep_work", tot=40,
                             phone=1, notifs=3), indent=2))
    print("\npopulation fingerprint:")
    print(ts.population_fingerprint().to_string(index=False))
