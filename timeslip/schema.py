"""
Self-logging schema for *Time Slip* — the bridge from simulation to real data.

The whole point of restricting the deployed model to self-loggable features is
that a real person (or their phone/calendar) can supply exactly the same inputs.
This module defines that schema, writes a blank template you can fill in, and
validates a completed log so it can flow into the same feature pipeline.

A row = one logged interval (e.g. every 15–30 min via experience sampling, or
one row per activity block). Likert fields are 1–5 and are rescaled to 0–1 to
match the simulator's `*_obs` columns.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd


@dataclass
class Field:
    name: str
    kind: str            # "time" | "likert" | "cat" | "binary" | "count" | "label"
    desc: str
    example: object


SELF_LOG_FIELDS: List[Field] = [
    Field("date",          "time",   "calendar date (YYYY-MM-DD)", "2026-06-08"),
    Field("clock_min",     "time",   "minutes since midnight at start of interval", 555),
    Field("activity",      "cat",    "what you were doing (free label)", "deep_work"),
    Field("task_type",     "cat",    "category: deep_work/meeting/admin/study/meal/break/commute/chores/childcare/exercise/leisure/social/...", "deep_work"),
    Field("difficulty",    "likert", "how cognitively demanding (1 easy – 5 very hard)", 4),
    Field("intrinsic",     "likert", "how intrinsically interesting/valued (1 – 5)", 3),
    Field("aversive",      "likert", "how much you wanted to avoid it (1 – 5)", 4),
    Field("boredom",       "likert", "felt boredom right then (1 – 5)", 2),
    Field("stress",        "likert", "felt stress/anxiety (1 – 5)", 3),
    Field("energy",        "likert", "physical/mental energy (1 – 5)", 3),
    Field("hunger",        "likert", "hunger (1 – 5)", 2),
    Field("alertness",     "likert", "alertness/wakefulness (1 – 5)", 3),
    Field("location",      "cat",    "home/office/transit/cafe/outdoors/gym/...", "office"),
    Field("social",        "cat",    "alone/colleagues/family/friends", "alone"),
    Field("phone_in_reach","binary", "was your phone within reach? (0/1)", 1),
    Field("notif",         "count",  "notifications received in this interval", 2),
    Field("deadline",      "likert", "deadline pressure right now (1 – 5)", 4),
    Field("open_tasks",    "count",  "roughly how many tasks were open/unfinished", 3),
    # ---- labels (only needed to TRAIN on your own data; optional for inference) ----
    Field("slip",          "label",  "did your attention slip in this interval? (0/1)", 1),
    Field("slip_channel",  "label",  "if slip: phone/mind_wandering/task_switch/snack/social", "phone"),
    Field("slip_minutes",  "label",  "if slip: roughly how long off-task (min)", 12),
]

LIKERT_FIELDS = [f.name for f in SELF_LOG_FIELDS if f.kind == "likert"]
REQUIRED_FOR_INFERENCE = [f.name for f in SELF_LOG_FIELDS
                          if f.kind in ("time", "cat", "likert", "binary", "count")]


def make_template_csv(path: str, example_rows: int = 2) -> None:
    """Write a blank/example CSV you can fill in to log a real day."""
    cols = [f.name for f in SELF_LOG_FIELDS]
    ex = {f.name: f.example for f in SELF_LOG_FIELDS}
    df = pd.DataFrame([ex] * max(1, example_rows), columns=cols)
    df.to_csv(path, index=False)


def validate_self_log(df: pd.DataFrame) -> List[str]:
    """Return a list of problems; empty list means the log is usable."""
    problems = []
    missing = [c for c in REQUIRED_FOR_INFERENCE if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
    for c in LIKERT_FIELDS:
        if c in df.columns:
            bad = df[(df[c] < 1) | (df[c] > 5)][c]
            if len(bad):
                problems.append(f"'{c}' has {len(bad)} values outside 1–5")
    if "phone_in_reach" in df.columns and not df["phone_in_reach"].isin([0, 1]).all():
        problems.append("'phone_in_reach' must be 0 or 1")
    return problems


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rescale a validated self-log into the simulator's observed-feature space.

    Likert 1–5 -> 0–1 via (x-1)/4, matching the `*_obs` columns the model was
    trained on. (A full deployment would re-fit the model on pooled real data;
    this helper makes the column conventions line up.)
    """
    out = df.copy()
    for c in LIKERT_FIELDS:
        if c in out.columns:
            out[c] = (out[c].astype(float) - 1.0) / 4.0
    rename = {"boredom": "boredom_obs", "stress": "stress_obs",
              "energy": "energy_obs", "hunger": "hunger_obs",
              "alertness": "alertness_obs"}
    out = out.rename(columns=rename)
    return out


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "outputs", "data", "self_log_template.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_template_csv(path)
    print("wrote template:", path)
    print("required for inference:", REQUIRED_FOR_INFERENCE)
