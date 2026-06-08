"""
Feature engineering for *Time Slip*.

Turns the minute-level log into a supervised design matrix for predicting the
*onset* of an attention lapse in the next minute. We build two parallel feature
sets:

  REALISTIC  -- only quantities a person could plausibly self-log or that a
                phone/calendar could capture automatically (self-rated boredom/
                stress/energy/hunger/alertness, the task they're doing and how
                hard/interesting/aversive it feels, time-of-day, time-on-task,
                notifications, recent slip history, stable traits). This is the
                model you would actually deploy on real self-tracked data.

  ORACLE     -- the exact latent states that drive the generative hazard. Used
                only to validate that the recovery machinery in explain.py can
                reconstruct the true causal coefficients when given clean inputs.

Rows where the person is *already* inside a slip are removed: they are not "at
risk" of a new onset, so including them would bias the hazard. The train/test
split is by day (the last TEST_DAYS_FRACTION of each person's days are held
out), which prevents temporal leakage from rolling features.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from . import config as C


# Categorical context columns that get one-hot encoded.
_CAT_COLS = ["task_type", "location", "social"]

# Stable per-person trait columns (joined from the persona table).
_TRAIT_COLS = ["trait_self_control", "neuroticism", "conscientiousness",
               "habit_strength", "chronotype", "caffeine_use"]

# Oracle feature construction -> matches the terms in config.HAZARD exactly.
# Each entry maps an oracle feature name to (callable building it, true beta).
def _oracle_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    o = pd.DataFrame(index=df.index)
    o["boredom"]       = df["boredom"]
    o["fatigue"]       = df["fatigue"]
    o["stress"]        = df["stress"]
    o["aversive"]      = df["aversive"]
    o["vigilance"]     = df["vigilance"].clip(upper=C.VIGILANCE_CAP)
    o["hunger"]        = df["hunger"]
    o["urge"]          = df["urge_eff"]
    o["low_intrinsic"] = 1.0 - df["intrinsic"]
    o["low_mood"]      = 1.0 - df["mood"]
    o["focus_reserve"] = df["focus_reserve"]
    true_beta = {
        "boredom":       C.HAZARD["boredom"],
        "fatigue":       C.HAZARD["fatigue"],
        "stress":        C.HAZARD["stress"],
        "aversive":      C.HAZARD["aversive"],
        "vigilance":     C.HAZARD["vigilance"],
        "hunger":        C.HAZARD["hunger"],
        "urge":          C.HAZARD["urge"],
        "low_intrinsic": C.HAZARD["low_intrinsic"],
        "low_mood":      C.HAZARD["low_mood"],
        "focus_reserve": -C.HAZARD["self_control"],   # protective -> negative
    }
    return o, true_beta


def _add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Within-day rolling / recency features (vectorised; no future leakage)."""
    df = df.sort_values(["pid", "day", "clock_min"]).copy()
    keys = [df["pid"], df["day"]]
    g = df.groupby(["pid", "day"], sort=False)

    # notifications in the trailing 15 minutes
    df["notif_15"] = (g["notif"].rolling(15, min_periods=1).sum()
                      .reset_index(level=[0, 1], drop=True))

    # cumulative slip onsets earlier in the day (excludes "now")
    df["slips_today"] = g["slip_onset"].cumsum() - df["slip_onset"]

    # minutes since the previous slip onset (0 at an onset; 600 if none yet)
    onset_clock = df["clock_min"].where(df["slip_onset"] == 1)
    df["since_slip"] = (df["clock_min"] - onset_clock.groupby(keys).ffill()
                        ).clip(upper=600).fillna(600)

    # minutes since last meal (within the day; before any meal -> minutes awake)
    meal_clock = df["clock_min"].where(df["is_meal"] == 1)
    day_start = g["clock_min"].transform("first")
    df["since_meal"] = (df["clock_min"] - meal_clock.groupby(keys).ffill()
                        ).fillna(df["clock_min"] - day_start)

    # rolling mean of self-logged boredom/stress over trailing 20 min (trend)
    for col in ["boredom_obs", "stress_obs"]:
        df[col + "_roll20"] = (g[col].rolling(20, min_periods=1).mean()
                               .reset_index(level=[0, 1], drop=True))

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    return df


def build_features(minutes: pd.DataFrame, personas: pd.DataFrame
                   ) -> Dict[str, object]:
    """Return a dict with design matrices, label, splits and column groups."""
    df = minutes.merge(
        personas[["pid"] + _TRAIT_COLS], on="pid", how="left")
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df = _add_temporal(df)

    # at-risk set: cannot have a new onset while already inside a slip
    at_risk = df[df["in_slip"] == 0].copy()

    # ----- realistic numeric features -----
    realistic_numeric = [
        "boredom_obs", "stress_obs", "energy_obs", "hunger_obs", "alertness_obs",
        "boredom_obs_roll20", "stress_obs_roll20",
        "difficulty", "intrinsic", "aversive", "deadline", "open_tasks",
        "focus", "is_meal", "is_break", "phone_in_reach",
        "minutes_awake", "time_on_task", "vigilance", "notif", "notif_15",
        "slips_today", "since_slip", "since_meal", "hour_sin", "hour_cos",
        "sex_M",
    ] + _TRAIT_COLS

    # one-hot categoricals
    cat = pd.get_dummies(at_risk[_CAT_COLS], prefix=_CAT_COLS, dtype=float)
    X_real = pd.concat([at_risk[realistic_numeric].astype(float),
                        cat.reset_index(drop=True).set_index(at_risk.index)], axis=1)

    # ----- oracle features -----
    X_oracle, true_beta = _oracle_frame(at_risk)

    y = at_risk["slip_onset"].astype(int).to_numpy()

    # deployable target: does *any* slip onset occur within the next N minutes?
    # (forward-looking max over the horizon, computed within each person-day so
    # it never leaks across day boundaries)
    W = C.SLIP_HORIZON_MIN
    y_window = (at_risk.groupby(["pid", "day"])["slip_onset"]
                .transform(lambda s: s[::-1].rolling(W, min_periods=1).sum()[::-1])
                > 0).astype(int).to_numpy()

    # ----- split by day (last fraction held out, per person) -----
    max_day = at_risk["day"].max()
    cutoff = int(np.ceil((1 - C.TEST_DAYS_FRACTION) * (max_day + 1)))
    train_mask = (at_risk["day"] < cutoff).to_numpy()
    test_mask = ~train_mask

    meta = at_risk[["pid", "sex", "day", "clock_min", "hour", "activity",
                    "task_type", "focus", "slip_channel"]].reset_index(drop=True)

    return dict(
        X_real=X_real.reset_index(drop=True),
        X_oracle=X_oracle.reset_index(drop=True),
        y=y,
        y_window=y_window,
        horizon_min=C.SLIP_HORIZON_MIN,
        train_mask=train_mask,
        test_mask=test_mask,
        meta=meta,
        true_beta=true_beta,
        realistic_columns=list(X_real.columns),
        oracle_columns=list(X_oracle.columns),
        at_risk=at_risk.reset_index(drop=True),
        personas=personas,
    )


if __name__ == "__main__":
    from .simulate import simulate_all
    m, e, pt = simulate_all()
    fb = build_features(m, pt)
    print("X_real:", fb["X_real"].shape, "| X_oracle:", fb["X_oracle"].shape)
    print("positives:", int(fb["y"].sum()), "/", len(fb["y"]),
          f"({fb['y'].mean():.4f})")
    print("train/test rows:", int(fb["train_mask"].sum()), int(fb["test_mask"].sum()))
    print("n realistic features:", len(fb["realistic_columns"]))
