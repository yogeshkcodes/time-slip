"""
Intervention simulator for Time Slip.

Because the simulator is a *causal* model, we can do what no observational
study can: apply an intervention (do-operator) and measure its effect on the
same people, same days, same randomness. We re-simulate the eight archetype
personas under each policy and compare slips/day and minutes lost/day against
baseline.

Policies are persona-level knobs, i.e. realistic behaviour changes:
    phone_away   always keep the phone out of reach during focused work
    dnd          batch/suppress notifications (rate cut to 10%)
    sleep_45m    go to bed ~45 minutes earlier on average
    all          all three combined

Caveat printed with every result: these are *model-implied* effects under the
simulator's assumptions - a hypothesis generator for a real A/B experiment,
not a guarantee.
"""

from __future__ import annotations
from dataclasses import replace
from typing import Dict, List
import numpy as np
import pandas as pd

from . import config as C
from .personas import build_personas
from .simulate import simulate_person

POLICIES: Dict[str, Dict] = {
    "baseline":   {},
    "phone_away": {"phone_away_policy": 1.0},
    "dnd":        {"notif_rate_mult": 0.10},
    "sleep_45m":  {"sleep_plus_h": 0.75},
    "all":        {"phone_away_policy": 1.0, "notif_rate_mult": 0.10,
                   "sleep_plus_h": 0.75},
}


def _apply(p, policy: Dict):
    q = replace(p)
    if "phone_away_policy" in policy:
        q.phone_away_policy = policy["phone_away_policy"]
    if "notif_rate_mult" in policy:
        q.notif_rate = p.notif_rate * policy["notif_rate_mult"]
    if "sleep_plus_h" in policy:
        q.baseline_sleep_h = min(9.0, p.baseline_sleep_h + policy["sleep_plus_h"])
    return q


def run_policies(policies: List[str] = None, seed: int = C.GLOBAL_SEED
                 ) -> pd.DataFrame:
    """Simulate the 8 archetypes under each policy; return per-policy effects."""
    names = policies or list(POLICIES.keys())
    rng = np.random.default_rng(seed)
    base_people = build_personas(np.random.default_rng(seed))
    # one fixed stream per person, reused across policies -> paired comparison
    streams = {p.pid: rng.integers(0, 2**63 - 1) for p in base_people}

    rows = []
    for name in names:
        pol = POLICIES[name]
        for p in base_people:
            q = _apply(p, pol)
            prng = np.random.default_rng(streams[p.pid])
            minutes, episodes = simulate_person(q, prng)
            days = max(1, len({m["day"] for m in minutes}))
            n_slips = sum(m["slip_onset"] for m in minutes)
            lost = sum(e["duration"] for e in episodes)
            phone_lost = sum(e["duration"] for e in episodes
                             if e["channel"] == "phone")
            rows.append(dict(policy=name, pid=p.pid,
                             slips_per_day=n_slips / days,
                             min_lost_per_day=lost / days,
                             phone_min_per_day=phone_lost / days))
    df = pd.DataFrame(rows)
    agg = df.groupby("policy").mean(numeric_only=True)
    base = agg.loc["baseline"]
    out = agg.copy()
    out["slips_change_%"] = 100 * (agg["slips_per_day"] / base["slips_per_day"] - 1)
    out["time_change_%"] = 100 * (agg["min_lost_per_day"] / base["min_lost_per_day"] - 1)
    order = [n for n in ["baseline", "phone_away", "dnd", "sleep_45m", "all"]
             if n in out.index]
    return out.loc[order].round(2)


if __name__ == "__main__":
    print("Simulating intervention policies on the 8 archetypes "
          f"({C.N_DAYS} days each; paired seeds) ...")
    tab = run_policies()
    print(tab.to_string())
    print("\nModel-implied effects under the simulator's assumptions - "
          "a hypothesis generator for a real experiment, not a guarantee.")
