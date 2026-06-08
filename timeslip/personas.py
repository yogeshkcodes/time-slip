"""
Persona definitions for *Time Slip*.

We model eight hand-built archetypes (``build_personas``); the study cohort
(``build_cohort``) extends these with randomised members sampled across the trait
space, for a larger, harder population. None are named; each is identified by an
opaque id (P01..PNN), a sex label (M/F) and a short archetype describing their
life context. Their *stable traits* are expressed on 0..1 scales chosen to map
onto established individual-difference constructs:

  trait_self_control   Tangney, Baumeister & Boone (2004) Brief Self-Control Scale.
                       Higher -> larger, slower-depleting focus reserve.
  neuroticism          Big Five. Higher -> stronger stress reactivity.
  conscientiousness    Big Five. Higher -> stronger guilt after off-task slips,
                       and a slightly higher protective effect on aversive tasks.
  habit_strength       Smartphone-use automaticity (Self-Report Habit Index;
                       Oulasvirta et al. 2012). Higher -> stronger phone urge.
  chronotype           0 = strong morning type, 1 = strong evening type
                       (Roenneberg MCTQ). Shifts the circadian alertness curve.
  caffeine_use         0..1 propensity to use caffeine, which transiently masks
                       sleep pressure in the late morning / early afternoon.
  baseline_sleep_h     Habitual sleep duration; shortfall accrues sleep debt.
  notif_rate           Per-hour notification arrival multiplier (job dependent).
  phone_away_policy    Probability the phone is *not* in reach during focused
                       work (a behavioural self-binding strategy).

Gender is recorded because the brief asks for it and a real study would report
it, but by design it has **no direct edge** into the slip hazard (see
docs/THEORY.md). Differences between the two men/women below are driven by their
trait and context profiles, not by sex per se. This is a deliberate modelling
choice to avoid baking in a stereotype the data cannot justify.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List
import numpy as np


@dataclass
class Persona:
    pid: str
    sex: str                    # "M" or "F"
    archetype: str
    trait_self_control: float
    neuroticism: float
    conscientiousness: float
    habit_strength: float
    chronotype: float           # 0 morning .. 1 evening
    caffeine_use: float
    baseline_sleep_h: float
    notif_rate: float           # notifications per hour (job-dependent base)
    phone_away_policy: float    # P(phone out of reach during focused work)
    intercept: float = 0.0      # random-effect intercept on the slip hazard
    obs_noise: float = 0.09     # this person's self-report noise (heterogeneous)
    drift: float = 0.0          # slow drift of habit/self-control over the period

    def to_row(self) -> Dict:
        return asdict(self)


def build_personas(rng: np.random.Generator) -> List[Persona]:
    """Return the fixed roster of eight people.

    Trait values are hand-set to span the space (a disciplined morning lark, an
    anxious night owl, a fragmented-day parent, a shift worker, etc.) so that
    the recovered 'slip fingerprints' differ in interpretable ways. A small
    random-effect intercept is drawn per person to reflect unmodelled baseline
    differences in lapse propensity.
    """
    specs = [
        # pid    sex  archetype                      sc   neu  con  hab  chr  caf  sleep notif away
        ("P01", "F", "early-career analyst, anxious night owl",
                                                     0.42, 0.78, 0.55, 0.80, 0.82, 0.6, 6.3, 9.0, 0.15),
        ("P02", "M", "senior engineer, disciplined morning lark",
                                                     0.86, 0.30, 0.82, 0.28, 0.18, 0.5, 7.6, 5.0, 0.75),
        ("P03", "F", "graduate student, intrinsically driven but procrastinates",
                                                     0.55, 0.58, 0.60, 0.74, 0.70, 0.7, 6.8, 7.0, 0.30),
        ("P04", "M", "sales manager, meeting- and notification-heavy",
                                                     0.50, 0.45, 0.58, 0.66, 0.45, 0.8, 6.9, 18.0, 0.10),
        ("P05", "F", "freelance creative, flow-prone but boredom-sensitive",
                                                     0.48, 0.52, 0.40, 0.78, 0.66, 0.6, 7.1, 8.0, 0.20),
        ("P06", "M", "rotating-shift worker, circadian-disrupted",
                                                     0.58, 0.50, 0.62, 0.55, 0.50, 0.9, 6.0, 7.0, 0.40),
        ("P07", "F", "parent with part-time work, fragmented day",
                                                     0.62, 0.60, 0.70, 0.60, 0.40, 0.6, 6.2, 11.0, 0.25),
        ("P08", "M", "student athlete, well-rested, structured",
                                                     0.72, 0.38, 0.68, 0.50, 0.35, 0.4, 8.0, 8.0, 0.55),
    ]
    personas: List[Persona] = []
    for s in specs:
        p = Persona(
            pid=s[0], sex=s[1], archetype=s[2],
            trait_self_control=s[3], neuroticism=s[4], conscientiousness=s[5],
            habit_strength=s[6], chronotype=s[7], caffeine_use=s[8],
            baseline_sleep_h=s[9], notif_rate=s[10], phone_away_policy=s[11],
            intercept=float(rng.normal(0.0, 0.25)),
        )
        personas.append(p)
    return personas


# job-flavour templates for the randomised cohort (notif rate, phone policy)
_JOB_FLAVOURS = [
    ("knowledge worker",        (5, 12),  (0.25, 0.75)),
    ("manager / sales",         (12, 22), (0.05, 0.25)),
    ("student",                 (6, 12),  (0.15, 0.45)),
    ("creative / freelance",    (6, 12),  (0.15, 0.45)),
    ("operations / shift",      (5, 10),  (0.30, 0.55)),
    ("caregiver / part-time",   (8, 16),  (0.15, 0.40)),
]


def build_cohort(n: int, rng: np.random.Generator,
                 obs_noise_range=(0.06, 0.16), drift_max=0.18):
    """Return ``n`` people: the 8 fixed archetypes plus randomised draws.

    Randomised members sample their traits from plausible ranges so the cohort
    spans a realistic population rather than eight hand-picked points. Each
    person also gets a heterogeneous self-report noise level and a small slow
    drift in habit strength / self-control over the study window.
    """
    people = build_personas(rng)
    for p in people:                      # give the archetypes realism knobs too
        p.obs_noise = float(rng.uniform(*obs_noise_range))
        p.drift = float(rng.uniform(-drift_max, drift_max))

    for i in range(len(people), max(len(people), n)):
        sex = "M" if rng.random() < 0.5 else "F"
        job, notif_rng, away_rng = _JOB_FLAVOURS[rng.integers(len(_JOB_FLAVOURS))]
        people.append(Persona(
            pid=f"P{i+1:02d}", sex=sex, archetype=f"randomised cohort member ({job})",
            trait_self_control=float(np.clip(rng.normal(0.6, 0.15), 0.28, 0.92)),
            neuroticism=float(np.clip(rng.normal(0.5, 0.17), 0.15, 0.9)),
            conscientiousness=float(np.clip(rng.normal(0.6, 0.15), 0.25, 0.92)),
            habit_strength=float(np.clip(rng.normal(0.6, 0.16), 0.25, 0.95)),
            chronotype=float(np.clip(rng.normal(0.5, 0.2), 0.12, 0.92)),
            caffeine_use=float(np.clip(rng.normal(0.65, 0.18), 0.2, 0.98)),
            baseline_sleep_h=float(np.clip(rng.normal(7.0, 0.7), 5.3, 8.6)),
            notif_rate=float(rng.uniform(*notif_rng)),
            phone_away_policy=float(rng.uniform(*away_rng)),
            intercept=float(rng.normal(0.0, 0.25)),
            obs_noise=float(rng.uniform(*obs_noise_range)),
            drift=float(rng.uniform(-drift_max, drift_max)),
        ))
    return people[:n] if n >= 1 else people
