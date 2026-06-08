"""
Persona definitions for *Time Slip*.

We model eight people. None are named; each is identified by an opaque id
(P01..P08), a sex label (M/F) and a short archetype describing their life
context. Their *stable traits* are expressed on 0..1 scales chosen to map onto
established individual-difference constructs:

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
