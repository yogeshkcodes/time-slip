"""
The *Time Slip* structural causal simulator.

Every person's day is generated minute-by-minute. Eleven latent
psychological/physiological states evolve under simple, literature-motivated
dynamics (see docs/THEORY.md). At each waking minute we evaluate a *slip
hazard* -- the instantaneous probability that attention disengages from the
current activity -- as a logistic function of those states and the task context,
using the ground-truth coefficients in ``config.HAZARD``. When a slip fires we
draw its *channel* (phone / mind-wandering / task-switch / snack / social) from
a softmax over channel-specific scores and a *duration* (phone slips are
heavy-tailed: the literal "time slip" where a quick check becomes 25 minutes).

The simulator emits two tables:
  * ``minutes``  -- one row per waking minute, with full latent state, a noisy
                    self-logged subset, the task context, and the slip label.
  * ``episodes`` -- one row per slip event, with a snapshot of the state at
                    onset (used for channel/cause attribution and survival).

Because the data-generating process is fully known, downstream code can check
whether the ML+counterfactual pipeline recovers the true causal structure.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from . import config as C
from .personas import Persona, build_personas


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- #
# Activity prototypes. Each entry gives default duration (min) and the
# attribute *centres* used when a block of that type is placed. Values are
# jittered per occurrence. ``focus`` flags cognitively demanding on-task work
# (where the vigilance decrement and self-control cost are largest).
# --------------------------------------------------------------------------- #
PROTO: Dict[str, dict] = {
    "morning_routine": dict(task="routine", dur=40, diff=0.20, intr=0.45, aver=0.25, loc="home",     soc="alone",      focus=0, meal=0, brk=1),
    "breakfast":       dict(task="meal",    dur=20, diff=0.10, intr=0.55, aver=0.10, loc="home",     soc="family",     focus=0, meal=1, brk=1),
    "commute":         dict(task="commute", dur=35, diff=0.10, intr=0.20, aver=0.55, loc="transit",  soc="alone",      focus=0, meal=0, brk=0),
    "deep_work":       dict(task="deep_work", dur=90, diff=0.80, intr=0.60, aver=0.40, loc="office",  soc="alone",      focus=1, meal=0, brk=0),
    "creative_work":   dict(task="deep_work", dur=85, diff=0.70, intr=0.82, aver=0.30, loc="home",    soc="alone",      focus=1, meal=0, brk=0),
    "study":           dict(task="study",   dur=80, diff=0.72, intr=0.62, aver=0.55, loc="home",      soc="alone",      focus=1, meal=0, brk=0),
    "email_admin":     dict(task="admin",   dur=30, diff=0.30, intr=0.20, aver=0.72, loc="office",    soc="alone",      focus=1, meal=0, brk=0),
    "meeting":         dict(task="meeting", dur=45, diff=0.50, intr=0.40, aver=0.50, loc="office",    soc="colleagues", focus=1, meal=0, brk=0),
    "shift_work":      dict(task="shift_work", dur=120, diff=0.55, intr=0.35, aver=0.55, loc="office", soc="colleagues", focus=1, meal=0, brk=0),
    "lunch":           dict(task="meal",    dur=40, diff=0.10, intr=0.60, aver=0.10, loc="cafe",      soc="colleagues", focus=0, meal=1, brk=1),
    "dinner":          dict(task="meal",    dur=45, diff=0.10, intr=0.65, aver=0.10, loc="home",      soc="family",     focus=0, meal=1, brk=1),
    "short_break":     dict(task="break",   dur=12, diff=0.05, intr=0.55, aver=0.10, loc="office",    soc="colleagues", focus=0, meal=0, brk=1),
    "errands":         dict(task="errands", dur=45, diff=0.20, intr=0.30, aver=0.55, loc="outdoors",  soc="alone",      focus=0, meal=0, brk=0),
    "chores":          dict(task="chores",  dur=40, diff=0.20, intr=0.20, aver=0.62, loc="home",      soc="alone",      focus=0, meal=0, brk=0),
    "childcare":       dict(task="childcare", dur=60, diff=0.40, intr=0.62, aver=0.35, loc="home",    soc="family",     focus=0, meal=0, brk=0),
    "exercise":        dict(task="exercise", dur=55, diff=0.45, intr=0.70, aver=0.30, loc="gym",      soc="alone",      focus=0, meal=0, brk=1),
    "leisure":         dict(task="leisure", dur=50, diff=0.10, intr=0.80, aver=0.10, loc="home",      soc="alone",      focus=0, meal=0, brk=1),
    "social":          dict(task="social",  dur=60, diff=0.20, intr=0.80, aver=0.15, loc="home",      soc="friends",    focus=0, meal=0, brk=1),
    "wind_down":       dict(task="wind_down", dur=40, diff=0.08, intr=0.70, aver=0.10, loc="home",     soc="family",     focus=0, meal=0, brk=1),
}


def _weekday_agenda(p: Persona) -> List[str]:
    """Ordered list of prototype names for a working day, specialised per role."""
    pid = p.pid
    if pid == "P03":      # graduate student
        return ["morning_routine", "breakfast", "commute", "study", "short_break",
                "meeting", "study", "lunch", "study", "short_break", "email_admin",
                "errands", "exercise", "dinner", "study", "leisure", "wind_down"]
    if pid == "P04":      # sales manager: meeting/notification heavy
        return ["morning_routine", "breakfast", "commute", "email_admin", "meeting",
                "meeting", "deep_work", "lunch", "meeting", "email_admin", "meeting",
                "deep_work", "commute", "exercise", "dinner", "social", "wind_down"]
    if pid == "P05":      # freelance creative, works from home
        return ["morning_routine", "breakfast", "leisure", "creative_work", "short_break",
                "creative_work", "lunch", "errands", "creative_work", "short_break",
                "email_admin", "exercise", "dinner", "leisure", "social", "wind_down"]
    if pid == "P06":      # rotating shift worker
        return ["morning_routine", "breakfast", "chores", "errands", "leisure", "lunch",
                "short_break", "exercise", "dinner", "commute", "shift_work",
                "short_break", "shift_work", "wind_down"]
    if pid == "P07":      # parent, part-time, fragmented
        return ["morning_routine", "breakfast", "childcare", "chores", "commute",
                "email_admin", "deep_work", "lunch", "deep_work", "commute", "childcare",
                "chores", "dinner", "childcare", "leisure", "wind_down"]
    if pid == "P08":      # student athlete
        return ["morning_routine", "breakfast", "study", "exercise", "short_break",
                "study", "lunch", "study", "meeting", "exercise", "dinner", "leisure",
                "wind_down"]
    # default knowledge worker (P01, P02)
    return ["morning_routine", "breakfast", "commute", "deep_work", "short_break",
            "email_admin", "meeting", "deep_work", "lunch", "meeting", "deep_work",
            "short_break", "email_admin", "commute", "exercise", "dinner", "leisure",
            "social", "wind_down"]


def _weekend_agenda(p: Persona) -> List[str]:
    if p.pid == "P07":    # parent: weekend is mostly childcare + chores
        return ["morning_routine", "breakfast", "childcare", "chores", "errands",
                "lunch", "childcare", "exercise", "leisure", "dinner", "social", "wind_down"]
    if p.pid == "P06":    # shift worker may still work a weekend shift
        return ["morning_routine", "breakfast", "chores", "leisure", "lunch", "exercise",
                "errands", "dinner", "shift_work", "wind_down"]
    return ["morning_routine", "breakfast", "leisure", "errands", "exercise", "lunch",
            "chores", "leisure", "social", "dinner", "leisure", "wind_down"]


@dataclass
class DayPlan:
    wake_h: float
    bed_h: float
    is_weekend: bool
    deadline: float                 # day-level deadline pressure 0..1
    blocks: List[dict]              # each with absolute clock-minute start/end + attrs


def _build_day_plan(p: Persona, day_idx: int, rng: np.random.Generator,
                    crunch_days: set) -> DayPlan:
    weekday = day_idx % 7
    is_weekend = weekday >= 5
    wake_h = 6.5 + p.chronotype * 2.5 + (0.9 if is_weekend else 0.0) + rng.normal(0, 0.25)
    # shift worker sleeps in on shift days
    if p.pid == "P06" and not is_weekend:
        wake_h += 2.5
    bed_h = 22.0 + p.chronotype * 2.0 + (0.6 if is_weekend else 0.0) + rng.normal(0, 0.3)
    if p.pid == "P06" and not is_weekend:
        bed_h = 26.5 + rng.normal(0, 0.4)        # past midnight after a night shift

    deadline = 0.15 + rng.uniform(0, 0.15)
    if day_idx in crunch_days and not is_weekend:
        deadline = 0.7 + rng.uniform(0, 0.25)

    agenda = _weekend_agenda(p) if is_weekend else _weekday_agenda(p)

    cursor = wake_h * 60.0
    end_min = bed_h * 60.0
    blocks: List[dict] = []
    for name in agenda:
        proto = PROTO[name]
        dur = max(6, proto["dur"] * rng.normal(1.0, 0.18))
        diff = clip01(proto["diff"] + rng.normal(0, 0.07))
        intr = clip01(proto["intr"] + rng.normal(0, 0.08))
        aver = clip01(proto["aver"] + rng.normal(0, 0.07))
        # deadline pressure inflates aversiveness/difficulty of work tasks
        if proto["focus"]:
            aver = clip01(aver + 0.25 * deadline)
            diff = clip01(diff + 0.15 * deadline)
        # phone self-binding only attempted on focus work
        phone_away = bool(proto["focus"] and rng.random() < p.phone_away_policy)
        open_tasks = int(round(1 + 4 * deadline)) if proto["focus"] else 0
        start = cursor
        finish = min(cursor + dur, end_min)
        if finish - start < 4:
            break
        blocks.append(dict(
            activity=name, task_type=proto["task"], start=start, end=finish,
            difficulty=diff, intrinsic=intr, aversive=aver, location=proto["loc"],
            social=proto["soc"], focus=proto["focus"], meal=proto["meal"],
            brk=proto["brk"], deadline=deadline, open_tasks=open_tasks,
            phone_in_reach=0 if phone_away else 1,
        ))
        cursor = finish
        if cursor >= end_min:
            break
    # pad the evening with leisure if the agenda ran short
    while cursor < end_min - 15:
        proto = PROTO["leisure"]
        finish = min(cursor + proto["dur"], end_min)
        blocks.append(dict(
            activity="leisure", task_type="leisure", start=cursor, end=finish,
            difficulty=0.1, intrinsic=0.8, aversive=0.1, location="home",
            social="alone", focus=0, meal=0, brk=1, deadline=deadline,
            open_tasks=0, phone_in_reach=1,
        ))
        cursor = finish

    return DayPlan(wake_h=wake_h, bed_h=bed_h, is_weekend=is_weekend,
                   deadline=deadline, blocks=blocks)


def _notif_rate_per_min(p: Persona, hour: float, social: str) -> float:
    """Poisson rate of notifications this minute (job + time-of-day + context)."""
    tod = 1.0 if 9 <= hour <= 19 else (0.45 if 7 <= hour < 9 or 19 < hour <= 22 else 0.15)
    soc = 1.25 if social in ("family", "friends") else 1.0
    return (p.notif_rate / 60.0) * tod * soc


def _choose_channel(state: dict, p: Persona, rng: np.random.Generator) -> str:
    feats = {
        "urge_to_check": state["urge_eff"],
        "boredom": state["boredom"],
        "habit_strength": p.habit_strength,
        "fatigue": state["fatigue"],
        "low_intrinsic": 1.0 - state["intrinsic"],
        "difficulty": state["difficulty"],
        "stress": state["stress"],
        "deadline_pressure": state["deadline"],
        "open_tasks": state["open_tasks"] / 5.0,
        "hunger": state["hunger"],
        "low_energy": 1.0 - state["energy"],
        "social_present": 1.0 if state["social"] != "alone" else 0.0,
    }
    scores = []
    for ch in C.SLIP_CHANNELS:
        s = sum(w * feats.get(k, 0.0) for k, w in C.CHANNEL_WEIGHTS[ch].items())
        if ch == "phone" and state["phone_in_reach"] == 0:
            s -= 2.5                     # phone away -> phone slip very unlikely
        scores.append(s)
    scores = np.array(scores)
    ex = np.exp(scores - scores.max())
    probs = ex / ex.sum()
    return str(rng.choice(C.SLIP_CHANNELS, p=probs))


def _slip_duration(channel: str, state: dict, p: Persona, rng: np.random.Generator) -> int:
    d = C.CHANNEL_DURATION[channel]
    base = d["base"]
    if channel == "phone":               # rabbit-hole lengthening (variable reward)
        base *= (1.0 + 1.6 * state["boredom"]) * (1.0 + 0.8 * p.habit_strength)
    elif channel == "social":
        base *= (1.0 + 0.8 * state["boredom"])
    dur = rng.lognormal(mean=np.log(max(1.0, base)), sigma=0.45)
    return int(max(1, min(75, round(dur))))


# --------------------------------------------------------------------------- #
# main per-person simulation
# --------------------------------------------------------------------------- #
def simulate_person(p: Persona, rng: np.random.Generator
                    ) -> Tuple[List[dict], List[dict]]:
    minute_rows: List[dict] = []
    episode_rows: List[dict] = []

    # two crunch (high-deadline) weekdays in the fortnight
    crunch_days = set(rng.choice([d for d in range(C.N_DAYS) if d % 7 < 5],
                                 size=2, replace=False).tolist())

    sleep_debt = 0.0                      # 0..1 accumulated
    prev_bed_h = None

    for day in range(C.N_DAYS):
        plan = _build_day_plan(p, day, rng, crunch_days)

        # ---- overnight recovery -> starting states ----
        if prev_bed_h is not None:
            got = (24.0 - (prev_bed_h % 24)) + plan.wake_h if prev_bed_h % 24 > 12 \
                  else plan.wake_h - (prev_bed_h % 24)
            got = max(3.0, min(11.0, got))
            shortfall = max(0.0, p.baseline_sleep_h - got)
            sleep_debt = clip01(0.55 * sleep_debt + 0.18 * shortfall)
        prev_bed_h = plan.bed_h

        # shift worker carries chronic circadian-misalignment debt
        extra = 0.18 if (p.pid == "P06" and not plan.is_weekend) else 0.0
        sleep_pressure = clip01(0.12 + 0.55 * sleep_debt + extra + rng.normal(0, 0.03))
        energy = clip01(0.88 - 0.35 * sleep_pressure)
        hunger = clip01(0.35 + rng.normal(0, 0.05))
        boredom = clip01(0.12 + rng.normal(0, 0.04))
        stress = clip01(0.18 + 0.25 * p.neuroticism + 0.35 * plan.deadline)
        mood = clip01(0.68 - 0.3 * stress + 0.2 * (1 - sleep_pressure))
        focus_capacity = clip01(0.35 + 0.6 * p.trait_self_control)
        focus_reserve = clip01(focus_capacity * (1 - 0.3 * sleep_pressure))
        urge = clip01(0.15 * p.habit_strength)
        recent_caffeine = 0.0

        # ---- flatten the day into per-minute context arrays ----
        ctx: List[dict] = []
        for b in plan.blocks:
            n = int(round(b["end"] - b["start"]))
            for k in range(n):
                ctx.append(dict(block=b, clock_min=b["start"] + k, t_in_block=k))
        T = len(ctx)
        if T == 0:
            continue

        wake_min = plan.wake_h * 60.0
        time_on_task = 0
        slip_remaining = 0
        slip_channel = "none"
        last_meal_min = wake_min
        minutes_since_meal = 0.0

        for t in range(T):
            cell = ctx[t]
            b = cell["block"]
            clock_min = cell["clock_min"]
            hour = clock_min / 60.0
            minutes_awake = clock_min - wake_min

            difficulty = b["difficulty"]
            intrinsic = b["intrinsic"]
            aversive = b["aversive"]
            focus = b["focus"]
            is_meal = b["meal"]
            is_break = b["brk"]
            deadline = b["deadline"]

            # -------- circadian + fatigue (two-process model) --------
            acro = 13.0 + 5.0 * p.chronotype
            circ = 0.5 + 0.32 * np.cos(2 * np.pi * (hour - acro) / 24.0)
            dip = 0.13 * np.exp(-((hour - 14.5) ** 2) / (2 * 1.5 ** 2))
            circadian = clip01(circ - dip)
            # caffeine intake events
            if p.caffeine_use > 0.45 and (abs(minutes_awake - 90) < 1 or abs(hour - 13.5) < 0.02):
                recent_caffeine = min(1.0, recent_caffeine + 0.6 * p.caffeine_use)
            recent_caffeine *= 0.992                                   # ~3h decay
            sleep_pressure = clip01(sleep_pressure + 0.00060)          # Process S rise
            alertness = clip01(0.15 + 0.85 * circadian - 0.65 * sleep_pressure
                               + 0.25 * recent_caffeine)
            fatigue = 1.0 - alertness

            # -------- hunger / energy --------
            minutes_since_meal += 1
            if is_meal:
                hunger = clip01(hunger - 0.06)
                energy = clip01(energy + 0.004)
                last_meal_min = clock_min
                minutes_since_meal = 0.0
            else:
                hunger = clip01(hunger + 0.0016)
            if focus:
                energy = clip01(energy - 0.0016 * (0.3 + difficulty) - 0.0002)
            elif is_break or b["task_type"] == "leisure":
                energy = clip01(energy + 0.0018)

            # -------- boredom --------
            understim = (1 - intrinsic) * (0.55 + 0.45 * (1 - difficulty))
            overload = 0.4 * max(0.0, difficulty - 0.5) * (1 - intrinsic)
            target_bore = clip01(understim + overload)
            boredom = clip01(boredom + 0.020 * (target_bore - boredom)
                             + 0.0009 * (time_on_task / 10.0) * (0.5 + (1 - intrinsic)))
            if is_break or is_meal:
                boredom = clip01(boredom - 0.02)

            # -------- stress --------
            target_stress = clip01(0.16 + 0.5 * deadline + 0.4 * difficulty * focus
                                   + 0.2 * p.neuroticism)
            stress = clip01(stress + 0.03 * (target_stress - stress)
                            + rng.normal(0, 0.01 + 0.02 * p.neuroticism))
            if b["task_type"] == "meeting":
                stress = clip01(stress + 0.002)

            # -------- mood --------
            social_pos = 0.12 if b["social"] in ("family", "friends") else 0.0
            target_mood = clip01(0.7 - 0.4 * stress + 0.2 * (1 - fatigue) + social_pos)
            mood = clip01(mood + 0.02 * (target_mood - mood))

            # -------- focus reserve (self-control capacity) --------
            cap = clip01((0.35 + 0.6 * p.trait_self_control) * (1 - 0.3 * fatigue))
            if focus:
                focus_reserve = clip01(focus_reserve
                                       - 0.0016 * (0.4 + aversive) * (0.5 + urge))
            else:
                focus_reserve = clip01(focus_reserve + 0.004 * (cap - focus_reserve))

            # -------- notifications + phone urge --------
            lam = _notif_rate_per_min(p, hour, b["social"])
            notif = int(rng.poisson(lam))
            target_urge = p.habit_strength * (0.2 + 0.5 * boredom)
            urge = clip01(urge + 0.04 * (target_urge - urge))
            if notif > 0:
                urge = clip01(urge + 0.40 * notif * rng.uniform(0.6, 1.0))
            urge_eff = urge * (1.0 if b["phone_in_reach"] else 0.35)

            # -------- assemble state snapshot --------
            state = dict(
                fatigue=fatigue, boredom=boredom, stress=stress, hunger=hunger,
                mood=mood, energy=energy, focus_reserve=focus_reserve,
                alertness=alertness, sleep_pressure=sleep_pressure, circadian=circadian,
                urge=urge, urge_eff=urge_eff, intrinsic=intrinsic, difficulty=difficulty,
                deadline=deadline, open_tasks=b["open_tasks"], social=b["social"],
                phone_in_reach=b["phone_in_reach"],
            )

            # -------- slip hazard --------
            in_slip = 1 if slip_remaining > 0 else 0
            slip_onset = 0
            vig = min(C.VIGILANCE_CAP, (time_on_task / 60.0)) if focus else \
                  min(0.5, time_on_task / 120.0)
            if in_slip == 0:
                H = C.HAZARD
                logit = (H["intercept"] + p.intercept
                         + H["boredom"] * boredom
                         + H["fatigue"] * fatigue
                         + H["stress"] * stress
                         + H["aversive"] * aversive
                         + H["vigilance"] * vig
                         + H["hunger"] * hunger
                         + H["urge"] * urge_eff
                         + H["low_intrinsic"] * (1 - intrinsic)
                         + H["low_mood"] * (1 - mood)
                         - H["self_control"] * focus_reserve)
                p_slip = sigmoid(logit)
                if rng.random() < p_slip:
                    slip_onset = 1
                    channel = _choose_channel(state, p, rng)
                    dur = _slip_duration(channel, state, p, rng)
                    slip_remaining = dur
                    slip_channel = channel
                    episode_rows.append(dict(
                        pid=p.pid, sex=p.sex, day=day, clock_min=clock_min,
                        hour=hour, activity=b["activity"], task_type=b["task_type"],
                        channel=channel, duration=dur, p_slip=p_slip,
                        boredom=boredom, fatigue=fatigue, stress=stress, hunger=hunger,
                        mood=mood, energy=energy, focus_reserve=focus_reserve,
                        urge_eff=urge_eff, intrinsic=intrinsic, aversive=aversive,
                        difficulty=difficulty, vigilance=vig, deadline=deadline,
                        notif=notif, phone_in_reach=b["phone_in_reach"],
                        time_on_task=time_on_task, minutes_awake=minutes_awake,
                    ))
            else:
                channel = slip_channel
                # in-slip effects: relief + maybe snacking
                boredom = clip01(boredom - 0.03)
                if channel == "snack":
                    hunger = clip01(hunger - 0.02)
                slip_remaining -= 1
                if slip_remaining == 0:
                    # returning to task: guilt bump for conscientious people on work
                    if focus and p.conscientiousness > 0.55:
                        stress = clip01(stress + 0.05 * p.conscientiousness)
                    urge = clip01(urge * 0.4)
                    time_on_task = 0

            # measurement model: noisy self-logged subset (Likert-ish)
            def obs(v):
                n = clip01(v + rng.normal(0, C.RNG_NOISE_OBSERVED))
                return round(n * 4) / 4.0           # 0,.25,.5,.75,1 ~ 1..5 Likert

            row = dict(
                pid=p.pid, sex=p.sex, day=day, weekday=day % 7,
                is_weekend=int(plan.is_weekend), clock_min=clock_min, hour=hour,
                minutes_awake=minutes_awake, activity=b["activity"],
                task_type=b["task_type"], difficulty=difficulty, intrinsic=intrinsic,
                aversive=aversive, location=b["location"], social=b["social"],
                focus=focus, is_meal=is_meal, is_break=is_break, deadline=deadline,
                open_tasks=b["open_tasks"], phone_in_reach=b["phone_in_reach"],
                notif=notif, time_on_task=time_on_task, vigilance=vig,
                # latent ground-truth states
                boredom=boredom, fatigue=fatigue, stress=stress, hunger=hunger,
                mood=mood, energy=energy, focus_reserve=focus_reserve,
                alertness=alertness, sleep_pressure=sleep_pressure, circadian=circadian,
                urge=urge, urge_eff=urge_eff,
                # noisy self-logged versions
                boredom_obs=obs(boredom), stress_obs=obs(stress), energy_obs=obs(energy),
                hunger_obs=obs(hunger), alertness_obs=obs(alertness),
                # labels
                slip_onset=slip_onset, in_slip=in_slip, slip_channel=slip_channel,
            )
            minute_rows.append(row)

            # advance time-on-task (resets on slip end handled above, and on breaks)
            if in_slip:
                pass
            elif is_break or is_meal:
                time_on_task = 0
            else:
                time_on_task += 1
            if cell["t_in_block"] == 0:        # new block resets vigilance clock
                time_on_task = 0

    return minute_rows, episode_rows


def simulate_all(seed: int = C.GLOBAL_SEED
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    personas = build_personas(rng)
    all_minutes: List[dict] = []
    all_eps: List[dict] = []
    for p in personas:
        # give each person an independent stream for reproducibility
        prng = np.random.default_rng(rng.integers(0, 2**63 - 1))
        m, e = simulate_person(p, prng)
        all_minutes.extend(m)
        all_eps.extend(e)
    minutes = pd.DataFrame(all_minutes)
    episodes = pd.DataFrame(all_eps)
    persona_tbl = pd.DataFrame([pp.to_row() for pp in personas])
    return minutes, episodes, persona_tbl


if __name__ == "__main__":
    m, e, pt = simulate_all()
    print("minutes:", m.shape, "| episodes:", e.shape, "| personas:", pt.shape)
    print("overall slip-onset rate per waking minute: "
          f"{m['slip_onset'].mean():.4f}")
    print(m.groupby("pid")["slip_onset"].sum())
    print(e["channel"].value_counts())
