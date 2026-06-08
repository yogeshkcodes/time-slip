"""
Central configuration for the *Time Slip* project.

This file is the single source of truth shared by the simulator, the feature
builder, the models and the explanation/validation code. Keeping the
ground-truth causal coefficients here (rather than buried in the simulator) is
deliberate: the headline scientific claim of this project is that an
explainable-ML + counterfactual pipeline can *recover* the causal drivers of
attention lapses. To test that claim honestly we must be able to compare what
the model recovers against the true generative coefficients, so both the
simulator and the validator import them from the same place.

Every coefficient is annotated with the cognitive-science construct it stands
in for and a pointer (see docs/THEORY.md for full citations).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
GLOBAL_SEED = 20260608          # fixed so a paper reviewer can reproduce exactly

# --------------------------------------------------------------------------- #
# Simulation scope
# --------------------------------------------------------------------------- #
N_DAYS = 14                     # days simulated per person
MINUTES_PER_STEP = 1            # temporal resolution of the log (1 minute)

# --------------------------------------------------------------------------- #
# The latent internal states tracked minute-by-minute.
# These are the "true" psychological/physiological variables. A real person
# cannot measure them precisely; the simulator also emits a noisy, discretised
# "self-logged" version of a subset (see schema.py / OBSERVED_STATES).
# --------------------------------------------------------------------------- #
LATENT_STATES = [
    "sleep_pressure",   # homeostatic sleep drive   (Borbely two-process: Process S)
    "circadian",        # circadian alerting signal  (Process C; chronotype)
    "alertness",        # net alertness = f(circadian, sleep_pressure)
    "fatigue",          # 1 - alertness (convenience)
    "energy",           # metabolic/cognitive energy (depleted by effort, fed by meals)
    "hunger",           # time-since-meal driven
    "boredom",          # under-stimulation / attentional disengagement (Eastwood)
    "stress",           # anxiety/arousal load (attentional control theory, Eysenck)
    "mood",             # affective valence
    "focus_reserve",    # momentary self-control capacity (Tangney trait * dynamic)
    "urge_to_check",    # latent phone-checking urge (habit + notifications + boredom)
]

# Subset that a human could plausibly self-report each block (1-5 Likert, etc.)
# The "realistic" model is restricted to these; the "oracle" model may use all.
OBSERVED_STATES = ["boredom", "stress", "energy", "hunger", "alertness"]

# --------------------------------------------------------------------------- #
# GROUND-TRUTH causal coefficients for the per-minute slip hazard.
#
#   logit P(slip onset this minute) = INTERCEPT
#       + B["boredom"]      * boredom
#       + B["fatigue"]      * fatigue
#       + B["stress"]       * stress
#       + B["aversive"]     * task_aversiveness
#       + B["vigilance"]    * time_on_task_norm        (vigilance decrement)
#       + B["hunger"]       * hunger
#       + B["urge"]         * urge_to_check            (phone pull)
#       + B["low_intrinsic"]* (1 - intrinsic_motivation)
#       + B["low_mood"]     * (1 - mood)
#       - B["self_control"] * focus_reserve            (protective: note minus)
#       + persona_intercept                            (random effect)
#
# Signs: everything that *erodes* attention is positive; focus_reserve is the
# only protective term and enters with a minus sign.
# These are on the *per-minute logit* scale, so the absolute numbers are small;
# what is scientifically meaningful (and what the recovery validation in
# explain.py checks) is the RATIO and SIGN between drivers. They were tuned so a
# typical knowledge-worker day yields a realistic number of lapses (on the order
# of one self-interruption every several minutes during low-engagement work, far
# fewer during flow), consistent with field studies of self-interruption.
# --------------------------------------------------------------------------- #
HAZARD = {
    "intercept":     -5.90,  # low baseline per-minute hazard
    "boredom":        2.016,  # strongest internal driver (Eastwood; mind-wandering)
    "fatigue":        1.128,  # sleep/circadian (two-process model)
    "stress":         0.984,  # attentional control theory
    "aversive":       1.416,  # task aversiveness (Temporal Motivation Theory, Steel)
    "vigilance":      0.816,  # time-on-task decrement (Mackworth)
    "hunger":         0.600,  # metabolic
    "urge":           2.184,  # phone urge: habit + notifications + cue (Oulasvirta; Ward)
    "low_intrinsic":  1.272,  # lack of intrinsic motivation (Self-Determination Theory)
    "low_mood":       0.456,  # negative affect
    "self_control":   1.584,  # PROTECTIVE (subtracted): trait * focus_reserve (Tangney)
}
VIGILANCE_CAP = 2.0          # time_on_task (hours) is capped here in the hazard
SLIP_HORIZON_MIN = 10        # deployable target: "will a slip occur in the next N min?"

# Mapping from each hazard term to the human-readable "cause" it represents and
# the feature(s) the model should manipulate in counterfactual analysis.
CAUSE_LABELS = {
    "boredom":       "Boredom / under-stimulation",
    "fatigue":       "Fatigue (sleep + circadian)",
    "stress":        "Stress / anxiety",
    "aversive":      "Task aversiveness (procrastination)",
    "vigilance":     "Time-on-task (vigilance decrement)",
    "hunger":        "Hunger",
    "urge":          "Phone pull (habit + notifications)",
    "low_intrinsic": "Low intrinsic motivation",
    "low_mood":      "Low mood / negative affect",
    "self_control":  "Depleted self-control capacity",
}

# --------------------------------------------------------------------------- #
# Slip channels: the *form* the lapse takes once it occurs. Each has a linear
# score; the realised channel is drawn from a softmax over these scores using
# the internal state at the moment of the slip.
# --------------------------------------------------------------------------- #
SLIP_CHANNELS = ["phone", "mind_wandering", "task_switch", "snack", "social"]

CHANNEL_WEIGHTS = {
    # channel:        {state_or_context: weight}
    "phone":          {"urge_to_check": 1.7, "boredom": 1.0, "habit_strength": 0.7},
    "mind_wandering": {"fatigue": 1.5, "low_intrinsic": 1.3, "difficulty": 0.9},
    "task_switch":    {"stress": 1.4, "deadline_pressure": 0.9, "open_tasks": 0.5},
    "snack":          {"hunger": 1.7, "low_energy": 0.9},
    "social":         {"social_present": 1.6, "boredom": 0.9},
}

# Base mean duration (minutes) of a slip by channel, before boredom/habit
# lengthening. Phone slips are heavy-tailed -> the literal "time slip" where a
# 2-minute check becomes 25 minutes (variable-ratio reinforcement).
CHANNEL_DURATION = {
    "phone":          {"base": 4.0,  "tail": 2.2},   # lognormal-ish, long tail
    "mind_wandering": {"base": 2.5,  "tail": 1.3},
    "task_switch":    {"base": 6.0,  "tail": 1.5},
    "snack":          {"base": 5.0,  "tail": 1.0},
    "social":         {"base": 7.0,  "tail": 1.6},
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
TEST_DAYS_FRACTION = 0.30       # hold out the last ~30% of days per person (no leakage)
RNG_NOISE_OBSERVED = 0.09       # std of measurement noise added to self-logged states

# Features the realistic model is allowed to use (self-loggable only).
# Trait + context + temporal features are listed in features.py.
