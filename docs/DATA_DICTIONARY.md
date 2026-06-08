# Data dictionary

All tables are written to `outputs/data/` by `python run_all.py`. Latent states
are the simulator's ground truth (a real study cannot observe them); the
`*_obs` columns are the noisy self-logged versions the deployed model actually
uses.

> At cohort scale the full minute log is ~0.9M rows (~380 MB), so `run_all.py`
> saves a 20k-row **`minutes_sample.csv`** for inspection (and git-ignores the
> full file). The columns below describe both.

## `minutes_sample.csv` — one row per waking minute (sample of ~0.9M)

**Identifiers & context**
| Column | Meaning |
|---|---|
| `pid` | person id (P01–P08), no name |
| `sex` | M / F (no direct causal role; see THEORY.md) |
| `day`, `weekday`, `is_weekend` | day index 0–27, day-of-week 0–6, weekend flag |
| `is_event` | 1 if a disruption ("travel/sick/off") day |
| `clock_min`, `hour`, `minutes_awake` | minute-of-day, hour, minutes since waking |
| `activity`, `task_type` | current activity label and category |
| `difficulty`, `intrinsic`, `aversive` | task challenge / intrinsic interest / aversiveness (0–1) |
| `location`, `social` | where, and social context |
| `focus`, `is_meal`, `is_break` | flags for focus-demanding work / meal / break |
| `deadline`, `open_tasks` | deadline pressure (0–1), count of open tasks |
| `phone_in_reach` | 1 if phone within reach this block |
| `notif` | notifications arriving this minute |
| `time_on_task`, `vigilance` | minutes continuously on task; capped time-on-task (hours) |

**Latent ground-truth states (0–1)**
| Column | Meaning |
|---|---|
| `boredom`, `stress`, `hunger`, `mood`, `energy` | internal states |
| `fatigue`, `alertness`, `sleep_pressure`, `circadian` | sleep/circadian system |
| `focus_reserve` | momentary self-control capacity |
| `urge`, `urge_eff` | phone-checking urge; `urge_eff` gated by phone availability |

**Self-logged observations (model inputs)**
| Column | Meaning |
|---|---|
| `boredom_obs`, `stress_obs`, `energy_obs`, `hunger_obs`, `alertness_obs` | noisy, Likert-discretised self-reports |

**Labels**
| Column | Meaning |
|---|---|
| `slip_onset` | 1 if an attention lapse begins this minute (the prediction target) |
| `in_slip` | 1 if this minute is inside an ongoing lapse (excluded from the at-risk set) |
| `slip_channel` | phone / mind_wandering / task_switch / snack / social / none |

## `episodes.csv` — one row per slip (~34k rows; git-ignored, regenerable)
`pid, sex, day, clock_min, hour, activity, task_type, channel, duration, p_slip`
plus a snapshot of every internal state and the context at the moment of onset
(`boredom, fatigue, stress, hunger, mood, energy, focus_reserve, urge_eff,
intrinsic, aversive, difficulty, vigilance, deadline, notif, phone_in_reach,
time_on_task, minutes_awake`). Used for channel/cause attribution and survival.

## `personas.csv` — one row per person
`pid, sex, archetype, trait_self_control, neuroticism, conscientiousness,
habit_strength, chronotype, caffeine_use, baseline_sleep_h, notif_rate,
phone_away_policy, intercept`.

## Analysis outputs
| File | Contents |
|---|---|
| `recovery_table.csv` | true vs recovered hazard coefficients and ranks |
| `hazard_ratios.csv` | discrete-time hazard ratios (per +1 SD) with 95% CIs |
| `odds_ratios.csv` | logistic odds ratios on self-logged features (per SD) |
| `fingerprints_self_logged.csv` | per-person counterfactual cause shares (model) |
| `fingerprints_ground_truth.csv` | per-person cause shares (true generator) |
| `shap_importance.csv` | global mean\|SHAP\| per feature |
| `learning_curve.csv` | cold-start ROC vs number of training people |
| `summary.json` | headline metrics in one place (both regimes, calibration, recovery) |
| `self_log_template.csv` | blank template for logging your own day (see schema.py) |
| `model/timeslip_model.joblib` | persisted production model + calibrator + feature schema |
