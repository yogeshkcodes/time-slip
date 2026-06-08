# Time Slip

> 👉 **New here? Read [START_HERE.md](START_HERE.md) first** — the whole project
> explained in plain English (no jargon): what it is, how to run it, what the
> results mean, and how to explain it to others.

**Finding *where*, *when*, and *why* attention slips in a daily routine — and
proving the "why" is real.**

Most work on everyday distraction stops at correlation ("people check their phone
when bored") or at a black-box predictor. Time Slip goes further: it recovers,
*per person*, the **causes** of attention lapses and **validates** that recovery
against known ground truth — the precondition for trusting such a method on real
self-tracked data.

The trick: we can't get ground-truth causes from real life (when someone grabs
their phone, the true cause is unobserved). So we **simulate** detailed daily
routines from an explicit *structural causal model* grounded in cognitive
science, then show an explainable-ML + counterfactual pipeline — using only
**self-loggable** features — recovers the generator's true causal structure.

> Attention is becoming the scarce currency. Time Slip is about measuring its
> exchange rate, person by person.

---

## What it does

1. **Simulates** 8 contrasting people (no names; sex M/F) over 14 days at
   1-minute resolution. Eleven internal states (boredom, stress, fatigue, hunger,
   self-control reserve, phone urge, sleep/circadian, …) evolve under
   literature-grounded dynamics and drive a per-minute **slip hazard**. Lapses
   take one of five channels — phone, mind-wandering, task-switch, snack, social —
   with heavy-tailed durations (the literal "time slip": a 2-min check becomes 25).

2. **Predicts** a slip in the next 10 minutes from self-loggable features only
   (held-out **ROC-AUC ≈ 0.71**, vs **0.62** for a notifications-only baseline —
   internal states, not just pings, carry the signal).

3. **Recovers the causes** and checks itself:
   - True vs recovered hazard coefficients: **Spearman 0.92**, 90% sign agreement.
   - Per-person counterfactual "slip fingerprints" vs ground truth: **Spearman 0.85**.

4. **Explains, per person**, what share of *their* lapses is attributable to each
   cause, when their focus tends to break (vigilance decrement, time-of-day), and
   what to do about it.

## The conceptual punchline

Two questions, two different answers — and the project keeps them separate:

| Question | Method | Dominant answer |
|---|---|---|
| What raises your **baseline lapse hazard**? | hazard coefficients / ratios | boredom, depleted self-control |
| What is most **reducible at the moments you slip**? | counterfactual fingerprint | the phone pull, task aversiveness |

Distraction isn't one thing. The right lever is personal — see
`outputs/figures/per_person_fingerprint.png`.

---

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py            # ~25s; writes data, figures, reports to ./outputs/
python analyze_me.py         # analyse a self-logged routine (example if no file)
```

Then read **`outputs/reports/findings.md`** and browse `outputs/figures/`.

Run any stage standalone (each has a `__main__` demo):
```bash
python -m timeslip.simulate     # generate + summarise the cohort
python -m timeslip.model        # train predictor + coefficient recovery
python -m timeslip.survival     # hazard ratios + attention-survival
python -m timeslip.explain      # counterfactual slip fingerprints
python -m timeslip.schema       # write a blank self-logging template
```

## Key figures (in `outputs/figures/`)
| File | Shows |
|---|---|
| `recovery_scatter.png` | recovered vs true causal coefficients (the validation) |
| `per_person_fingerprint.png` | each person's cause breakdown |
| `population_fingerprint.png` | model vs ground-truth cause shares |
| `hazard_ratios.png` | lapse hazard ratios during focused work |
| `vigilance_curve.png` | risk rising with time-on-task |
| `km_curve.png` | how long focus lasts before a slip |
| `shap_summary.png` | feature drivers of next-10-min risk |
| `model_performance.png` | ROC / PR / calibration vs baselines |
| `circadian_slips.png` | when slips happen across the day |
| `channels.png` | slip types and the state behind each |
| `day_timeline_P0*.png` | a representative day with state curves + slips |

## Use your own data — `analyze_me.py`

The deployed model uses only things you can log, so you can run the whole
analysis on **yourself**:

```bash
python -m timeslip.schema      # 1. creates outputs/data/self_log_template.csv
#                                2. fill it in: one row per ~15-30 min or per
#                                   activity block, including when you slipped
python analyze_me.py my_log.csv # 3. get your personal slip report + figures
```

No data yet? `python analyze_me.py` (no argument) builds a realistic **example
log** and analyses it, so you can see the exact output format first.

It produces two levels (into `outputs/me/`):
- **Level 1 — descriptive** (works as soon as you log your slips): *when* and
  *where* you slip, your channel mix, time lost, and which felt states are
  elevated in slip moments vs. focused moments. This is a faithful summary of
  *your* data — no model.
- **Level 2 — personal fingerprint** (once you have ≥30 rows / ≥6 slips): a
  logistic risk model fit on *your* rows with a counterfactual breakdown of what
  share of your slips each cause is responsible for, cross-validated.

See `docs/DATA_DICTIONARY.md` and `timeslip/schema.py` for the schema.

## Project structure

```
Time Slip/
├─ run_all.py                end-to-end pipeline (the simulation study)
├─ analyze_me.py             analyse YOUR own logged routine
├─ requirements.txt
├─ timeslip/
│  ├─ config.py              constructs, ground-truth coefficients, channels
│  ├─ personas.py            8 people: traits + sex (no names)
│  ├─ simulate.py            structural causal simulator (states → hazard → slips)
│  ├─ features.py            self-loggable vs latent feature sets; risk windows
│  ├─ model.py               risk model, baselines, coefficient recovery
│  ├─ survival.py            Kaplan–Meier + discrete-time hazard + vigilance
│  ├─ explain.py             SHAP + counterfactual fingerprints + validation
│  ├─ schema.py              self-logging schema for real data
│  ├─ realdata.py            analyse a real person's logged routine
│  └─ report.py              figures + per-person reports
├─ docs/
│  ├─ THEORY.md              literature grounding + the causal DAG
│  ├─ PAPER_OUTLINE.md       full manuscript scaffold + references
│  └─ DATA_DICTIONARY.md     every column explained
└─ outputs/                  generated data, figures, reports
```

## Honest limitations (by design, reported in `findings.md`)
- **Simulated** data: results validate the *method*, not human numbers. Real-data
  study is the next step.
- **Collinear constructs are unidentifiable**: low mood can't be separated from
  stress — the recovery flags this rather than hiding it.
- **Noisy self-report under-powers fatigue and boredom**: the latent-input check
  recovers them, motivating better passive sensing of alertness/engagement.
- **No sex → behaviour mechanism** is assumed; differences between people come
  from traits and context.

## Reproducibility
Everything is seeded (`config.GLOBAL_SEED`). Same config → identical outputs.

## Ethics
Attention data is intimate, and a model that protects attention can be inverted to
exploit it. This project is framed for **user-owned data, on-device inference, and
intervention (not engagement) objectives**. Please keep it that way.
