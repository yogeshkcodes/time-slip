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
**self-loggable** features — recovers the generator's true causal structure. Then
we close the loop with **real data**: an on-device tracker of your own computer
use, and an **external check against a published human dataset** (Kane et al.
2017, 274 adults) that *corroborates* the simulator's causal story.

> Attention is becoming the scarce currency. Time Slip is about measuring its
> exchange rate, person by person.

## Real data, not just simulation
- **`track_me.py`** — a zero-dependency, 100%-on-device tracker that logs your
  real foreground app + idle time, reconstructing your actual focus spells,
  task-switches and rabbit holes (`analyze_tracker.py`).
- **`timeslip/realworld.py`** — validates the model's drivers against real
  experience-sampling data from 274 people (~10k probes): **83% sign agreement,
  rank-correlation 0.83**. Boredom, fatigue, low task-interest, stress and low
  mood all predict real mind-wandering in the predicted direction.
- **`whatif.py`** — type your current state, get your calibrated next-10-min slip
  risk *with an uncertainty interval* (Venn–Abers) and the single best lever.
- **`timeslip/interventions.py`** — a causal *do-operator*: re-runs the same
  people under "phone away / notifications off / more sleep" and quantifies the
  effect (combined: ~46% less time lost — model-implied, a hypothesis for a real
  trial).

---

## What it does

1. **Simulates** a cohort of **36 people** (8 hand-built archetypes + 28
   randomised members; no names; sex M/F) over **28 days** at 1-minute resolution
   — **~0.9M logged minutes, ~34k slips** — with the realism stressors that make
   it *hard*: heterogeneous self-report noise, slow habit/self-control drift over
   the weeks, and disruption ("travel/sick") days. Eleven internal states evolve
   under literature-grounded dynamics and drive a per-minute **slip hazard**;
   lapses take one of five channels (phone, mind-wandering, task-switch, snack,
   social) with heavy-tailed durations (the literal "time slip").

2. **Predicts** a slip in the next 10 minutes from self-loggable features only,
   evaluated under two honest regimes:
   - **Cold-start** (people the model has *never seen*): **ROC-AUC ≈ 0.76**.
   - **Personalised** (known person, *future* days — the Obsidian case): **≈ 0.77**,
     isotonic-calibrated.
   - vs **≈ 0.60** for a notifications-only baseline — internal states, not just
     pings, carry the signal. A **learning curve** shows accuracy plateauing after
     ~15 people: the remaining gap to 1.0 is the *irreducible* randomness of the
     exact minute a lapse starts, not a fixable error. The trained model is
     **persisted** to `outputs/model/` for inference.

3. **Recovers the causes** and checks itself:
   - True vs recovered hazard coefficients: **Spearman 0.98**, 90% sign agreement.
   - Per-person counterfactual "slip fingerprints" vs ground truth: **per-person
     Spearman ≈ 0.75** (attribution uses an additive logistic surrogate — faithful
     for counterfactuals — while XGBoost does the predicting).

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
python run_all.py            # ~2.5 min; cohort study + real-data validation + interventions
python analyze_me.py         # analyse a self-logged routine (example if no file)
python track_me.py --minutes 90   # track your REAL computer use, then:
python analyze_tracker.py    # behavioural focus/slip report from real data
python whatif.py --boredom 4 --task deep_work --tot 40   # live risk + best lever
python obsidian_sync.py "C:/path/to/Vault" --init        # log routines in Obsidian
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
| `learning_curve.png` | accuracy vs amount of data (then it plateaus) |
| `regime_compare.png` | cold-start vs personalised accuracy |
| `per_person_auc.png` | accuracy distribution across people |
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

## Log in Obsidian — `obsidian_sync.py`

Keep the whole loop inside your Obsidian vault:

```bash
python obsidian_sync.py "C:/path/to/Vault" --init   # scaffold a TimeSlip/ folder
#   ... log your days as a markdown table in TimeSlip/logs/*.md (a template and
#       an example are created for you; columns are mostly 1-5 scales) ...
python obsidian_sync.py "C:/path/to/Vault"          # parse logs -> write report
```

It writes **`TimeSlip/Time Slip Report.md`** back into the vault with embedded
charts (when/where/why you slip + your fingerprint), so you read your results in
Obsidian. Re-run any time you add more days. See `TimeSlip/README.md` (created by
`--init`) for the exact column format.

## Project structure

```
Time Slip/
├─ run_all.py                end-to-end pipeline (study + real-data validation)
├─ analyze_me.py             analyse YOUR own logged routine
├─ track_me.py               on-device real-time attention tracker (Windows)
├─ analyze_tracker.py        behavioural report from tracked real data
├─ whatif.py                 live slip-risk + best-lever recommender
├─ obsidian_sync.py          log routines in Obsidian, get a report back
├─ requirements.txt
├─ timeslip/
│  ├─ config.py              constructs, ground-truth coefficients, cohort scale
│  ├─ personas.py            archetypes + randomised cohort (traits + sex, no names)
│  ├─ simulate.py            structural causal simulator (states → hazard → slips)
│  ├─ features.py            self-loggable vs latent feature sets; risk windows
│  ├─ model.py               recovery, hazard, oracle + logistic attribution model
│  ├─ evaluate.py            regimes, calibration, learning curve, Venn–Abers, persistence
│  ├─ survival.py            Kaplan–Meier + discrete-time hazard + vigilance
│  ├─ explain.py             SHAP + counterfactual fingerprints + validation
│  ├─ interventions.py       causal do-operator: simulate policies, quantify effect
│  ├─ realworld.py           external validation vs real human ESM data (Kane 2017)
│  ├─ tracker.py             parse + analyse real tracker logs
│  ├─ schema.py              self-logging schema for real data
│  ├─ realdata.py            analyse a real person's log (+ empirical-Bayes shrinkage)
│  ├─ obsidian.py            parse vault logs + write report back into the vault
│  └─ report.py              figures + per-person reports
├─ docs/
│  ├─ THEORY.md              literature grounding + the causal DAG
│  ├─ PAPER_OUTLINE.md       full manuscript scaffold + references
│  └─ DATA_DICTIONARY.md     every column explained
└─ outputs/                  generated data, figures, reports, model/
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
