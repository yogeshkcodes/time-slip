# Time Slip

> 👉 **New here? Read [START_HERE.md](START_HERE.md) first** — the whole project
> explained in plain English: what it is, how to run it, what the results mean,
> and how to explain it to others.

**Finding *where*, *when*, and *why* your attention slips — from your own real
data, with a causally validated model behind it.**

Most tools stop at screen-time totals ("you spent 3h on your phone") or vague
correlations ("people scroll when bored"). Time Slip goes after the real
question: **what is causing *your* lapses, and what single change would cut them
most** — and it backs every claim with validation instead of vibes.

> Attention is becoming the scarce currency. Time Slip measures its exchange
> rate, person by person.

---

## What you can do with it

| Command | What you get |
|---|---|
| `python track_me.py` | **Tracks your real computer use** (foreground app + idle, 100% on-device, nothing uploaded) |
| `python analyze_tracker.py` | Your real focus spells, slips, rabbit holes, vigilance curve — from actual behaviour, no logging effort |
| `python analyze_me.py my_log.csv` | Your personal **slip fingerprint**: what share of your lapses each cause drives |
| `python obsidian_sync.py "<vault>"` | Log your routine in Obsidian daily notes; a report with charts is written back into the vault |
| `python whatif.py --boredom 4 --tot 40` | **Live risk right now** with an uncertainty interval + the single best lever to pull |
| `python whatif.py --policy` | Causal estimates of week-long interventions (phone away / DND / more sleep) |
| `python run_all.py` | Reproduce the full research study (validation + figures + paper-ready reports) |

## The science behind it

The model predicts a slip in the next 10 minutes and decomposes *why* into a
per-person **fingerprint** (phone pull, boredom, stress, fatigue, task
aversiveness, hunger, time-on-task, low motivation). What makes it trustworthy
is that it's validated **three independent ways** — and actively *attacked*:

1. **Ground-truth benchmark.** Causal attribution can't be graded on field data
   (the true cause of a phone grab is unobservable), so the method is graded on
   a cognitive-science-grounded benchmark cohort where the true causal structure
   is known by construction — the standard approach in causal inference. The
   pipeline **recovers the true hazard coefficients at Spearman 0.98** (90% sign
   agreement) and per-person fingerprints at **0.75**, using only self-loggable
   inputs. Risk prediction: ROC-AUC **0.76** for never-seen people, **0.77**
   personalised — versus 0.60 for a notifications-only baseline, with a learning
   curve that cleanly plateaus (the rest is irreducible randomness, not error).

2. **Real humans.** The model's causal story was then tested against a published
   experience-sampling study — **274 adults beeped ~8×/day for a week (~10k
   probes; Kane et al. 2017, *Psychological Science*)**. Boredom, fatigue, low
   task-interest, stress and low mood all predict real mind-wandering in the
   predicted direction: **83% sign agreement, rank-correlation 0.83**. (One
   genuine refinement: *effort* protects in real data — effort means engagement.)

3. **Falsification ("we tried to break it").** Most behavioural-ML work only
   shows evidence *for* a model; Time Slip ships a **refutation suite** that tries
   to refute itself — and passes **5/5**. A *placebo label* drives the model to
   chance (ROC 0.499); a *pure-noise "cause"* gets ~0% attribution (0.0001);
   every driver shows a monotone *dose-response*; recovery beats a shuffled-label
   *permutation null* (p≈0.03); and a *no-op intervention* yields ~0 effect. If
   any of these failed, the headline numbers would be artefacts — they don't.

And because the model is causal, it supports **intervention simulation** (a
do-operator): re-running the same people under "notifications batched + phone
away + slightly more sleep" cuts time lost to slips by **~46%** — a
model-implied effect, i.e. a quantified hypothesis ready for a real A/B trial.
Risk predictions ship with **Venn–Abers uncertainty intervals**, and population
fingerprints with bootstrap confidence bands — so every number has an error bar.

## The conceptual punchline

Two questions, two different answers — and the project keeps them separate:

| Question | Method | Dominant answer |
|---|---|---|
| What raises your **baseline lapse hazard**? | hazard ratios | boredom, depleted self-control |
| What is most **reducible at the moments you slip**? | counterfactual fingerprint | the phone pull, task aversiveness |

Distraction isn't one thing. The right lever is personal — see
`outputs/figures/per_person_fingerprint.png`.

---

## Quickstart

```bash
pip install -r requirements.txt

# use it on yourself, today
python track_me.py --minutes 90     # watch your real behaviour (local only)
python analyze_tracker.py           # -> your real focus/slip report
python whatif.py --boredom 4 --task deep_work --tot 40   # live risk + best lever

# or reproduce the full research study (~2.5 min)
python run_all.py                   # validation, figures, reports, trained model
```

Then read **`outputs/reports/findings.md`** and browse `outputs/figures/`.

## Key figures (in `outputs/figures/`)
| File | Shows |
|---|---|
| `realworld_validation.png` | the drivers corroborated in 274 real people |
| `falsification.png` | the 5 refutation tests the model survives |
| `recovery_scatter.png` | recovered vs true causal coefficients (benchmark) |
| `per_person_fingerprint.png` | each person's cause breakdown |
| `interventions.png` | causal effect of phone-away / DND / sleep policies |
| `hazard_ratios.png` | lapse hazard ratios during focused work |
| `vigilance_curve.png` | risk rising with time-on-task |
| `km_curve.png` | how long focus lasts before a slip |
| `model_performance.png` | ROC / PR / calibration vs baselines |
| `learning_curve.png` | accuracy vs amount of data (then it plateaus) |
| `regime_compare.png` | new-person vs personalised accuracy |
| `circadian_slips.png` | when slips happen across the day |
| `day_timeline_P0*.png` | a day's internal states with slips marked |

## Use your own data — `analyze_me.py`

```bash
python -m timeslip.schema       # 1. creates a blank log template (CSV)
#                                 2. fill in a row every ~15-30 min: what you
#                                    were doing, how it felt (1-5), did you slip
python analyze_me.py my_log.csv # 3. your personal report + fingerprint
```

No data yet? `python analyze_me.py` with no file analyses a generated example so
you can see the output format. Reports show *when* and *where* you slip, what's
elevated at slip moments, and — once you have ≥30 rows / ≥6 slips — your
personal fingerprint (cross-validated, shrunk toward the population prior while
your data is small).

## Embed it in your own project — `timeslip.api`

A small, stable, JSON-friendly facade so any app (a focus timer, a journaling
tool, a wearable companion, a study backend) can plug in:

```python
from timeslip.api import TimeSlip
ts = TimeSlip()                                    # loads the trained model once

ts.risk(boredom=4, task="deep_work", tot=40, phone=1, notifs=3)
# -> {"risk": 0.97, "interval": [0.88, 0.99],
#     "best_lever": {"name": "Make the task engaging / reframe it", ...},
#     "levers": [...] }                            # ranked, with effect sizes

ts.fingerprint("my_log.csv")     # -> per-cause share DataFrame (personal)
ts.tracker("outputs/tracker")    # -> behavioural summary from tracked data
```

Everything returns plain dicts/DataFrames and runs locally — drop it into a
Flask/FastAPI route, a notebook, or another pipeline unchanged. This is the
intended **hook for sister projects**.

## Log in Obsidian — `obsidian_sync.py`

```bash
python obsidian_sync.py "C:/path/to/Vault" --init   # scaffold TimeSlip/ folder
# ...log days as a markdown table in TimeSlip/logs/*.md (template provided)...
python obsidian_sync.py "C:/path/to/Vault"          # report written into vault
```

## Project structure

```
Time Slip/
├─ track_me.py               on-device real-time attention tracker (Windows)
├─ analyze_tracker.py        behavioural report from tracked real data
├─ analyze_me.py             analyse YOUR own logged routine
├─ whatif.py                 live slip-risk + best-lever recommender
├─ obsidian_sync.py          log routines in Obsidian, get a report back
├─ run_all.py                reproduce the full research study
├─ requirements.txt
├─ timeslip/
│  ├─ api.py                 embeddable facade (risk / fingerprint / tracker)
│  ├─ falsify.py             refutation suite (negative controls, dose-response)
│  ├─ tracker.py             parse + analyse real tracker logs
│  ├─ realworld.py           validation against real human ESM data (Kane 2017)
│  ├─ realdata.py            personal-log analysis (+ empirical-Bayes shrinkage)
│  ├─ interventions.py       causal do-operator: simulate policies, quantify effect
│  ├─ schema.py              self-logging schema; obsidian.py: vault integration
│  ├─ config.py              constructs + ground-truth coefficients (benchmark)
│  ├─ personas.py / simulate.py   benchmark cohort generator (36 people, 28 days)
│  ├─ features.py            self-loggable vs latent feature sets; risk windows
│  ├─ model.py               recovery, hazard, additive attribution surrogate
│  ├─ evaluate.py            regimes, calibration, learning curve, Venn-Abers
│  ├─ survival.py            Kaplan-Meier + discrete-time hazard + vigilance
│  ├─ explain.py             SHAP + counterfactual fingerprints + validation
│  └─ report.py              figures + reports
├─ docs/                     THEORY.md, PAPER_OUTLINE.md, DATA_DICTIONARY.md
└─ outputs/                  generated data, figures, reports, trained model
```

## Validation status & limits (the honest fine print)

- Benchmark-cohort metrics (recovery 0.98, fingerprints 0.75, ROC 0.76/0.77,
  interventions −46%) are measured on the ground-truth benchmark; real-human
  corroboration currently covers the *direction and ranking* of the drivers
  (Kane et al. 2017). The next step is a prospective study on real participants.
- Collinear constructs are unidentifiable (low mood vs stress) — flagged, not hidden.
- Noisy self-report under-powers fatigue/boredom attribution; passive sensing
  (the tracker) is the fix.
- Sex has no causal edge in the model; differences come from traits and context.
- Fully seeded and reproducible (`config.GLOBAL_SEED`).

## Ethics

Attention data is intimate, and a model that protects attention could be
inverted to exploit it. Time Slip is built for **user-owned data, on-device
inference, and intervention (not engagement) objectives**. The tracker never
uploads anything. Please keep it that way.
