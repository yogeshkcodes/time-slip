# Theoretical foundations of *Time Slip*

This document grounds every variable and causal edge in the simulator and the
analysis in the cognitive-science literature. The goal of the project is to find
**where, when and why** attention disengages during a day ("time slips") and to
do it in a way that distinguishes *causes* from mere *correlates*. The strategy
is to (1) encode the leading theories of attention failure as an explicit
structural causal model, (2) generate detailed daily routines from it, and (3)
show that an explainable-ML + counterfactual pipeline recovers the causal
structure — a precondition for trusting it on real self-logged data.

> Throughout, "slip" / "lapse" means a discrete episode in which attention
> disengages from the current activity: a glance-turned-scroll on the phone, a
> bout of mind-wandering, an impulsive task-switch, a snack run, or drifting into
> conversation. Distraction is treated as **multi-channel**, not phone-only.

---

## 1. The constructs and who they come from

### Internal states (time-varying)

| State | Construct | Key references (themes) |
|---|---|---|
| `sleep_pressure` | Homeostatic sleep drive (Process S) | Borbély's **two-process model** of sleep regulation |
| `circadian` | Circadian alerting signal (Process C); chronotype | Borbély; Roenneberg **chronotype / social jetlag (MCTQ)** |
| `alertness` / `fatigue` | Net arousal available for control | Two-process synthesis; **vigilance** literature |
| `energy` | Metabolic/cognitive resource | Glucose-and-self-control debate (Gailliot & Baumeister), treated cautiously |
| `hunger` | Visceral drive competing for attention | Interoception / drive theory |
| `boredom` | Attentional disengagement under-stimulation | **Eastwood's attention-based theory of boredom**; meaning/under-stimulation accounts |
| `stress` | Anxiety/arousal that narrows control | **Attentional Control Theory** (Eysenck et al.); Arnsten on prefrontal control under stress |
| `mood` | Affective valence influencing regulation | Affect-as-information; self-regulation-and-affect |
| `focus_reserve` | Momentary self-control capacity | **Trait self-control** (Tangney, Baumeister & Boone); strength/resource model (used as a mechanism, while acknowledging the ego-depletion replication debate) |
| `urge_to_check` | Phone-checking impulse | **Habit automaticity** (Wood & Neal; Oulasvirta "checking habits"); cue-reactivity |

### Task / context (largely exogenous each minute)

| Variable | Construct | References (themes) |
|---|---|---|
| `difficulty`, `intrinsic`, `aversive` | Challenge, intrinsic motivation, task aversiveness | **Self-Determination Theory** (Deci & Ryan); **Flow** challenge–skill balance (Csikszentmihalyi); **Temporal Motivation Theory** of procrastination (Steel) |
| `time_on_task` → `vigilance` | Vigilance decrement | **Mackworth clock test**; sustained-attention decrement |
| `phone_in_reach` | Cue availability / "mere presence" | Ward et al. **"Brain Drain"**; smartphone-proximity effects |
| `notif` | External interruption; variable-ratio reward | Mark et al. on interruptions; operant **variable-ratio** reinforcement (Skinner); attention-economy critiques (Wu; Newport) |
| `deadline`, `open_tasks` | Goal pressure, interruption load | **Zeigarnik effect**; goal-conflict |
| `social` | Social facilitation/temptation | Social context of self-control |

### Outcome

| Variable | Construct |
|---|---|
| `slip_onset` | Instantaneous attention disengagement (a discrete hazard event) |
| `slip_channel` | The *form* of the lapse: phone / mind-wandering / task-switch / snack / social — connecting to **mind-wandering** (Smallwood & Schooler, decoupling) and media-multitasking literatures |
| `duration` | Length of the off-task episode; phone episodes are heavy-tailed (the literal "time slip") |

---

## 2. The structural causal model (DAG)

Stable **traits** set the parameters of a person's day. Within the day,
**internal states** evolve under simple dynamics and, together with the
**task/context**, drive an instantaneous **slip hazard**. A slip feeds back:
it relieves boredom, resets time-on-task, consumes time, and (for conscientious
people on important work) creates a guilt-driven stress bump.

```
            traits (self-control, neuroticism, conscientiousness,
                    habit strength, chronotype, caffeine use, sex*)
                 │              │                │
                 ▼              ▼                ▼
   ┌─────────────────────────────────────────────────────────┐
   │ INTERNAL STATES (minute dynamics)                         │
   │  sleep_pressure ─┐                                        │
   │  circadian ──────┼──► alertness/fatigue                   │
   │  energy, hunger  │        │                               │
   │  boredom ◄───────┴── time_on_task, (1-intrinsic)          │
   │  stress ◄── deadline, difficulty, neuroticism             │
   │  mood ◄── stress, fatigue, social                         │
   │  focus_reserve ◄── self-control, fatigue, effort spent    │
   │  urge_to_check ◄── habit, boredom, notifications          │
   └───────────────┬─────────────────────────────┬────────────┘
                   │                              │
       TASK/CONTEXT│ (difficulty, intrinsic,      │
       aversive, vigilance, phone_in_reach,       │
       notifications, deadline, social)           │
                   ▼                              ▼
            ┌────────────────────────────────────────┐
            │  SLIP HAZARD  (logistic, per minute)     │
            │  logit = Σ βᵢ·driverᵢ − β_sc·focus_reserve│
            └───────────────┬──────────────────────────┘
                            ▼
                  slip_onset → channel (softmax) → duration
                            │
        feedback: boredom↓, time_on_task→0, time consumed, guilt→stress↑
```

\* **Sex** is recorded but has **no direct edge** into the hazard. Observed
differences between people of different sexes arise only through their trait and
context profiles. This is a deliberate, conservative choice: the data do not
justify a direct sex→distraction mechanism, and baking one in would manufacture a
stereotype. (The framework can test for one if a real study warranted it.)

### The hazard equation (ground truth)

For each at-risk minute,

```
logit P(slip onset) = β0 + uₚ
    + β_boredom·boredom + β_fatigue·fatigue + β_stress·stress
    + β_aversive·aversiveness + β_vigilance·min(time_on_task, cap)
    + β_hunger·hunger + β_urge·urge_eff
    + β_lowIntr·(1−intrinsic) + β_lowMood·(1−mood)
    − β_selfControl·focus_reserve
```

where `uₚ` is a person random intercept and `urge_eff` is the urge gated by
phone availability. The coefficients live in `timeslip/config.py::HAZARD`. Their
**ratios and signs** are the scientific content; the absolute scale is tuned only
so the per-minute event rate is realistic.

The **channel** of a realised slip is drawn from a softmax over channel-specific
scores (`config.CHANNEL_WEIGHTS`): e.g., phone is pulled by urge + boredom +
habit; mind-wandering by fatigue + low intrinsic motivation + overload;
task-switch by stress + deadline; snack by hunger + low energy; social by company
+ boredom. Phone episode **duration** lengthens with boredom and habit strength —
the rabbit-hole / variable-reward mechanism behind a quick check becoming 25
minutes.

---

## 3. Why a simulator (and why this is not circular)

Real distraction data has no ground truth: when someone picks up their phone we
never observe the true cause. So we cannot directly test whether an attribution
method is *right* — only whether it predicts. By generating data from a known
causal model we obtain ground-truth coefficients and per-slip causal
contributions, and can ask the decisive question: **does the pipeline recover
them?**

This is not circular because the analysis is **blind to the generator**:

- The deployed predictor uses only **self-loggable** features (noisy, discretised
  self-reports + calendar/phone metadata) — never the latent states or the true
  coefficients.
- Recovery is judged against held-out structure (Spearman of recovered vs true
  coefficients; per-person Spearman of counterfactual vs true fingerprints).
- The honest failures are reported, not hidden: e.g., *low mood* cannot be
  separated from stress (collinearity), and *fatigue/boredom* are under-attributed
  when only noisy self-reports are available.

If the method could **not** recover known causes from clean data, we would have no
business running it on real data. Passing this bar is the licence to deploy.

---

## 4. Two questions, two answers (the core conceptual contribution)

The project insists on separating two things that the literature often conflates:

1. **What raises the baseline hazard of lapsing?** — answered by the hazard
   coefficients / hazard ratios. Here boredom and depleted self-control are
   dominant: a chronically bored or self-control-depleted state makes lapses
   likely *at any moment*.

2. **What is most reducible at the moments a person actually slips?** — answered
   by the counterfactual fingerprint. Here the phone pull and task aversiveness
   dominate, because they spike sharply right before real slips and can be moved
   to a calm baseline.

Both are "causes," but they answer different intervention questions. A boredom-
prone person is not best helped by a notification blocker; a notification-driven
person is not best helped by adding challenge. *Attention is the currency; the
exchange rate is personal.*

---

## 5. Mapping to the code

| Concept | File |
|---|---|
| Constructs, coefficients, channels | `timeslip/config.py` |
| Traits per person (no names; M/F) | `timeslip/personas.py` |
| State dynamics + hazard + channels | `timeslip/simulate.py` |
| Self-loggable vs latent feature sets | `timeslip/features.py` |
| Risk model, baselines, coefficient recovery | `timeslip/model.py` |
| Kaplan–Meier + discrete-time hazard + vigilance | `timeslip/survival.py` |
| SHAP + counterfactual fingerprints + validation | `timeslip/explain.py` |

The citations above are given by theme rather than as a formatted bibliography;
`docs/PAPER_OUTLINE.md` lists the works to cite formally in a manuscript.
