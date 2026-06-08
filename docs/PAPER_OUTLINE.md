# Paper scaffold

**Working title:** *Time Slip: Recovering the Causes of Attention Lapses in Daily
Routines with a Structural Causal Model and Counterfactual Explanation*

> Numbers below are from the seeded run (`python run_all.py`); regenerate and
> update if you change the configuration. Figures referenced live in
> `outputs/figures/`.

---

## Abstract (draft)

Attention is increasingly described as a scarce currency, yet most quantitative
work on everyday distraction stops at correlation ("phone use co-occurs with
boredom") or at black-box prediction. We ask a harder question: *can we recover,
per person, the causal drivers of attention lapses — and prove the recovery is
trustworthy?* We build a structural causal model of a day in which eleven
internal psychological/physiological states evolve under literature-grounded
dynamics and drive a per-minute lapse hazard; lapses take one of five channels
(phone, mind-wandering, task-switch, snack, social) with heavy-tailed durations.
From minute-level logs of a cohort of 36 people over four weeks (~0.9M minutes),
with heterogeneous self-report noise, habit drift and disruption days, we train a
gradient-boosted model that predicts a lapse within the next ten minutes from
**self-loggable features only**. Under two honest regimes it reaches ROC-AUC 0.76
on people the model has never seen (cold-start) and 0.77 on known people's future
days (personalised, isotonic-calibrated), versus 0.60 for a notifications-only
baseline; a learning curve plateaus after ~15 people. Critically, because the
generative coefficients are known, we show the pipeline recovers them (Spearman
0.98 between recovered and true hazard coefficients; 90% sign agreement) and
recovers per-person counterfactual "slip fingerprints" (per-person Spearman 0.75
against ground truth, using an additive logistic surrogate for faithful
attribution). The
analysis separates two questions the literature conflates — *what raises baseline
lapse hazard* (boredom, depleted self-control) versus *what is most reducible at
the moments people actually slip* (the phone pull, task aversiveness) — and shows
the dominant lever is person-specific. We discuss honest limitations (low mood is
unrecoverable due to collinearity with stress; fatigue/boredom are
under-attributed from noisy self-report) and argue the validated method is a
precondition for applying it to real self-tracked data.

---

## 1. Introduction
- The attention economy framing (Wu, 2016; Newport, 2016): attention as currency.
- Gap: correlation and prediction dominate; causal, per-person, actionable
  accounts of *why* attention slips are missing.
- The measurement problem: real distraction data has no ground truth for cause.
- Contribution:
  1. An SCM of daily attention lapses grounded in cognitive science.
  2. A deployable self-logged predictor of near-term lapse risk.
  3. A **validation protocol**: recover known coefficients + counterfactual
     fingerprints before trusting the method on real data.
  4. The hazard-driver vs reducible-cause distinction.

## 2. Related work
- Sleep/circadian regulation: two-process model (Borbély, 1982); chronotype (Roenneberg et al., 2003, 2007).
- Vigilance decrement (Mackworth, 1948).
- Boredom as attentional disengagement (Eastwood et al., 2012).
- Mind-wandering (Smallwood & Schooler, 2006, 2015; Killingsworth & Gilbert, 2010).
- Stress and executive control (Eysenck et al., 2007, Attentional Control Theory; Arnsten, 2009).
- Self-control: trait (Tangney et al., 2004); strength model and its replication debate (Baumeister et al., 1998; Hagger et al., 2016).
- Motivation: Self-Determination Theory (Deci & Ryan, 1985, 2000); Flow (Csikszentmihalyi, 1990); procrastination / Temporal Motivation Theory (Steel, 2007; Steel & König, 2006).
- Smartphone habits & cues: checking habits (Oulasvirta et al., 2012); mere-presence "brain drain" (Ward et al., 2017); interruptions (Mark et al., 2008); variable-ratio reinforcement (Skinner).
- Interrupted-goal tension (Zeigarnik, 1927).
- Methods: SCM and counterfactuals (Pearl, 2009); SHAP (Lundberg & Lee, 2017); gradient boosting (Chen & Guestrin, 2016); survival/Cox (Cox, 1972); Experience Sampling (Larson & Csikszentmihalyi, 1983).

## 3. Methods
### 3.1 Cohort and routines
- 36 people (no names; sex M/F): 8 hand-built archetypes (disciplined morning
  lark, anxious night owl, notification-heavy manager, fragmented-day parent,
  rotating-shift worker, …) + 28 randomised members sampled across the trait
  space. 28 days, 1-minute resolution, waking hours (~0.9M minute-rows).
- Activity schedules from role-specific agendas with weekday/weekend variation and
  high-deadline "crunch" days.
- Difficulty stressors (`config.HARD`): per-person heterogeneous self-report
  noise, slow non-stationary drift in habit strength / self-control over the
  window, and low-frequency disruption (travel/sick) days for distribution shift.
### 3.2 Structural causal model
- State dynamics (Table: each state, its update rule, its references).
- The per-minute logistic hazard and its coefficients (`config.HAZARD`).
- Channel softmax and heavy-tailed durations.
- Person random intercepts.
### 3.3 Measurement model
- Latent states vs **self-logged** observations (Likert-discretised + Gaussian
  noise, σ=0.09). Deployed model restricted to self-loggable features.
### 3.4 Models & evaluation regimes
- Target: slip within next 10 min. Gradient-boosted trees (XGBoost, early
  stopping) vs logistic and notifications-only baselines.
- Two regimes: **cold-start** (train/test split by *person* — unseen people) and
  **personalised** (split by *day* — known people, future days). No leakage.
- Isotonic calibration on a held-out slice; learning curve over #training people.
- Discrete-time hazard (pooled logistic) with time-on-task terms → hazard ratios.
- Kaplan–Meier attention-survival. Production model + calibrator persisted.
### 3.5 Causal recovery and attribution
- Coefficient recovery: unpenalised logistic on latent inputs + person dummies vs
  true coefficients.
- Counterfactual fingerprints: per-slip leave-one-cause-out risk reduction to the
  person's calm baseline; normalised to shares. **Attribution uses an additive
  logistic surrogate** (the true process is additive on the logit scale; tree
  one-feature ablation is unfaithful, per-person Spearman ~0.25 vs ~0.75).
- Validation: against the identical counterfactual computed on the true hazard.

## 4. Results
- **Prediction:** cold-start ROC-AUC 0.76 (PR 0.71); personalised ROC-AUC 0.77
  (PR 0.71), Brier 0.182 → 0.175 after calibration; notifications-only ROC 0.60.
  Learning curve plateaus ~15 people. (Figs. `model_performance.png`,
  `regime_compare.png`, `learning_curve.png`, `per_person_auc.png`)
- **Recovery:** Spearman 0.98, sign agreement 90%; low-mood the lone outlier.
  (Fig. `recovery_scatter.png`)
- **Hazard / survival:** discrete-time hazard during focus; HR>1 for phone urge,
  task aversiveness, time-on-task, boredom, stress; HR<1 for self-control. Clear
  vigilance decrement. (Figs. `hazard_ratios.png`, `vigilance_curve.png`,
  `km_curve.png`)
- **Fingerprints:** self-logged vs ground-truth per-person Spearman 0.75 (latent-
  input check 0.96). Population reducible-cause ranking: phone pull ~45%, low
  intrinsic motivation ~16%, task aversiveness ~15%, stress ~10%, vigilance ~7%,
  hunger ~4%, fatigue ~2%, boredom ~1% (last two under-attributed under noise).
  Per-person heterogeneity (e.g., manager → phone pull dominant; disciplined lark
  → task aversiveness). (Figs. `population_fingerprint.png`,
  `per_person_fingerprint.png`, `attribution_validation.png`)
- **When/how:** time-of-day pattern (post-lunch dip, evening rise); channel-by-state
  structure. (Figs. `circadian_slips.png`, `channels.png`, `day_timeline_*.png`)

## 5. Discussion
- Two questions, two answers; person-specific levers; intervention implications.
- Why validation-before-deployment matters for behavioural ML.
- Honest limitations: collinear constructs (low mood vs stress) are unidentifiable;
  noisy self-report under-powers fatigue/boredom → motivates passive sensing;
  simulator is a model, not reality; single-cohort scale.

## 6. Ethics
- Attention data is intimate and manipulable; same model that *protects* attention
  can be inverted to *exploit* it. Argue for user-owned data, on-device inference,
  and intervention (not engagement) objectives. No sex→behaviour mechanism assumed.

## 7. From simulation to reality (future work)
- The self-logging schema (`timeslip/schema.py` / `features.py`) is designed so a
  real person's ESM + phone/calendar logs drop straight in.
- Next: small ESM study; passive alertness/engagement sensing; personalised
  hazard with hierarchical pooling; closed-loop just-in-time interventions; test
  whether the recovered fingerprints predict which intervention helps.

---

## References (verify formatting before submission)

1. Borbély, A. A. (1982). A two process model of sleep regulation. *Human Neurobiology*, 1(3), 195–204.
2. Roenneberg, T., Wirz-Justice, A., & Merrow, M. (2003). Life between clocks: daily temporal patterns of human chronotypes. *J. Biological Rhythms*, 18(1), 80–90.
3. Mackworth, N. H. (1948). The breakdown of vigilance during prolonged visual search. *Quarterly J. Experimental Psychology*, 1(1), 6–21.
4. Eastwood, J. D., Frischen, A., Fenske, M. J., & Smilek, D. (2012). The unengaged mind: Defining boredom in terms of attention. *Perspectives on Psychological Science*, 7(5), 482–495.
5. Smallwood, J., & Schooler, J. W. (2006). The restless mind. *Psychological Bulletin*, 132(6), 946–958.
6. Smallwood, J., & Schooler, J. W. (2015). The science of mind wandering. *Annual Review of Psychology*, 66, 487–518.
7. Killingsworth, M. A., & Gilbert, D. T. (2010). A wandering mind is an unhappy mind. *Science*, 330(6006), 932.
8. Eysenck, M. W., Derakshan, N., Santos, R., & Calvo, M. G. (2007). Anxiety and cognitive performance: Attentional control theory. *Emotion*, 7(2), 336–353.
9. Arnsten, A. F. T. (2009). Stress signalling pathways that impair prefrontal cortex structure and function. *Nature Reviews Neuroscience*, 10(6), 410–422.
10. Tangney, J. P., Baumeister, R. F., & Boone, A. L. (2004). High self-control predicts good adjustment, less pathology, better grades, and interpersonal success. *J. Personality*, 72(2), 271–324.
11. Baumeister, R. F., Bratslavsky, E., Muraven, M., & Tice, D. M. (1998). Ego depletion: Is the active self a limited resource? *JPSP*, 74(5), 1252–1265.
12. Hagger, M. S., et al. (2016). A multilab preregistered replication of the ego-depletion effect. *Perspectives on Psychological Science*, 11(4), 546–573.
13. Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits: Self-determination theory. *Psychological Inquiry*, 11(4), 227–268.
14. Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience.* Harper & Row.
15. Steel, P. (2007). The nature of procrastination. *Psychological Bulletin*, 133(1), 65–94.
16. Steel, P., & König, C. J. (2006). Integrating theories of motivation. *Academy of Management Review*, 31(4), 889–913.
17. Wood, W., & Neal, D. T. (2007). A new look at habits and the habit–goal interface. *Psychological Review*, 114(4), 843–863.
18. Oulasvirta, A., Rattenbury, T., Ma, L., & Raita, E. (2012). Habits make smartphone use more pervasive. *Personal and Ubiquitous Computing*, 16(1), 105–114.
19. Ward, A. F., Duke, K., Gneezy, A., & Bos, M. W. (2017). Brain drain: The mere presence of one's own smartphone reduces available cognitive capacity. *J. Association for Consumer Research*, 2(2), 140–154.
20. Mark, G., Gudith, D., & Klocke, U. (2008). The cost of interrupted work: more speed and stress. *CHI '08*, 107–110.
21. Zeigarnik, B. (1927). Über das Behalten von erledigten und unerledigten Handlungen. *Psychologische Forschung*, 9, 1–85.
22. Wu, T. (2016). *The Attention Merchants.* Knopf.
23. Newport, C. (2016). *Deep Work.* Grand Central.
24. Larson, R., & Csikszentmihalyi, M. (1983). The experience sampling method. *New Directions for Methodology of Social and Behavioral Science*, 15, 41–56.
25. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
26. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
27. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
28. Cox, D. R. (1972). Regression models and life-tables. *J. Royal Statistical Society B*, 34(2), 187–220.
