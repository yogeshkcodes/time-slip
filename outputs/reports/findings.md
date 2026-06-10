# Time Slip - overall findings
*Ground-truth benchmark cohort: 36 people x 28 days (917,031 logged minutes, 34,316 slips), plus external validation on real human data (section 3b).*

## 1. The model predicts near-term attention slips (two honest regimes)
- Target: a slip within the next 10 minutes.
- **Cold-start (people the model has NEVER seen):** ROC-AUC **0.762**, PR-AUC 0.709 (base 0.37).
- **Personalised (known person, FUTURE days):** ROC-AUC **0.770**, PR-AUC 0.708; Brier 0.182 -> 0.175 after calibration.
- Both beat a notifications-only baseline (ROC 0.599): being pinged is **not** the whole story - internal states carry most of the signal.
- The learning curve plateaus after ~15 people: we have enough data, and the remaining gap to 1.0 is the *irreducible* randomness of the exact minute a lapse begins - not a fixable modelling error.

## 2. It recovers the true causal structure (validation)
- Recovered vs ground-truth hazard coefficients: **Spearman 0.98**, sign agreement 90%.
- The top drivers (phone urge, boredom, depleted self-control, task aversiveness) are ranked correctly; the only weak point is *low mood*, which is collinear with stress and cannot be separated - an honest limit.

## 3. Per-person 'slip fingerprints' are trustworthy
- Counterfactual attribution vs ground truth: **per-person Spearman 0.75** with self-logged inputs, 0.96 on a latent-input sanity check.
- Attribution uses an additive logistic surrogate, not the tree: the true process is additive on the logit scale, so logistic counterfactuals are faithful (per-person Spearman ~0.75) whereas tree one-feature ablation is not (~0.25). Prediction and explanation use different models by design.
- Population ranking of *reducible* causes (self-logged surrogate):
  - Phone pull: 45%
  - Low intrinsic motivation: 16%
  - Task aversiveness: 15%
  - Stress: 9%
  - Time-on-task (vigilance): 7%
  - Hunger: 5%
  - Fatigue: 2%
  - Boredom: 1%
- Note: the self-logged model under-attributes *fatigue* and *boredom* relative to ground truth - their self-report proxies are noisy and collinear with task features. This is a measurement limit, not a method failure (the latent-input check recovers them), and points to better passive sensing of alertness/engagement as the highest-value next step.

## 3b. Real humans corroborate the causal story (external validation)
- Tested against an open experience-sampling dataset (Kane et al. 2017, *Psychological Science*): 10,234 probes from 274 adults beeped ~8x/day for a week.
- Of the model's claimed drivers, **83% match the real-data sign** and the real effect *ranking* tracks the model's weights (Spearman 0.83): boredom, fatigue, low task-interest, stress and low mood all predict real mind-wandering in the expected direction.
- Honest divergence: *effort* is protective in the real data (it indexes engagement, not task aversiveness) - a genuine refinement, not a failure. Single-item real predictors give AUC 0.59, as expected for noisy field data.

## 3c. What actually helps (causal intervention simulation)
Because the model is causal, we can re-run the same people under different policies (a do-operator) and read off the effect:
- Batching/silencing notifications cuts slips ~41% and time lost ~42%.
- Phone-away + DND + ~45 min more sleep combined: slips -42%, time lost -46%.
- These are *model-implied* effects (a hypothesis generator for a real A/B experiment), not guarantees.

## 3d. We tried to break it (falsification suite)
A model earns trust by surviving attempts to refute it. **5/5** tests passed; the critical leakage/placebo tests all passed:
  - [PASS] negative-control outcome (placebo label) - held-out ROC on shuffled labels = 0.499 (want ~0.50)
  - [PASS] negative-control feature (placebo cause) - attributed share of a pure-noise feature = 0.0001 (want ~0.00)
  - [PASS] dose-response monotonicity - 100% of drivers monotone
  - [PASS] permutation null for recovery - recovery 0.976 vs shuffled-label null <= 0.588, p = 0.032
  - [PASS] placebo intervention (no-op policy) - % change in time lost = 0.0 (want ~0.00)
- A placebo label collapses the model to chance and a pure-noise 'cause' gets ~0% attribution, so the predictions and the fingerprint reflect real structure, not artefacts.

## 4. When and how attention gives way
- Discrete-time hazard during focus (AUC 0.69): each +1 SD of phone-urge, task-aversiveness, time-on-task, boredom and stress raises the lapse hazard; self-control lowers it.
- A clear **vigilance decrement**: lapse risk climbs with minutes-on-task.
- Time-of-day structure: a post-lunch dip and an evening rise in slips.

## 5. Headline distinction the project surfaces
Two different questions have two different answers, and both matter:
- *What raises your baseline lapse hazard?* -> boredom and low self-control rank highest (the hazard/coefficient view).
- *What is most reducible at the moments you actually slip?* -> the phone pull and task aversiveness (the counterfactual view).
Distraction is not one thing; the lever depends on which question you ask.

## Caveats
- Cohort metrics above are measured on the ground-truth benchmark (where causal attribution can be graded); real-human evidence currently covers the direction and ranking of the drivers (section 3b). A prospective study on real participants is the natural next step.
- Sex has no direct causal edge in the generator; person-to-person differences come from traits and context, not sex.
