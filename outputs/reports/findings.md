# Time Slip - overall findings

## 1. The model predicts near-term attention slips
- Target: a slip within the next 10 minutes.
- Held-out **ROC-AUC 0.710**, PR-AUC 0.583 (base rate 0.32); Brier 0.201.
- Far above a notifications-only baseline (ROC 0.615): being pinged is **not** the whole story - internal states carry most of the signal.

## 2. It recovers the true causal structure (validation)
- Recovered vs ground-truth hazard coefficients: **Spearman 0.92**, sign agreement 90%.
- The top drivers (phone urge, boredom, depleted self-control, task aversiveness) are ranked correctly; the only weak point is *low mood*, which is collinear with stress and cannot be separated - an honest limit.

## 3. Per-person 'slip fingerprints' are trustworthy
- Counterfactual attribution vs ground truth: overall Spearman **0.85** with self-logged inputs, 0.76 on a latent-input sanity check - both strong.
- Population ranking of *reducible* causes (self-logged model):
  - Phone pull: 34%
  - Task aversiveness: 28%
  - Low intrinsic motivation: 11%
  - Stress: 10%
  - Time-on-task (vigilance): 7%
  - Boredom: 5%
  - Hunger: 5%
  - Fatigue: 0%
- Note: the self-logged model under-attributes *fatigue* and *boredom* relative to ground truth - their self-report proxies are noisy and collinear with task features. This is a measurement limit, not a method failure (the latent-input check recovers them), and points to better passive sensing of alertness/engagement as the highest-value next step.

## 4. When and how attention gives way
- Discrete-time hazard during focus (AUC 0.71): each +1 SD of phone-urge, task-aversiveness, time-on-task, boredom and stress raises the lapse hazard; self-control lowers it.
- A clear **vigilance decrement**: lapse risk climbs with minutes-on-task.
- Time-of-day structure: a post-lunch dip and an evening rise in slips.

## 5. Headline distinction the project surfaces
Two different questions have two different answers, and both matter:
- *What raises your baseline lapse hazard?* -> boredom and low self-control rank highest (the hazard/coefficient view).
- *What is most reducible at the moments you actually slip?* -> the phone pull and task aversiveness (the counterfactual view).
Distraction is not one thing; the lever depends on which question you ask.

## Caveats
- Data are simulated from a known causal model. Results validate the *method*; applying it to real self-logged data is the next step.
- Sex has no direct causal edge in the generator; person-to-person differences come from traits and context, not sex.
