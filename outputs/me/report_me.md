# Your slip report

- Logged **1276 intervals** across 28 days.
- **761 slips** (~27.2/day).
- **~181 min/day** off-task in slips.
- Channel mix: phone 34%, mind_wandering 31%, task_switch 16%, social 11%, snack 8%

## WHEN — your highest-risk hours
- 13:00 (76%), 12:00 (75%), 11:00 (74%)

## WHERE — activities you slip in most
| Activity | slip rate | logged |
|---|---|---|
| break | 96% | 27 |
| admin | 94% | 63 |
| meeting | 88% | 88 |
| deep_work | 80% | 265 |
| commute | 76% | 75 |
| errands | 74% | 19 |

## WHY — what's elevated when you slip (model-free)
Standardised gap between slip moments and focused moments (positive = higher when you slip):
- **task aversiveness**: +0.76 SD
- **task difficulty**: +0.56 SD
- **stress**: +0.41 SD
- **alertness**: +0.40 SD
- **boredom**: +0.22 SD

## Your personal fingerprint (model-based)
- Personal risk model trained on your data (cross-validated AUC 0.76 — 0.50 is chance).

| Cause | Share of your reducible slip risk |
|---|---|
| Phone pull | 68% |
| Task aversiveness | 18% |
| Low intrinsic motivation | 7% |
| Hunger | 5% |
| Fatigue | 1% |
| Boredom | 1% |
| Stress | 1% |
| Time-on-task (vigilance) | 0% |

**Your dominant trigger: Phone pull.**

> Reading the two 'why' sections together: the WHY list shows what is *present* when you slip (often hard or aversive tasks — that's simply when focus is demanded). The fingerprint shows what is most *reducible* — the lever that would cut your slips the most. They answer different questions, so they can differ.

---
*Figures: `me_when.png`, `me_why.png`, `me_fingerprint.png` in `outputs\me`.*