# Your slip report

- Logged **629 intervals** across 14 days.
- **388 slips** (~27.7/day).
- **~191 min/day** off-task in slips.
- Channel mix: mind_wandering 31%, phone 27%, task_switch 20%, social 14%, snack 8%

## WHEN — your highest-risk hours
- 13:00 (86%), 17:00 (79%), 14:00 (77%)

## WHERE — activities you slip in most
| Activity | slip rate | logged |
|---|---|---|
| admin | 97% | 29 |
| meeting | 95% | 44 |
| errands | 89% | 9 |
| deep_work | 84% | 128 |
| commute | 82% | 33 |
| exercise | 64% | 39 |

## WHY — what's elevated when you slip (model-free)
Standardised gap between slip moments and focused moments (positive = higher when you slip):
- **task aversiveness**: +0.79 SD
- **task difficulty**: +0.71 SD
- **stress**: +0.44 SD
- **alertness**: +0.43 SD
- **deadline pressure**: +0.23 SD

## Your personal fingerprint (model-based)
- Personal risk model trained on your data (cross-validated AUC 0.77 — 0.50 is chance).

| Cause | Share of your reducible slip risk |
|---|---|
| Phone pull | 49% |
| Fatigue | 29% |
| Hunger | 13% |
| Low intrinsic motivation | 8% |
| Task aversiveness | 1% |
| Stress | 0% |
| Boredom | 0% |
| Time-on-task (vigilance) | 0% |

**Your dominant trigger: Phone pull.**

> Reading the two 'why' sections together: the WHY list shows what is *present* when you slip (often hard or aversive tasks — that's simply when focus is demanded). The fingerprint shows what is most *reducible* — the lever that would cut your slips the most. They answer different questions, so they can differ.

---
*Figures: `me_when.png`, `me_why.png`, `me_fingerprint.png` in `outputs\me`.*