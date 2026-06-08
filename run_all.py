"""
Time Slip - end-to-end pipeline.

    python run_all.py

Simulates the cohort, engineers features, trains the slip-risk model, runs the
survival, recovery and counterfactual-attribution analyses, then writes data,
figures and reports under ./outputs/. Everything is seeded for reproducibility.
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
import pandas as pd

from timeslip import config as C
from timeslip.simulate import simulate_all
from timeslip.features import build_features
from timeslip.model import (train_slip_model, train_oracle_model,
                            recover_coefficients, fit_discrete_hazard)
from timeslip.survival import (build_spells, km_curves, fit_discrete_time_hazard,
                               vigilance_curve)
from timeslip.explain import (counterfactual_attribution, ground_truth_attribution,
                              validate_attribution, shap_analysis)
from timeslip import report

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
DATA, FIG, REP = (os.path.join(OUT, d) for d in ("data", "figures", "reports"))


def main():
    t0 = time.time()
    for d in (DATA, FIG, REP):
        os.makedirs(d, exist_ok=True)

    print("[1/7] simulating cohort ...")
    minutes, episodes, personas = simulate_all(seed=C.GLOBAL_SEED)
    print(f"      {len(minutes):,} minute-rows, {len(episodes):,} slips, "
          f"{len(personas)} people")

    print("[2/7] engineering features ...")
    fb = build_features(minutes, personas)

    print("[3/7] training slip-risk model + baselines ...")
    model_res = train_slip_model(fb)
    oracle_res = train_oracle_model(fb)
    m = model_res["metrics"]
    print(f"      ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
          f"Brier={m['brier']:.3f}  (notif-only ROC={m['baseline_notif_roc']:.3f})")

    print("[4/7] recovering causal coefficients ...")
    rec = recover_coefficients(fb)
    odds = fit_discrete_hazard(fb)
    print(f"      Spearman(recovered,true)={rec['spearman']:.3f}  "
          f"sign-agreement={rec['sign_agreement']:.0%}")

    print("[5/7] survival / hazard analysis ...")
    spells = build_spells(fb["at_risk"])
    km = km_curves(spells)
    haz = fit_discrete_time_hazard(fb["at_risk"])
    vc = vigilance_curve(fb["at_risk"])

    print("[6/7] counterfactual attribution + SHAP ...")
    truth = ground_truth_attribution(fb)
    cf_real = counterfactual_attribution(model_res["model"], fb, "real")
    cf_oracle = counterfactual_attribution(oracle_res["model"], fb, "oracle")
    val_real = validate_attribution(cf_real, truth)
    val_oracle = validate_attribution(cf_oracle, truth)
    sh = shap_analysis(model_res["model"], fb)
    print(f"      attribution fidelity (self-logged) Spearman={val_real['overall']:.3f} "
          f"| latent-input check={val_oracle['overall']:.3f}")

    ctx = dict(
        minutes=minutes, episodes=episodes, personas=personas, horizon=fb["horizon_min"],
        model=model_res, oracle=oracle_res, recovery=rec, odds=odds,
        spells=spells, km=km, hazard=haz, vigilance=vc,
        truth=truth, cf_real=cf_real, cf_oracle=cf_oracle,
        val_real=val_real, val_oracle=val_oracle, shap=sh,
    )

    print("[7/7] writing data, figures, reports ...")
    # data
    minutes.to_csv(os.path.join(DATA, "minutes.csv"), index=False)
    episodes.to_csv(os.path.join(DATA, "episodes.csv"), index=False)
    personas.to_csv(os.path.join(DATA, "personas.csv"), index=False)
    rec["table"].to_csv(os.path.join(DATA, "recovery_table.csv"), index=False)
    if haz.get("available"):
        haz["summary"].to_csv(os.path.join(DATA, "hazard_ratios.csv"), index=False)
    odds["odds_ratios"].to_csv(os.path.join(DATA, "odds_ratios.csv"), index=False)
    cf_real["per_person"].to_csv(os.path.join(DATA, "fingerprints_self_logged.csv"), index=False)
    truth.to_csv(os.path.join(DATA, "fingerprints_ground_truth.csv"))
    if sh.get("available"):
        sh["global_importance"].to_csv(os.path.join(DATA, "shap_importance.csv"), index=False)

    summary = dict(
        n_minutes=int(len(minutes)), n_slips=int(len(episodes)),
        n_people=int(len(personas)), horizon_min=int(fb["horizon_min"]),
        metrics=m,
        recovery_spearman=rec["spearman"], recovery_sign_agreement=rec["sign_agreement"],
        attribution_spearman_self_logged=val_real["overall"],
        attribution_spearman_oracle=val_oracle["overall"],
        hazard_auc=haz.get("auc"),
        dominant_cause_per_person=cf_real["dominant"],
    )
    with open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    # figures + reports
    report.generate_all(ctx, FIG, REP)

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in ./outputs/")
    print(f"  data    -> {DATA}")
    print(f"  figures -> {FIG}")
    print(f"  reports -> {REP}  (start with findings.md)")


if __name__ == "__main__":
    main()
