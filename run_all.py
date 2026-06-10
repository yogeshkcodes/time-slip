"""
Time Slip - end-to-end pipeline (the simulation study).

    python run_all.py

Simulates the full cohort, engineers features, evaluates the slip-risk model
under two honest regimes (cold-start + personalised), calibrates it, draws a
learning curve, trains and PERSISTS a production model, runs the survival,
coefficient-recovery and counterfactual-attribution analyses, then writes data,
figures and reports under ./outputs/. Seeded for reproducibility.

To analyse one person's own logged routine instead, use: python analyze_me.py
"""

from __future__ import annotations
import os
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score

from timeslip import config as C
from timeslip.simulate import simulate_all
from timeslip.features import build_features
from timeslip.model import (recover_coefficients, fit_discrete_hazard,
                            train_oracle_model, train_attribution_model)
from timeslip.evaluate import (regime_cold_start, regime_personalized, calibrate,
                               learning_curve, train_production_model, save_artifacts)
from timeslip.survival import (build_spells, km_curves, fit_discrete_time_hazard,
                               vigilance_curve)
from timeslip.explain import (counterfactual_attribution, ground_truth_attribution,
                              validate_attribution, shap_analysis)
from timeslip import report

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
DATA, FIG, REP, MODEL = (os.path.join(OUT, d) for d in
                         ("data", "figures", "reports", "model"))


def main():
    t0 = time.time()
    for d in (DATA, FIG, REP, MODEL):
        os.makedirs(d, exist_ok=True)

    print("[1/8] simulating cohort ...")
    minutes, episodes, personas = simulate_all(seed=C.GLOBAL_SEED)
    print(f"      {len(minutes):,} minute-rows, {len(episodes):,} slips, "
          f"{len(personas)} people, {minutes['day'].nunique()} days")

    print("[2/8] engineering features ...")
    fb = build_features(minutes, personas)

    print("[3/8] evaluating model (cold-start + personalised) ...")
    cold = regime_cold_start(fb)
    pers = regime_personalized(fb)
    cal = calibrate(pers["model"].predict_proba(pers["val"][0])[:, 1], pers["val"][1],
                    pers["p"], pers["y"])
    # notifications-only baseline on the personalised test set
    te = fb["test_mask"]
    notif_p = (fb["X_real"]["notif_15"].to_numpy()[te]
               + 0.1 * fb["X_real"]["notif"].to_numpy()[te])
    notif_base = {"y": fb["y_window"][te], "p": notif_p}
    print(f"      cold-start ROC={cold['metrics']['roc_auc']:.3f} | "
          f"personalised ROC={pers['metrics']['roc_auc']:.3f} | "
          f"notif-only ROC={roc_auc_score(notif_base['y'], notif_base['p']):.3f}")

    print("[4/8] learning curve + production model ...")
    lc = learning_curve(fb)
    prod = train_production_model(fb)

    print("[5/8] recovering causal coefficients ...")
    rec = recover_coefficients(fb)
    odds = fit_discrete_hazard(fb)
    print(f"      Spearman(recovered,true)={rec['spearman']:.3f}  "
          f"sign-agreement={rec['sign_agreement']:.0%}")

    print("[6/8] survival / hazard analysis ...")
    spells = build_spells(fb["at_risk"])
    km = km_curves(spells)
    haz = fit_discrete_time_hazard(fb["at_risk"])
    vc = vigilance_curve(fb["at_risk"])

    print("[7/8] counterfactual attribution + SHAP ...")
    truth = ground_truth_attribution(fb)
    # attribution uses an additive logistic surrogate (faithful counterfactuals);
    # the XGBoost model above remains the predictor for accuracy.
    attr_real = train_attribution_model(fb, "real")
    attr_oracle = train_attribution_model(fb, "oracle")
    cf_real = counterfactual_attribution(attr_real, fb, "real")
    cf_oracle = counterfactual_attribution(attr_oracle, fb, "oracle")
    val_real = validate_attribution(cf_real, truth)
    val_oracle = validate_attribution(cf_oracle, truth)
    sh = shap_analysis(prod["model"], fb)
    pp_mean = float(np.mean(list(val_real["per_person"].values())))
    print(f"      attribution fidelity (self-logged): per-person Spearman={pp_mean:.3f} "
          f"| latent-input check={val_oracle['overall']:.3f}")

    # stash the population fingerprint in the artifact (used as the shrinkage
    # prior when personalising on small real logs), then persist
    prod["population_fingerprint"] = cf_real["overall"][["cause", "share"]].copy()
    save_artifacts(prod, os.path.join(MODEL, "timeslip_model.joblib"))
    print(f"      saved production model -> {os.path.join(MODEL, 'timeslip_model.joblib')}")

    print("[8/9] interventions + real-human-data validation ...")
    from timeslip.interventions import run_policies
    interventions = run_policies()
    print(f"      intervention 'all' -> slips {interventions.loc['all','slips_change_%']:+.0f}%, "
          f"time {interventions.loc['all','time_change_%']:+.0f}%")
    realworld = None
    try:
        from timeslip.realworld import validate as rw_validate
        realworld = rw_validate()
        print(f"      real-data (Kane 2017) sign agreement "
              f"{realworld['sign_agreement']:.0%}, rank corr "
              f"{realworld['rank_corr_vs_sim']:.2f}")
    except Exception as ex:
        print(f"      (real-data validation skipped: {ex})")

    ctx = dict(
        minutes=minutes, episodes=episodes, personas=personas,
        horizon=fb["horizon_min"], n_people=int(len(personas)),
        n_days=int(minutes["day"].nunique()), n_minutes=int(len(minutes)),
        n_slips=int(len(episodes)),
        eval_cold=cold, eval_pers=pers, calib=cal, notif_base=notif_base,
        learning_curve=lc, recovery=rec, odds=odds,
        spells=spells, km=km, hazard=haz, vigilance=vc,
        truth=truth, cf_real=cf_real, cf_oracle=cf_oracle,
        val_real=val_real, val_oracle=val_oracle, attr_per_person_mean=pp_mean,
        shap=sh, interventions=interventions, realworld=realworld,
    )

    print("[9/9] writing data, figures, reports ...")
    # the full minute log is huge at cohort scale -> save a sample for inspection
    minutes.sample(n=min(20000, len(minutes)), random_state=C.GLOBAL_SEED) \
        .sort_values(["pid", "day", "clock_min"]) \
        .to_csv(os.path.join(DATA, "minutes_sample.csv"), index=False)
    episodes.to_csv(os.path.join(DATA, "episodes.csv"), index=False)
    personas.to_csv(os.path.join(DATA, "personas.csv"), index=False)
    rec["table"].to_csv(os.path.join(DATA, "recovery_table.csv"), index=False)
    if haz.get("available"):
        haz["summary"].to_csv(os.path.join(DATA, "hazard_ratios.csv"), index=False)
    odds["odds_ratios"].to_csv(os.path.join(DATA, "odds_ratios.csv"), index=False)
    cf_real["per_person"].to_csv(os.path.join(DATA, "fingerprints_self_logged.csv"), index=False)
    truth.to_csv(os.path.join(DATA, "fingerprints_ground_truth.csv"))
    lc.to_csv(os.path.join(DATA, "learning_curve.csv"), index=False)
    interventions.to_csv(os.path.join(DATA, "interventions.csv"))
    if realworld is not None:
        realworld["table"].to_csv(os.path.join(DATA, "realworld_validation.csv"), index=False)
    if sh.get("available"):
        sh["global_importance"].to_csv(os.path.join(DATA, "shap_importance.csv"), index=False)

    summary = dict(
        n_minutes=int(len(minutes)), n_slips=int(len(episodes)),
        n_people=int(len(personas)), n_days=int(minutes["day"].nunique()),
        horizon_min=int(fb["horizon_min"]),
        cold_start=cold["metrics"], personalised=pers["metrics"],
        calibration_brier_before=cal["brier_before"],
        calibration_brier_after=cal["brier_after"],
        notif_only_roc=float(roc_auc_score(notif_base["y"], notif_base["p"])),
        recovery_spearman=rec["spearman"], recovery_sign_agreement=rec["sign_agreement"],
        attribution_spearman_self_logged_flat=val_real["overall"],
        attribution_spearman_self_logged_per_person=pp_mean,
        attribution_spearman_oracle=val_oracle["overall"],
        hazard_auc=haz.get("auc"),
        per_person_auc_cold_median=float(cold["per_person_auc"].median()),
        per_person_auc_pers_median=float(pers["per_person_auc"].median()),
        realworld_sign_agreement=(None if realworld is None
                                  else realworld["sign_agreement"]),
        realworld_rank_corr=(None if realworld is None
                             else realworld["rank_corr_vs_sim"]),
        intervention_all_time_change_pct=float(interventions.loc["all", "time_change_%"]),
    )
    with open(os.path.join(DATA, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    report.generate_all(ctx, FIG, REP)

    print(f"\nDone in {time.time()-t0:.0f}s. Outputs in ./outputs/")
    print(f"  model   -> {MODEL}  (timeslip_model.joblib)")
    print(f"  reports -> {REP}  (start with findings.md)")


if __name__ == "__main__":
    main()
