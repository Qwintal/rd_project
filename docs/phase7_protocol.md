# Phase 7 protocol - FROZEN 18 Oct 2026, before any test-split result was computed

**Data:** test split only (15,000 applicants per dataset; 3 seeds x 5 gamma; pooled over seeds = 45,000 applicants per gamma). Models and triggers are the ones trained in Phases 4-6; nothing is re-tuned.
Arm A+ and the consistency trigger are re-fitted deterministically from the TRAIN split (they were not saved).
**Everything is reported, whichever way it falls.**

## Regimes and views
R1 = truthful-trained (scored on truthful view = Exp A, and declared view = Exp B). R2 = declared-trained (scored on declared view; includes Arm A+).

## Primary metrics and pre-declared comparisons (each at each of the 5 gamma values)
- **RQ1 (family F1, 20 tests):** RMSE of predicted p10 vs true_p10, R1, truthful view. Comparisons: XGBoost-ArmA, GLM-ArmA, Cox-ArmA, XGBoost-GLM.
- **RQ2 (family F2, 20 tests):** premium MAE vs true-p premium (same formula), R1, truthful view. Same four comparisons. Wilcoxon signed-rank on per-applicant absolute errors alongside the bootstrap.
- **RQ3 (family F3a, 20 tests):** (i) difference-in-differences of RMSE: [RMSE(ArmA, declared) - RMSE(model, declared)] - [RMSE(ArmA, truthful) - RMSE(model, truthful)], model in {XGBoost, GLM}, R1;
  positive = ML advantage grows under concealment, negative = shrinks/reverses. (ii) mean underpricing % on concealers (declared view), model minus Arm A, model in {XGBoost, GLM}, R1.
  (10 tests each for (i) and (ii).)
- **RQ3, declared-trained (family F3b, 55 tests):** declared-view RMSE: R2 GLM-ArmA+, R2 XGBoost-ArmA+, R2 XGBoost-GLM, R2-vs-R1 for Arm A, GLM, Cox, XGBoost; R1 declared XGBoost-ArmA, GLM-ArmA, Cox-ArmA, XGBoost-GLM... (11 comparisons x 5 gamma).
- **RQ3b detection (family F4, 10 tests):** consistency trigger minus rule trigger at the SAME number verified: hidden-loss-caught % and recall.
- **RQ4:** deterministic from pipeline outputs (Phase 8); Experiment E scenario table here.

## Statistics
Paired nonparametric bootstrap resampling APPLICANTS, B = 1000, same resampling indices for all statistics; 95% percentile CI; p-value from the bootstrap-SE z statistic (percentile p-values cannot reach Holm thresholds at B = 1000).
Holm correction within each family at alpha = 0.05 (two-sided). Per-seed sign agreement reported for F1/F2. DeLong for secondary AUC differences (own family, Holm). Descriptive tables (no tests): calibration, bands, portfolio, Exp C, Exp D.

## Definitions fixed now
- Premium = 1.25 x p x SA / 10 per year (ground truth uses true_p10). Over/underpriced: model premium > 1.10x / < 0.90x true premium.
- Risk bands by true_p10: low < 3%, medium 3-8%, high >= 8%.
- Portfolio loss ratio = sum(true_p10 x SA / 10) / sum(premium); a correctly priced book = 0.80. Re-weighted to 10% concealers (weights 0.10/s for concealers, 0.90/(1-s) for honest, s = cohort concealer share). Also reported unweighted.
- Hidden loss = SA x (true_p10 - p the true hazard gives the declared record). Matched budget = top-k of the consistency score with k = number the rule flags, per dataset.
- Experiment E value of verifying i = 10 x [premium(true view) - premium(declared view)] under the arm's own R1 pricing (zero for honest applicants); net = recovered - c x verified; c in {5k, 25k, 100k, 250k} INR; weights as above.
- Not done (stated limitations): ground-truth underwriting decisions (no guideline exists in the DGP) so no decision confusion matrix; decisions are described by mix and change under concealment only.
