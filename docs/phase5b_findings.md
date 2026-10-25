# Phase 5b findings (amendment A15: declared-trained regime + learning curve)

All numbers are on the VALIDATION split (regime comparison) or a separate 20,000-applicant evaluation set (learning curve). The project test split is untouched.

## 1. Regime 1 (truthful-trained) vs Regime 2 (declared-trained), scored on the DECLARED view, mean of 3 seeds
bias% = mean 100*(p - true)/true (positive = overcharged); under% = mean 100*(true - p)/true (positive = underpriced);
portfolio_ratio = sum(p*SA)/sum(true*SA), UNWEIGHTED (cohort has ~35% smokers, ~35% concealers; realistic re-weighting comes in Phase 7).
|   gamma | model   | regime   |   rmse_all |   bias%_honest_nonsmoker |   bias%_honest_smoker |   under%_diagnosis |   under%_smoking |   under%_multiple |   portfolio_ratio |
|--------:|:--------|:---------|-----------:|-------------------------:|----------------------:|-------------------:|-----------------:|------------------:|------------------:|
|       0 | arm_a   | R1       |      0.036 |                   -0.506 |                 5.002 |             19.244 |           57.122 |            65.347 |             0.729 |
|       0 | arm_a   | R2       |      0.033 |                   39.896 |                46.097 |            -13.111 |           39.755 |            51.481 |             1.018 |
|       0 | cox     | R1       |      0.038 |                    2.815 |                 2.562 |             29.217 |           56.601 |            69.59  |             0.704 |
|       0 | cox     | R2       |      0.03  |                   48.939 |                 3.249 |            -16.866 |           35.107 |            47.819 |             1.007 |
|       0 | glm     | R1       |      0.038 |                   -0.16  |                 3.162 |             31.016 |           57.723 |            70.253 |             0.698 |
|       0 | glm     | R2       |      0.03  |                   47.181 |                 3.908 |            -16.46  |           35.668 |            47.899 |             1.007 |
|       0 | xgb     | R1       |      0.039 |                   10.975 |                 4.237 |             20.285 |           53.325 |            66.541 |             0.734 |
|       0 | xgb     | R2       |      0.032 |                   55.462 |                 2.329 |            -20.506 |           32.952 |            47.069 |             1.006 |
|       1 | arm_a   | R1       |      0.043 |                    5.842 |                10.299 |             16.268 |           55.263 |            68.274 |             0.723 |
|       1 | arm_a   | R2       |      0.04  |                   48.83  |                53.547 |            -17.308 |           37.125 |            55.561 |             1.01  |
|       1 | cox     | R1       |      0.046 |                   -1.42  |                 5.057 |             31.696 |           59.368 |            74.429 |             0.664 |
|       1 | cox     | R2       |      0.037 |                   52.282 |                 4.498 |            -23.311 |           34.982 |            51.806 |             1.003 |
|       1 | glm     | R1       |      0.046 |                   -4.524 |                 5.536 |             33.552 |           60.564 |            75.059 |             0.658 |
|       1 | glm     | R2       |      0.037 |                   50.291 |                 5.06  |            -22.931 |           35.665 |            51.891 |             1.003 |
|       1 | xgb     | R1       |      0.046 |                    9.468 |                 4.918 |             23.646 |           54.639 |            71.46  |             0.695 |
|       1 | xgb     | R2       |      0.038 |                   58.068 |                 2.282 |            -26.934 |           32.9   |            51.033 |             1.002 |

Paired per-seed differences in declared-view RMSE (negative = first model better), 3 seeds each:
- GLM R2 vs GLM R1: better in all 6 (gamma 0 and 1), by 0.008-0.010.
- GLM R2 vs Arm A R2: better in all 6. XGBoost R2 vs Arm A R2: better in all 6. XGBoost R2 vs GLM R2: worse in all 6.

Caveats: (i) Arm A in Regime 2 is deliberately minimal (one recalibration scalar). An actuary would re-estimate loadings by experience (A/E by declared class),
so the Regime 2 gap to Arm A is probably overstated. (ii) The honest-applicant overcharge is inflated by the design choice that ~65% of smokers conceal.

## 2. Learning curve (separate cohort, seed 3000; fixed evaluation set; Regime 1; RMSE vs true p10, truthful view)

### gamma 0.0 (mean over replicate draws; 3 reps up to 90k, 1 rep at 270k)
        train_claims  rmse_arm_a  rmse_glm  rmse_cox  rmse_xgb
n                                                             
5000             268      0.0089    0.0246    0.0243    0.0270
15000            834      0.0089    0.0147    0.0145    0.0182
30000           1685      0.0090    0.0087    0.0086    0.0128
90000           4992      0.0088    0.0051    0.0051    0.0097
270000         15032      0.0088    0.0037    0.0037    0.0069

### gamma 1.0 (mean over replicate draws; 3 reps up to 90k, 1 rep at 270k)
        train_claims  rmse_arm_a  rmse_glm  rmse_cox  rmse_xgb
n                                                             
5000             266      0.0160    0.0262    0.0258    0.0324
15000            832      0.0153    0.0166    0.0142    0.0193
30000           1675      0.0152    0.0119    0.0113    0.0150
90000           4989      0.0152    0.0092    0.0085    0.0108
270000         15023      0.0152    0.0084    0.0077    0.0081

Reading: Arm A has no learning, so its error is flat. GLM/Cox pass Arm A between 15k and 30k training applicants at both gamma values.
XGBoost passes Arm A near 30k-90k at gamma 1 and between 90k and 270k at gamma 0; it only ties GLM/Cox at 270k (gamma 1, one replicate) and never beats them at gamma 0.
The project's 30k training size sits at the crossing point, which is why Phase 5 results looked ambiguous.
