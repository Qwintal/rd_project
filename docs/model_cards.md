# Model cards - Arm B (all trained on TRUTHFUL records of the train split; test split untouched)

## Shared
- **Inputs:** the 16 columns returned by `feature_view(df, 'true')` (age, sex, occupation class, BMI, SBP, cholesterol, HbA1c, family history, alcohol,
  hazardous hobby, log10 sum assured, smoker, pack-years, diabetes, hypertension, CVD). No hidden column can reach a model.
- **Target:** claim within 10 years (GLM, XGBoost) or time-to-claim censored at 10 years (Cox). Predicts the x-conditional probability, matching `true_p10`.
- **Training size:** 30,000 applicants (60%) per dataset, about 1,650 claims. **Validation:** 5,000 applicants (about 275 claims), used for early stopping,
  XGBoost grid selection and calibration selection, so validation numbers are mildly optimistic for XGBoost.
- **Calibration:** none / Platt / isotonic chosen by 5-fold CV log loss on validation (ties within 0.0003 go to the simpler method).
  Chosen in 45 of 45 model fits: **45 'none'** - the models are already calibrated on truthful data.
- **Grid (fixed before any run):** depth {2,3,4} x min_child_weight {10,40} x lambda {1,10}; eta 0.05; <=1000 trees, early stopping 50; subsample/colsample 0.8.
- **Scope of claims:** results are conditional on the assumed DGP (hazard ratios, five assumed interactions, frailty variance 0.25) and on a 10-year window.

## 1. Logistic GLM - interpretable additive baseline
Main effects plus clinical-band flags (BMI>=30, SBP>=140, cholesterol>=240, HbA1c>=6.5). **No interactions.** Statsmodels Binomial GLM; coefficient tables in `glm_coef_seed*_gamma*.csv`.
Example (seed 2026, gamma 0.5), log-odds coefficients:
|                |   coef |    se |   p_value |
|:---------------|-------:|------:|----------:|
| const          | -4.038 | 0.068 |     0     |
| age_dec        |  0.692 | 0.05  |     0     |
| male           |  0.437 | 0.059 |     0     |
| occ3           |  0.118 | 0.068 |     0.08  |
| occ4           |  0.296 | 0.085 |     0     |
| bmi_5          |  0.018 | 0.047 |     0.696 |
| bmi_ge_30      |  0.28  | 0.095 |     0.003 |
| sbp_10         | -0.016 | 0.03  |     0.587 |
| sbp_ge_140     |  0.004 | 0.103 |     0.966 |
| chol_30        |  0.024 | 0.031 |     0.45  |
| chol_ge_240    |  0.093 | 0.096 |     0.334 |
| hba1c_c        | -0.049 | 0.071 |     0.488 |
| hba1c_ge_6_5   | -0.112 | 0.192 |     0.558 |
| family_history |  0.163 | 0.062 |     0.008 |
| alc_moderate   |  0.046 | 0.059 |     0.439 |
| alc_heavy      |  0.271 | 0.08  |     0.001 |
| hobby          |  0.02  | 0.116 |     0.864 |
| log10_sa_c     |  0.088 | 0.072 |     0.224 |
| smoker         |  0.984 | 0.093 |     0     |
| pack_years_10  | -0.02  | 0.032 |     0.519 |
| diabetes       |  0.782 | 0.143 |     0     |
| hypertension   |  0.319 | 0.068 |     0     |
| cvd            |  0.668 | 0.086 |     0     |

## 2. Cox PH - reference rung
Same design as the GLM, penalizer 0.001, time-to-claim with administrative censoring at 10 years; 10-year probability = 1 - exp(-H(10|x)).
Known mismatch: hidden frailty makes marginal hazards only approximately proportional.

## 3. XGBoost - primary model (raw features; learns thresholds and interactions itself)
Selected hyperparameters per (seed, gamma):
|              |   max_depth |   min_child_weight |   reg_lambda |   best_iteration |
|:-------------|------------:|-------------------:|-------------:|-----------------:|
| (2026, 0.0)  |           4 |                 40 |            1 |              114 |
| (2026, 0.25) |           2 |                 40 |            1 |              165 |
| (2026, 0.5)  |           3 |                 40 |            1 |              134 |
| (2026, 0.75) |           2 |                 10 |            1 |              172 |
| (2026, 1.0)  |           2 |                 40 |            1 |              226 |
| (2027, 0.0)  |           4 |                 10 |           10 |               82 |
| (2027, 0.25) |           3 |                 10 |            1 |               82 |
| (2027, 0.5)  |           3 |                 10 |            1 |               83 |
| (2027, 0.75) |           3 |                 10 |           10 |               69 |
| (2027, 1.0)  |           2 |                 40 |           10 |              124 |
| (2028, 0.0)  |           2 |                 40 |            1 |              141 |
| (2028, 0.25) |           2 |                 10 |           10 |              180 |
| (2028, 0.5)  |           2 |                 40 |            1 |              186 |
| (2028, 0.75) |           2 |                 40 |           10 |              177 |
| (2028, 1.0)  |           2 |                 10 |            1 |              173 |

## Known weaknesses
- Truthful-trained models never see a lie, so they lean on declared flags and under-use the biomarker footprint (Regime 1). A declared-trained regime is pending a decision.
- About 1,650 training claims is thin for tree models; see docs/armb_val_preview.txt and docs/exploratory_phase5 notes in amendments.md.
- SHAP is descriptive only; the small interaction values at gamma 1 do not prove XGBoost recovered the true interaction structure.
