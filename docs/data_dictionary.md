# Data dictionary - `data/cohort_seed{2026,2027,2028}_gamma{0,0.25,0.5,0.75,1.0}.csv.gz`

50,000 rows each, **one row per applicant** (the truthful and declared views sit side by side, so the pairing and the
leakage rule are structural). Within a seed, the same applicants and the same split appear at every gamma.
Tags: **(V)** verified against a cited source, **(A)** assumed/illustrative, **(D)** design choice. Prevalences are design choices, not fitted to NHANES.

**Who may see what.** Risk models and rules may use ONLY `feature_view(df, 'true'|'declared')` columns (the OBSERVED group plus the
true_ or declared_ concealable fields). Everything marked HIDDEN is for evaluation only.

| Column | Type / unit | Role | Description |
|---|---|---|---|
| applicant_id | str | ID | `A000000`-style unique key |
| split | train/val/test | bookkeeping | 60/10/30, by applicant, stratified by concealment_type; identical across gamma |
| age | int, years | OBSERVED | Uniform 40-60 (D) |
| sex | M/F | OBSERVED | 55% male (A) |
| occupation_class | 1-4 | OBSERVED | 35/35/20/10%; 4 = hazardous (A) |
| bmi | kg/m2 | OBSERVED (insurer-measured) | Lognormal around 25.5, drifts up with age; cannot be concealed |
| sbp | mmHg | OBSERVED (measured) | Rises with age/BMI; **+14 if truly hypertensive** (footprint, D) |
| cholesterol | mg/dL | OBSERVED (measured) | Rises with age/BMI; **+6 if smoker** (weak footprint, D) |
| hba1c | % | OBSERVED (measured) | Rises with age/BMI; **+1.6 and wider spread if diabetic** (footprint, D) |
| family_history | 0/1 | OBSERVED | 20% (A) |
| alcohol | 0/1/2 | OBSERVED | none/moderate/heavy; heavier in men (A) |
| hazardous_hobby | 0/1 | OBSERVED | 5% (A) |
| sum_assured | INR | OBSERVED | Bands 10L to 2Cr, higher for lower-hazard occupation classes (A) |
| true_smoker | 0/1 | TRUE concealable | Current smoker; ~35% oversampled (D) |
| true_pack_years | float | TRUE concealable | 0 for non-smokers; (age - start age) x packs/day |
| true_diabetes | 0/1 | TRUE concealable | Underlying condition; rises with age, BMI, family history |
| true_hypertension | 0/1 | TRUE concealable | Underlying condition; rises with age, BMI, smoking |
| true_cvd | 0/1 | TRUE concealable | ~4.6%; rises with age, smoking, HTN, diabetes, male. No biomarker footprint |
| declared_smoker | 0/1 | DECLARED | What the applicant says; equals true_smoker unless concealed (false value, never blank) |
| declared_pack_years | float | DECLARED | Set to 0 when smoking is concealed |
| declared_diabetes / _hypertension / _cvd | 0/1 | DECLARED | Set to 0 for the one concealed diagnosis |
| concealment_type | truthful/smoking/diagnosis/multiple | HIDDEN (label) | 65/15/12/8 (A). multiple = smoking + one diagnosis |
| concealed_fields | str | HIDDEN (label) | `none`, `smoker`, `hypertension`, `smoker\|diabetes`, ... |
| frailty_z | float | HIDDEN truth | Gamma frailty, mean 1, variance 0.25 (A). Risk not on the form (e.g. depression) |
| true_p10 | prob | HIDDEN truth | **Primary evaluation target:** best possible 10-yr claim probability given every column (frailty-marginal), from the true record |
| p10_given_frailty | prob | HIDDEN truth | Individual probability given frailty_z; this is what generates claims |
| event_time | years | HIDDEN (latent) | Uncensored time to claim; do NOT train on it (use observed_time) |
| observed_time | years | OUTCOME | min(event_time, 10): administrative censoring at 10 years |
| claim_10yr | 0/1 | OUTCOME | 1 if claim within 10 years (training label on the train split) |
| loss_amount | INR | OUTCOME | claim_10yr x sum_assured |
| true_expected_loss | INR | HIDDEN truth | true_p10 x sum_assured (ground-truth expected loss for premium adequacy) |
| gamma | float | dataset parameter | Interaction strength, 0 to 1 |

## Hazard (what generated the truth)
`log h(a+t) = log A0 + b(age-40) + b t + log(sexfactor) + sum_k beta_k x_k + gamma * sum_jk beta_jk (x_j x_k)`, times frailty Z.
Main-effect hazard ratios: smoker 2.4, hypertension 1.30, BMI>=30 1.26, diabetes 1.80 (V/D, see config); other effects (A).
Five interaction terms (all assumed): smoker x age, smoker x HTN, smoker x BMI, HTN x BMI, smoker x HTN x BMI.
A0 is re-solved for each gamma so the mean true_p10 stays 5.5%.
