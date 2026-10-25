# Arm A - traditional actuarial engine (fixed 18 Oct 2026, before any results were seen)

**Pipeline:** base hazard (5-yr age band x sex) x product of loadings x recalibration scalar k -> p10 = 1 - exp(-10 k h M)
-> decision table -> premium P = 1.25 * p10 * SA / 10.
Code: `actuarial_engine/arm_a.py`. Tests: `tests/test_arm_a.py` (11 pass).

## Base table (annual hazard, standard-life reference; log-linear fit to CDC 2023 US male q_x at ages 40/49/50/59, evaluated at band mean age)
| Band | Male | Female (x0.60, assumed) |
|---|---|---|
| 40-44 | 0.00335 | 0.00201 |
| 45-49 | 0.00460 | 0.00276 |
| 50-54 | 0.00632 | 0.00379 |
| 55-60 | 0.00895 | 0.00537 |

SOA 2015 VBT was not accessible here; the table is CDC-shaped and population-average, **not** a select insured-lives table.

## Loadings (multiplicative, no interactions, no cap) - rule: literature/assumed value rounded to nearest 0.25
smoker 2.50 | hypertension 1.25 | BMI>=30 1.25 | diabetes 1.75 | CVD 2.00 | heavy alcohol 1.25 | occupation class 4 1.25 |
family history 1.25 | cholesterol>=240 1.25 | occupation class 3 1.00 | hazardous hobby 1.00.
Measured values override declarations: SBP>=140 => hypertension loading; HbA1c>=6.5 => diabetes loading.
Pack-years are not used.

## Decisions (combined multiplier M)
Standard: M <= 1.25 | Rated: 1.25 < M <= 5.0 | Decline: M > 5.0. "Verify" (rule-based inconsistency referral) is built in Phase 6.

## Premium
P = (1 + 0.15 expense + 0.10 profit) x p x SA / 10 (assumed loadings). Ground truth = same formula with true_p10.
Technical premium is computed for everyone, including declined applicants, so pricing accuracy can be compared.
No discounting, lapse or reserves.

## Recalibration
One scalar k fitted on the TRAIN split (truthful view) so mean predicted p10 = observed train claim rate. Fitted k ~ 0.53 at every gamma:
the population-average table plus loadings over-predicts ~1.9x, so k removes the double counting (standard tables describe standard lives).

## Disclosures
1. For minor factors with no literature source (alcohol, occupation, family history, cholesterol) Arm A uses the DGP's assumed values after rounding: mild insider knowledge, slightly favours Arm A.
2. Arm A has no interaction terms by design: this is the simplification of real substandard-rating practice the report must state.
3. Age slope comes from the same CDC fit as the DGP's baseline; only the coarse bands, rounding and constant-hazard-within-term differ.

## Smoke test (seed 2026, TRAIN fit, VALIDATION split only; test split untouched until Phase 7)
See `arm_a_smoke.txt`. RMSE vs true_p10 on truthful records rises with gamma (0.0089 -> 0.0145) as interactions grow; on declared
(concealed) records it is 0.035-0.042 at every gamma, so the lie matters ~3-4x more than interactions for Arm A.
