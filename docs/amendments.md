# Amendments log (spec frozen 18 Oct 2026; changes recorded with reasons)

| # | Change vs. original docs / Phase 1 spec | Reason | Status |
|---|---|---|---|
| A1 | Cohort 30,000 -> 50,000 (planned) | Thin subgroup event counts; generation is free | approved |
| A2 | Smokers oversampled to ~35%; ~65% of smokers conceal | Needed for 65/15/12/8 to be feasible | approved |
| A3 | Arm A multiplicative loadings; truthful-trained primary models; one row per applicant | Spec | approved |
| A4 | Baseline level A0 re-solved per gamma (mean p10 fixed at 5.5%) | gamma should change the SHAPE of risk, not the level; Arm A gets a one-scalar recalibration too | disclosed Phase 2 |
| A5 | Smoker HR 2.4 (docs: 1.8-2.0), diabetes HR 1.80 (docs: 1.57) | Literature check: Jha 2013 / Cho 2024; ERFC 2011. Docs' figures not verifiable | disclosed Phase 2 |
| A6 | CVD prevalence intercept raised (2.0% -> 3.4%) | Diagnosis-concealment pool too thin | done |
| A7 | Hidden gamma frailty Z (mean 1, var theta=0.25, assumed) added; true_p10 = x-conditional (frailty-marginal) probability | Option C chosen by user: risk not on the form (e.g. depression) without inventing columns | approved |
| A8 | Leakage guard: models may only see feature_view(df, 'true'|'declared') | New hidden columns (frailty_z, p10_given_frailty) raise leakage risk | done |
| A9 | "65% truthful = control group" replaced by within-person paired comparison; honest smokers = matched control for smoking concealers | Concealers are riskier at baseline by eligibility, so between-group comparison is confounded | approved |
| A10 | Footprint settings LOCKED (SBP +14 if HTN, HbA1c +1.6 if diabetic, cholesterol +6 if smoker). Probe AUCs (true labels, upper bounds): smoking 0.61, HTN 0.85, diabetes 0.96, CVD 0.69 | Spec open item; contrast in detectability by type is intentional | approved, locked |
| A11 | Full grid built: 3 seeds x 5 gamma x 50,000; `concealed_fields` uses 'none' not blank | Blank strings read back as NaN | done |
| A12 | Arm A fixed pre-results: loadings rounded to 0.25; measured SBP>=140 / HbA1c>=6.5 override declarations; decisions Standard/Rated/Decline (Verify deferred to Phase 6); premium 1.25 x p x SA / 10 | Realistic underwriter behaviour; same columns as Arm B | done |
| A13 | Recalibration scalar k ~ 0.53: CDC-shaped table is population-average, not select | Disclosed limitation; k fitted on train only | done |
| A14 | Phase 5 protocol fixed pre-run (grid, shared calibration rule, truthful-trained, clinical-band flags in GLM/Cox). Exploratory check (validation only, seed 2026): tuning XGBoost by 3-fold CV on train instead of validation helped modestly at gamma 0 (0.0127) and not at gamma 1 (0.0155); XGBoost stayed behind the GLM, so the protocol stands | A priori sample-size concern tested, not result-driven | done |
| A15 | Phase 5 preview: XGBoost does not beat GLM/Cox on truthful data at any gamma; truthful-trained models are not robust to concealment. Declared-trained regime (Regime 2) and a training-size learning curve are PROPOSED additions | See Phase 5 report | approved |
| A16 | A15 executed: declared-trained regime (R2) for Arm A (k on declared view) and all Arm B models; learning curve on a separate 335k cohort (seed 3000). Results in docs/phase5b_findings.md. Proposed next: Arm A+ (loadings re-estimated by experience ratio per declared class) for R2 | See findings | Arm A+ approved (A17) |
| A17 | Arm A+ (Regime 2): fixed loadings, one scalar per declared class (smoker x any diagnosis), A/E on train. Result: GLM no longer clearly beats Arm A+ (paired diffs mixed); XGBoost worse than Arm A+ in 6/6 | Regime 2 comparison against a minimal Arm A was a strawman | done |
| A18 | Verification: rule trigger (Arm A) vs label-free consistency model (Arm B), compared at matched number verified, with random baseline; purity enforced by tests; hidden loss as arm-independent value measure | Spec Phase 5/6 | done |
| A19 | Game inputs: d_pi over the term; P* = d_pi/(r v) with P = total loss on detection; c* per flagged applicant with re-weighting to 10% concealer share; Regime 1 pricing | Preliminary; marginal/by-SA-band analysis deferred to Phase 8 | done (preliminary) |
| A20 | Phase 7 executed once on the test split exactly as frozen (docs/phase7_protocol.md); no deviations. Note: F4 recall is identical across gamma by construction; applicant-bootstrap CIs ignore model-fit variance (3 seeds) | Disclosure | done |
| A21 | Phase 8 done: audit game with closed-form deterrence threshold P* = dpi/(r v) (L = total loss on detection) and mixed equilibrium sigma* = P*/L, theta* = c f / (r(v dpi - c) + c f); insurer thresholds c* by strategy, rank, sum assured, prevalence, gamma, with bootstrap CIs. Computed from test-split predictions | Spec Phase 8 | done |
| A22 | Phase 9 calculator built from exported light artifacts (XGBoost JSON, GLM coefficients, aggregates); not offered: Cox. Constant-hazard extrapolation compared with the exact DGP term probability; finding: 20-year premium understated ~28% in the example | Spec Phase 9 | done |
