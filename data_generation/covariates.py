"""Correlated applicant covariates (sequential conditional sampling).

Order: age, sex -> occupation -> smoker/pack-years -> BMI -> family history
-> hypertension -> diabetes -> CVD -> alcohol/hobby -> biomarkers -> sum assured.
Loosely shaped after NHANES-style relationships (age<->BP, BMI<->HTN/diabetes,
smoking<->pack-years); NOT fitted to NHANES microdata (not accessible here).
"""
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def draw_covariates(cfg, rng):
    n = cfg["n_applicants"]
    fp = cfg["footprint"]

    age = rng.integers(cfg["age"]["min"], cfg["age"]["max"] + 1, n)
    male = rng.random(n) < cfg["p_male"]
    sex = np.where(male, "M", "F")

    occ = rng.choice([1, 2, 3, 4], n, p=[0.35, 0.35, 0.20, 0.10])

    p_s = np.where(male, cfg["p_smoker_male"], cfg["p_smoker_female"])
    smoker = (rng.random(n) < p_s).astype(int)
    start = rng.integers(16, 26, n)
    ppd = np.clip(rng.lognormal(np.log(0.7), 0.4, n), 0.1, 3.0)
    pack_years = np.where(smoker == 1, np.round(np.maximum(age - start, 1) * ppd, 1), 0.0)

    bmi = np.clip(np.exp(rng.normal(np.log(25.5) + 0.002 * (age - 50), 0.16)), 15, 50)
    bmi = np.round(bmi, 1)
    family = (rng.random(n) < 0.20).astype(int)

    htn = (rng.random(n) < _sigmoid(-2.2 + 0.07 * (age - 40) + 0.07 * (bmi - 25) + 0.2 * smoker)).astype(int)
    dm = (rng.random(n) < _sigmoid(-3.3 + 0.045 * (age - 40) + 0.12 * (bmi - 25) + 0.3 * family)).astype(int)
    cvd = (rng.random(n) < _sigmoid(-4.4 + 0.07 * (age - 40) + 0.5 * smoker + 0.4 * htn
                                    + 0.5 * dm + 0.4 * male)).astype(int)

    u = rng.random(n)
    alc = np.where(male,
                   np.select([u < 0.45, u < 0.83], [0, 1], 2),
                   np.select([u < 0.75, u < 0.97], [0, 1], 2))
    hobby = (rng.random(n) < 0.05).astype(int)

    # insurer-measured biomarkers: carry the concealment footprint, never concealable
    chol = np.round(np.clip(rng.normal(190 + 0.4 * (age - 40) + 1.8 * (bmi - 25)
                                       + fp["chol_shift_smoker"] * smoker + 5 * htn, 32), 110, 380))
    sbp = np.round(np.clip(rng.normal(112 + 0.5 * (age - 40) + 0.8 * (bmi - 25)
                                      + fp["sbp_shift_htn"] * htn, 10), 85, 220))
    hb_mean = 5.3 + 0.01 * (age - 40) + 0.02 * (bmi - 25) + fp["hba1c_shift_dm"] * dm
    hba1c = np.round(np.clip(rng.normal(hb_mean, np.where(dm == 1, 0.9, 0.3)), 4.0, 14.0), 1)

    bands = np.array([1e6, 2.5e6, 5e6, 1e7, 2e7])
    probs = {1: [.05, .15, .30, .30, .20], 2: [.10, .25, .35, .20, .10],
             3: [.25, .35, .25, .10, .05], 4: [.35, .35, .20, .07, .03]}
    sa = np.array([rng.choice(bands, p=probs[o]) for o in occ])

    return pd.DataFrame({
        "age": age, "sex": sex, "occupation_class": occ, "bmi": bmi, "sbp": sbp,
        "cholesterol": chol, "hba1c": hba1c, "family_history": family,
        "alcohol": alc, "hazardous_hobby": hobby, "sum_assured": sa.astype(int),
        "true_smoker": smoker, "true_pack_years": pack_years,
        "true_diabetes": dm, "true_hypertension": htn, "true_cvd": cvd,
    })
