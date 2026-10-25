"""Hazard-based DGP.

    log h(a + t) = log A0 + b*(age - 40) + b*t + log(sex_factor)
                   + sum_k beta_k x_k + gamma * sum_jk beta_jk (x_j x_k)

One hazard function drives: 10-yr claim probability, time-to-claim (Cox),
censoring, and the exact true probability for ANY term (used to measure the
calculator's constant-hazard extrapolation error).
"""
import numpy as np
from scipy.optimize import brentq


def fit_gompertz_b():
    """Slope of log-mortality in age, fitted to CDC NVSR 74-06 US male 2023 q_x
    at ages 40, 49, 50, 59 (0.003176, 0.004859, 0.005182, 0.010581).
    Population-average curve: used for SHAPE only; the level is re-calibrated."""
    ages = np.array([40, 49, 50, 59], dtype=float)
    q = np.array([0.003176, 0.004859, 0.005182, 0.010581])
    b, _ = np.polyfit(ages, np.log(q), 1)
    return float(b)


def linear_predictor(df, cfg, gamma):
    """Log-hazard ratio vs. the reference applicant (all risk flags 0).
    Uses TRUE conditions (true_*), never the declared view."""
    hr = cfg["hazard_ratios"]
    ib = cfg["interaction_beta"]
    L = lambda k: np.log(hr[k])

    s = df["true_smoker"].to_numpy(float)
    h = df["true_hypertension"].to_numpy(float)
    d = df["true_diabetes"].to_numpy(float)
    c = df["true_cvd"].to_numpy(float)
    bmi30 = (df["bmi"].to_numpy() >= 30).astype(float)
    chol = (df["cholesterol"].to_numpy() >= 240).astype(float)
    occ = df["occupation_class"].to_numpy()
    alc = (df["alcohol"].to_numpy() == 2).astype(float)

    lp = (L("smoker") * s + L("hypertension") * h + L("bmi_ge_30") * bmi30
          + L("diabetes") * d + L("cvd") * c + L("alcohol_heavy") * alc
          + L("hazardous_hobby") * df["hazardous_hobby"].to_numpy(float)
          + L("occupation_class_3") * (occ == 3) + L("occupation_class_4") * (occ == 4)
          + L("family_history") * df["family_history"].to_numpy(float)
          + L("chol_ge_240") * chol)

    age_dec = (df["age"].to_numpy(float) - 50.0) / 10.0
    inter = (ib["smoker_x_age_per_decade"] * s * age_dec
             + ib["smoker_x_htn"] * s * h
             + ib["smoker_x_bmi"] * s * bmi30
             + ib["htn_x_bmi"] * h * bmi30
             + ib["smoker_x_htn_x_bmi"] * s * h * bmi30)
    return lp + gamma * inter


def _scale(df, cfg, b):
    """Per-applicant baseline factor: exp(b*(age-40)) * sex factor."""
    sexf = np.where(df["sex"].to_numpy() == "F", cfg["female_hazard_ratio"], 1.0)
    return np.exp(b * (df["age"].to_numpy(float) - 40.0)) * sexf


def cum_hazard(df, cfg, b, A0, lp, t):
    """Closed-form H(t) = A0 * scale * (exp(b t) - 1)/b * exp(lp). Age advances with t."""
    return A0 * _scale(df, cfg, b) * (np.expm1(b * t) / b) * np.exp(lp)


def _theta(cfg):
    return float(cfg.get("frailty_theta", 0.0))


def prob_within(df, cfg, b, A0, lp, t):
    """x-conditional ("marginal over frailty") probability of a claim within t years:
    the best any underwriter could do knowing every column. With gamma frailty
    (mean 1, var theta): 1 - (1 + theta*H)^(-1/theta).  theta=0 -> 1 - exp(-H)."""
    H = cum_hazard(df, cfg, b, A0, lp, t)
    th = _theta(cfg)
    if th > 0:
        return 1.0 - (1.0 + th * H) ** (-1.0 / th)
    return 1.0 - np.exp(-H)


def prob_given_frailty(df, cfg, b, A0, lp, t, z):
    """Individual-level probability given the applicant's hidden frailty z (drives the outcomes)."""
    return 1.0 - np.exp(-z * cum_hazard(df, cfg, b, A0, lp, t))


def calibrate_A0(df, cfg, b, lp, target, horizon):
    """Re-solve the baseline level so mean true p_horizon == target.
    Done per gamma (like an insurer re-basing its table to portfolio experience),
    so gamma changes the SHAPE of risk, not the overall level."""
    f = lambda logA: prob_within(df, cfg, b, np.exp(logA), lp, horizon).mean() - target
    return float(np.exp(brentq(f, -15.0, 3.0)))


def draw_event_times(df, cfg, b, A0, lp, E, z=None):
    """Invert z*H(t) = E (E ~ Exp(1), reused across gamma = common random numbers)."""
    E_eff = E if z is None else E / z
    denom = A0 * _scale(df, cfg, b) * np.exp(lp)
    return np.log1p(E_eff * b / denom) / b
