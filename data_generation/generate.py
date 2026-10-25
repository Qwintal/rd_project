"""Two-stage generation so gamma can be swept on IDENTICAL applicants:

  stage 1  generate_base(cfg)        -> covariates + concealment + Exp(1) draws  (gamma-free)
  stage 2  apply_gamma(base, E, cfg) -> true_p10, event time, claim, loss         (gamma-dependent)

Common random numbers: the same E is reused at every gamma, so outcome differences
across gamma come from the hazard, not from fresh sampling noise.
"""
import copy
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from .covariates import draw_covariates
from .concealment import assign_concealment
from . import hazard as hz


def load_config(path="config/default.yaml", **overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg.update(overrides)
    return cfg


def generate_base(cfg):
    rng = np.random.default_rng(cfg["seed"])
    df = draw_covariates(cfg, rng)
    df, pools = assign_concealment(df, cfg, rng)
    E = rng.exponential(1.0, len(df))
    th = float(cfg.get("frailty_theta", 0.0))
    df["frailty_z"] = rng.gamma(1.0 / th, th, len(df)) if th > 0 else np.ones(len(df))   # HIDDEN
    df.insert(0, "applicant_id", [f"A{i:06d}" for i in range(len(df))])
    return df, E, pools


def apply_gamma(base, E, cfg, gamma):
    df = base.copy()
    b = cfg.get("gompertz_b") or hz.fit_gompertz_b()
    H = cfg["horizon_years"]
    lp = hz.linear_predictor(df, cfg, gamma)
    A0 = hz.calibrate_A0(df, cfg, b, lp, cfg["target_p10"], H)
    th = float(cfg.get("frailty_theta", 0.0))
    z = df["frailty_z"].to_numpy() if th > 0 else None
    t = hz.draw_event_times(df, cfg, b, A0, lp, E, z)
    df["true_p10"] = hz.prob_within(df, cfg, b, A0, lp, H)          # x-conditional target (frailty-marginal)
    df["p10_given_frailty"] = (hz.prob_given_frailty(df, cfg, b, A0, lp, H, z)
                               if z is not None else df["true_p10"])  # HIDDEN, drives outcomes
    df["event_time"] = t
    df["claim_10yr"] = (t <= H).astype(int)
    df["observed_time"] = np.minimum(t, H)          # administrative censoring at H
    df["loss_amount"] = df["claim_10yr"] * df["sum_assured"]
    df["true_expected_loss"] = df["true_p10"] * df["sum_assured"]
    df["gamma"] = gamma
    meta = {"b": b, "A0": A0, "lp": lp}
    return df, meta


def assign_split(df, seed, fractions=(0.6, 0.1, 0.3)):
    """Split BY APPLICANT (one row per applicant, so pairing/leakage is structural),
    stratified by concealment type so every subgroup appears in every split."""
    tr, va, te = fractions
    idx = df.index.to_numpy()
    strat = df["concealment_type"]
    train_idx, rest = train_test_split(idx, train_size=tr, stratify=strat, random_state=seed)
    val_idx, test_idx = train_test_split(rest, train_size=va / (va + te),
                                         stratify=strat.loc[rest], random_state=seed)
    split = pd.Series("", index=df.index)
    split.loc[train_idx], split.loc[val_idx], split.loc[test_idx] = "train", "val", "test"
    out = df.copy()
    out.insert(1, "split", split)
    return out


# ---- leakage guard: the ONLY columns a model / rule / trigger may ever see -------------------
OBSERVED = ["age", "sex", "occupation_class", "bmi", "sbp", "cholesterol", "hba1c", "family_history",
            "alcohol", "hazardous_hobby", "sum_assured"]
CONCEALABLE = ["smoker", "pack_years", "diabetes", "hypertension", "cvd"]


def feature_view(df, view):
    """view='true' (truthful record) or 'declared' (what the applicant claims). Returns model inputs only.
    Everything else (true_p10, frailty_z, p10_given_frailty, concealment_*, outcomes) is hidden."""
    assert view in ("true", "declared")
    out = df[OBSERVED].copy()
    for c in CONCEALABLE:
        out[c] = df[f"{view}_{c}"].to_numpy()
    return out
