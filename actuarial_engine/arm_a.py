"""Arm A - traditional actuarial engine.  Deterministic and transparent.

    base hazard (5-yr age band x sex)  x  product of fixed loadings  x  one recalibration scalar k
        -> 10-yr claim probability  p = 1 - exp(-10 * k * h_band * M)
        -> decision table (Standard / Rated / Decline)
        -> closed-form premium  P = (1 + expense + profit) * p * SA / term

DELIBERATE SIMPLIFICATIONS (state these in the report): loadings are multiplied with NO interaction terms
and NO cap; ratings come in coarse 0.25 steps; the base table is CDC-shaped (SOA VBT not accessible here).
Loadings were fixed BEFORE any results were seen (docs/arm_a_spec.md).
"""
import numpy as np
import pandas as pd
from data_generation.hazard import fit_gompertz_b
from data_generation.generate import feature_view

# ---- fixed loadings: literature / assumed values rounded to the nearest 0.25 (rating-table practice) ----
LOADINGS = {
    "smoker": 2.50, "hypertension": 1.25, "bmi_ge_30": 1.25, "diabetes": 1.75, "cvd": 2.00,
    "alcohol_heavy": 1.25, "occupation_class_4": 1.25, "family_history": 1.25, "chol_ge_240": 1.25,
    "occupation_class_3": 1.00, "hazardous_hobby": 1.00,
}
SBP_HTN = 140.0      # measured SBP at/above this counts as hypertension even if undeclared
HBA1C_DM = 6.5       # measured HbA1c at/above this counts as diabetes even if undeclared
STANDARD_MAX = 1.25  # M <= this  -> Standard
DECLINE_ABOVE = 5.0  # M >  this  -> Decline ; in between -> Rated
EXPENSE, PROFIT = 0.15, 0.10   # (A) premium loadings
TERM = 10
FEMALE_FACTOR = 0.60           # (A) same assumption as the DGP's female/male ratio
BAND_EDGES = [40, 45, 50, 55, 61]


def _base_table():
    """5-year age-band x sex annual base hazard, from the log-linear CDC 2023 US-male fit
    (4 life-table points) evaluated at each band's mean age. Population-shaped, not select."""
    ages = np.array([40, 49, 50, 59], dtype=float)
    q = np.array([0.003176, 0.004859, 0.005182, 0.010581])
    b, c = np.polyfit(ages, np.log(q), 1)
    rows = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        mean_age = np.arange(lo, hi).mean()
        rows.append({"band": f"{lo}-{hi - 1}", "lo": lo, "hi": hi - 1, "male": np.exp(c + b * mean_age)})
    t = pd.DataFrame(rows)
    t["female"] = t["male"] * FEMALE_FACTOR
    return t


BASE_TABLE = _base_table()


def premium(p, sum_assured, term=TERM, expense=EXPENSE, profit=PROFIT):
    """Closed-form annual premium. No discounting, lapse or reserves (stated limitation)."""
    return (1.0 + expense + profit) * np.asarray(p) * np.asarray(sum_assured) / term


class TraditionalEngine:
    def __init__(self):
        self.k = 1.0   # recalibration scalar, fitted on the training split

    # ---------------- scoring ----------------
    def _multiplier(self, X):
        L = LOADINGS
        htn = ((X["hypertension"] == 1) | (X["sbp"] >= SBP_HTN)).to_numpy()
        dm = ((X["diabetes"] == 1) | (X["hba1c"] >= HBA1C_DM)).to_numpy()
        m = np.ones(len(X))
        m *= np.where(X["smoker"] == 1, L["smoker"], 1.0)
        m *= np.where(htn, L["hypertension"], 1.0)
        m *= np.where(dm, L["diabetes"], 1.0)
        m *= np.where(X["cvd"] == 1, L["cvd"], 1.0)
        m *= np.where(X["bmi"] >= 30, L["bmi_ge_30"], 1.0)
        m *= np.where(X["cholesterol"] >= 240, L["chol_ge_240"], 1.0)
        m *= np.where(X["alcohol"] == 2, L["alcohol_heavy"], 1.0)
        m *= np.where(X["hazardous_hobby"] == 1, L["hazardous_hobby"], 1.0)
        m *= np.where(X["occupation_class"] == 4, L["occupation_class_4"], 1.0)
        m *= np.where(X["occupation_class"] == 3, L["occupation_class_3"], 1.0)
        m *= np.where(X["family_history"] == 1, L["family_history"], 1.0)
        return m

    def _base_hazard(self, X):
        h = np.empty(len(X))
        age = X["age"].to_numpy()
        female = (X["sex"] == "F").to_numpy()
        for _, r in BASE_TABLE.iterrows():
            in_band = (age >= r["lo"]) & (age <= r["hi"])
            h[in_band & female] = r["female"]
            h[in_band & ~female] = r["male"]
        return h

    def p10(self, X, k=None):
        k = self.k if k is None else k
        return 1.0 - np.exp(-TERM * k * self._base_hazard(X) * self._multiplier(X))

    def fit(self, train_df, view="true"):
        """Fit the single recalibration scalar so mean predicted p10 == observed claim rate on TRAIN
        (insurer re-basing its table to portfolio experience). Uses the truthful view, like the ML arm."""
        X = feature_view(train_df, view)
        target = train_df["claim_10yr"].mean()
        lo, hi = 1e-3, 50.0
        for _ in range(80):                     # bisection; mean p is monotone in k
            mid = 0.5 * (lo + hi)
            if self.p10(X, mid).mean() < target:
                lo = mid
            else:
                hi = mid
        self.k = 0.5 * (lo + hi)
        return self

    def decide(self, M):
        return np.where(M > DECLINE_ABOVE, "Decline", np.where(M > STANDARD_MAX, "Rated", "Standard"))

    def score(self, df, view):
        """view = 'true' or 'declared'. Returns p10, premium (technical premium for EVERYONE, incl. declined),
        multiplier and decision."""
        X = feature_view(df, view)
        M = self._multiplier(X)
        p = self.p10(X)
        return pd.DataFrame({"multiplier": M, "p10": p, "premium": premium(p, df["sum_assured"].to_numpy()),
                             "decision": self.decide(M)}, index=df.index)
