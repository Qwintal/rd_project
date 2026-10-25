"""Plausibility / verification triggers. Both see ONLY the declared view (feature_view(df,'declared')): never the true record.

RuleTrigger        (Arm A) : explicit single-threshold inconsistency rules an underwriter would write.
ConsistencyTrigger (Arm B) : label-free statistical consistency model, trained on declared data only - predicts each
                             declared condition from the OTHER columns; suspicion = summed probability that a field declared
                             'no' is really 'yes'.  No concealment labels are used (an insurer would not have them).
hidden_loss()              : EVALUATION ONLY - what the lie hides (uses the true record and the true hazard).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from data_generation.generate import feature_view
from data_generation import hazard as hz
from ml_models.features import xgb_matrix

SBP_RULE, HBA1C_RULE = 140.0, 6.5


class RuleTrigger:
    name = "rule (Arm A)"

    def score(self, df):
        X = feature_view(df, "declared")
        r1 = (X["hypertension"] == 0) & (X["sbp"] >= SBP_RULE)
        r2 = (X["diabetes"] == 0) & (X["hba1c"] >= HBA1C_RULE)
        return (r1.astype(int) + r2.astype(int)).to_numpy().astype(float)      # number of rules fired

    def flag(self, df):
        return self.score(df) >= 1


class ConsistencyTrigger:
    name = "consistency model (Arm B)"
    FIELDS = {"smoker": ["smoker", "pack_years"], "hypertension": ["hypertension"], "diabetes": ["diabetes"], "cvd": ["cvd"]}

    def __init__(self, seed=0):
        self.seed, self.models = seed, {}

    def fit(self, train_df):
        X = xgb_matrix(train_df, "declared")
        for f, drop in self.FIELDS.items():
            self.models[f] = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, random_state=self.seed) \
                .fit(X.drop(columns=drop), X[f].astype(int))
        return self

    def score(self, df):
        X = xgb_matrix(df, "declared")
        s = np.zeros(len(X))
        for f, drop in self.FIELDS.items():
            s += self.models[f].predict_proba(X.drop(columns=drop))[:, 1] * (X[f] == 0).to_numpy()
        return s


def hidden_loss(df, cfg):
    """EVALUATION ONLY. Returns (p_declared_oracle, shortfall): the claim probability the TRUE hazard assigns to the
    DECLARED record, and what the lie hides (true_p10 - that). Zero for honest applicants."""
    g = float(df["gamma"].iloc[0])
    d = df.copy()
    for c in ["smoker", "hypertension", "diabetes", "cvd"]:
        d[f"true_{c}"] = d[f"declared_{c}"]
    dlp = hz.linear_predictor(d, cfg, g) - hz.linear_predictor(df, cfg, g)
    th = float(cfg["frailty_theta"])
    p = df["true_p10"].to_numpy()
    H = ((1 - p) ** (-th) - 1) / th
    pdcl = 1 - (1 + th * H * np.exp(dlp)) ** (-1 / th)
    return pdcl, p - pdcl
