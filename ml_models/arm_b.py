"""Arm B - parallel model ladder (NOT a pipeline): GLM, Cox PH, XGBoost, each trained independently on the
TRUTHFUL records of the train split.  Calibration (none / Platt / isotonic) is a post-processing step applied
identically to all three models and chosen by 5-fold CV log loss on the VALIDATION split (tie within 0.0003 -> simpler).
The test split is never used here.
"""
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from lifelines import CoxPHFitter
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from .features import xgb_matrix, glm_matrix

GRID = {"max_depth": [2, 3, 4], "min_child_weight": [10, 40], "reg_lambda": [1.0, 10.0]}   # fixed before any run
XGB_FIXED = dict(learning_rate=0.05, n_estimators=1000, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                 eval_metric="logloss", early_stopping_rounds=50, n_jobs=1)
EPS = 1e-6
TIE = 0.0003


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    def __init__(self, method):
        self.method = method

    def fit(self, p, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p, y)
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        if self.method == "none":
            return np.clip(p, EPS, 1 - EPS)
        if self.method == "platt":
            return self.m.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return np.clip(self.m.predict(p), EPS, 1 - EPS)


def select_calibrator(p, y, seed):
    """5-fold CV log loss on validation predictions; returns (fitted calibrator, cv table)."""
    cv = {m: [] for m in ("none", "platt", "isotonic")}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        for m in cv:
            cal = Calibrator(m).fit(p[tr], y[tr])
            cv[m].append(log_loss(y[te], cal.predict(p[te]), labels=[0, 1]))
    mean = {m: float(np.mean(v)) for m, v in cv.items()}
    best = min(mean, key=mean.get)
    for simpler in ("none", "platt", "isotonic"):               # simplest method within TIE of the best
        if mean[simpler] <= mean[best] + TIE:
            best = simpler
            break
    return Calibrator(best).fit(p, y), mean


class ArmB:
    def __init__(self, seed, train_view="true"):
        self.seed = seed
        self.tv = train_view          # 'true' = Regime 1 (truthful-trained); 'declared' = Regime 2 (what an insurer actually has)
        self.info = {"train_view": train_view}

    def fit(self, train, val):
        y_tr, y_va = train["claim_10yr"].to_numpy(), val["claim_10yr"].to_numpy()
        Gtr, Gva = glm_matrix(train, self.tv), glm_matrix(val, self.tv)
        Xtr, Xva = xgb_matrix(train, self.tv), xgb_matrix(val, self.tv)

        # ---- logistic GLM (additive baseline, interpretable) ----
        self.glm = sm.GLM(y_tr, sm.add_constant(Gtr), family=sm.families.Binomial()).fit()
        self.glm_table = pd.DataFrame({"coef": self.glm.params, "se": self.glm.bse, "p_value": self.glm.pvalues})
        self.glm.remove_data()
        # ---- Cox PH (time-to-claim, censored at 10y), main effects only ----
        d = Gtr.copy()
        d["T"], d["E"] = train["observed_time"].to_numpy(), y_tr
        self.cox = CoxPHFitter(penalizer=0.001).fit(d, "T", "E")
        self.cox_table = self.cox.summary[["coef", "se(coef)", "p"]].rename(columns={"se(coef)": "se", "p": "p_value"})
        # ---- XGBoost: fixed small grid, early stopping on validation ----
        best = None
        for depth, mcw, lam in itertools.product(GRID["max_depth"], GRID["min_child_weight"], GRID["reg_lambda"]):
            clf = xgb.XGBClassifier(max_depth=depth, min_child_weight=mcw, reg_lambda=lam, random_state=self.seed, **XGB_FIXED)
            clf.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
            ll = clf.evals_result()["validation_0"]["logloss"][clf.best_iteration]
            if best is None or ll < best[0]:
                best = (ll, clf, dict(max_depth=depth, min_child_weight=mcw, reg_lambda=lam, best_iteration=int(clf.best_iteration)))
        self.xgb, self.info["xgb_params"], self.info["xgb_val_logloss"] = best[1], best[2], float(best[0])

        # ---- calibration, same protocol for all three ----
        raw = self._raw(val, self.tv)
        self.cal, self.info["calibration"] = {}, {}
        for m in ("glm", "cox", "xgb"):
            self.cal[m], cvtab = select_calibrator(raw[m].to_numpy(), y_va, self.seed)
            self.info["calibration"][m] = {"chosen": self.cal[m].method, **{f"cv_logloss_{k}": v for k, v in cvtab.items()}}
        return self

    def _raw(self, df, view):
        G, X = glm_matrix(df, view), xgb_matrix(df, view)
        H = self.cox.predict_cumulative_hazard(G, times=[10.0]).to_numpy().ravel()
        return pd.DataFrame({"glm": self.glm.predict(sm.add_constant(G, has_constant="add")).to_numpy(),
                             "cox": 1 - np.exp(-H),
                             "xgb": self.xgb.predict_proba(X)[:, 1]}, index=df.index)

    def predict(self, df, view):
        """Calibrated 10-yr claim probabilities for each model; view = 'true' or 'declared'."""
        raw = self._raw(df, view)
        return pd.DataFrame({m: self.cal[m].predict(raw[m].to_numpy()) for m in raw}, index=df.index)
