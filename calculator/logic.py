"""Calculator logic - pure functions, no Streamlit. Everything the app shows is computed here (and unit-tested).

Premium: P = (1 + 0.15 expense + 0.10 profit) * p_term * SA / term  (annual, level; no discounting, lapse or reserves).
TENURE EXTRAPOLATION (stated in the UI): the models are trained and validated ONLY on a 10-year claim window. For any other
term, p10 is converted to an implied constant annual hazard lambda = -ln(1 - p10)/10 and compounded: p_T = 1 - (1 - p10)^(T/10).
That is a modelling assumption layered on top of the models, not something they learned or were validated against. Because the
synthetic ground truth is known, exact_true_p() shows how wrong it is.
"""
import json, os
import numpy as np
import pandas as pd
import xgboost as xgb
from actuarial_engine.arm_a import TraditionalEngine, premium, LOADINGS, SBP_HTN, HBA1C_DM
from actuarial_engine.arm_a_plus import TraditionalEnginePlus
from data_generation import hazard as hz
from data_generation.generate import load_config
from ml_models.features import xgb_matrix, glm_matrix
from verification.triggers import RuleTrigger

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CONC = ["smoker", "pack_years", "diabetes", "hypertension", "cvd"]
TRAIN_AGE = (40, 60)
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
EXTRAP_DISCLOSURE = ("The models are trained and validated only on a 10-year claim window. For other terms the 10-year claim probability is converted into an "
                     "implied constant annual hazard rate and compounded over the requested term. (So other values might not be equal)")


def build_record(obs, true, declared=None):
    """One-row frame in the project schema. `obs` = insurer-measured/observed fields; `true`/`declared` = concealable fields."""
    declared = dict(true) if declared is None else {**true, **declared}
    row = dict(obs)
    for c in CONC:
        row[f"true_{c}"], row[f"declared_{c}"] = true[c], declared[c]
    return pd.DataFrame([row])


def extrapolate(p10, term):
    """Constant-hazard extrapolation of a 10-year claim probability to `term` years."""
    return 1.0 - (1.0 - np.asarray(p10, float)) ** (term / 10.0)


def term_premium(p_term, sum_assured, term):
    return float(premium(p_term, sum_assured, term=term))


def exact_true_p(rec, cfg, gamma, A0, b, term):
    """Exact DGP claim probability within `term` years for the TRUE record (ages advance along the term).
    Beyond ~20 years applicants pass age 80, outside the 40-59 range the baseline slope was fitted on: treat as extrapolated too."""
    lp = hz.linear_predictor(rec, cfg, gamma)
    return float(hz.prob_within(rec, cfg, b, A0, lp, float(term))[0])


def age_warnings(age, term):
    w = []
    if not (TRAIN_AGE[0] <= age <= TRAIN_AGE[1]):
        w.append(f"Age {age} is outside the 40-60 range the models were trained on: predictions are extrapolation and unreliable.")
    if age + term > 85:
        w.append(f"Age at expiry would be {age + term}: far beyond the ages the baseline hazard was fitted on.")
    return w


class Models:
    """Everything needed to price one (gamma, regime) setting from exported artifacts (no pickles except the consistency trigger)."""
    def __init__(self, gamma, regime):
        self.gamma, self.regime = gamma, regime
        m = json.load(open(f"{ART}/gamma{gamma}.json"))
        self.A0, self.b = m["A0"], m["b"]
        self.k_arm_a = m["arm_a_k"][regime]
        self.plus = m["arm_a_plus"] if regime == "R2" else None
        self.glm = pd.read_csv(f"{ART}/glm_{regime}_gamma{gamma}.csv", index_col=0)["coef"]
        self.best_it = m["xgb_best_iteration"][regime]
        self.booster = xgb.Booster(); self.booster.load_model(f"{ART}/xgb_{regime}_gamma{gamma}.json")

    def arm_a(self, rec, view):
        e = TraditionalEngine(); e.k = self.k_arm_a
        s = e.score(rec, view).iloc[0]
        return float(s.p10), float(s.multiplier), str(s.decision)

    def arm_a_plus(self, rec, view):
        if self.plus is None:
            return None
        e = TraditionalEnginePlus(); e.k = self.plus["k"]; e.k_cls = {int(k): v for k, v in self.plus["k_cls"].items()}
        return float(e.score(rec, view).iloc[0].p10)

    def glm_p10(self, rec, view):
        G = glm_matrix(rec, view)
        z = self.glm["const"] + float((G[[c for c in self.glm.index if c != "const"]] * self.glm.drop("const")).sum(axis=1).iloc[0])
        return float(1.0 / (1.0 + np.exp(-z)))

    def _dm(self, rec, view):
        X = xgb_matrix(rec, view)
        return xgb.DMatrix(X, feature_names=list(X.columns)), list(X.columns)

    def xgb_p10(self, rec, view):
        dm, _ = self._dm(rec, view)
        return float(self.booster.predict(dm, iteration_range=(0, self.best_it + 1))[0])

    def xgb_contribs(self, rec, view):
        """TreeSHAP log-odds contributions (native XGBoost), largest first."""
        dm, cols = self._dm(rec, view)
        c = self.booster.predict(dm, pred_contribs=True, iteration_range=(0, self.best_it + 1))[0]
        return pd.Series(c[:-1], index=cols).sort_values(key=np.abs, ascending=False), float(c[-1])

    def all_p10(self, rec, view):
        out = {"Arm A (traditional)": self.arm_a(rec, view)[0], "GLM": self.glm_p10(rec, view), "XGBoost": self.xgb_p10(rec, view)}
        plus = self.arm_a_plus(rec, view)
        if plus is not None:
            out["Arm A+ (experience-rated)"] = plus
        return out


def arm_a_breakdown(rec, view):
    """Loading table for the applicant: which factors fired and by how much (explains Arm A completely)."""
    X = xgb_matrix(rec, view); r = X.iloc[0]
    htn = (r.hypertension == 1) or (r.sbp >= SBP_HTN); dm = (r.diabetes == 1) or (r.hba1c >= HBA1C_DM)
    fired = {"smoker": r.smoker == 1, "hypertension (declared or SBP>=140)": htn, "diabetes (declared or HbA1c>=6.5)": dm, "CVD": r.cvd == 1,
             "BMI>=30": r.bmi >= 30, "cholesterol>=240": r.cholesterol >= 240, "heavy alcohol": r.alcohol == 2, "occupation class 4": r.occupation_class == 4,
             "family history": r.family_history == 1}
    keymap = {"smoker": "smoker", "hypertension (declared or SBP>=140)": "hypertension", "diabetes (declared or HbA1c>=6.5)": "diabetes", "CVD": "cvd",
              "BMI>=30": "bmi_ge_30", "cholesterol>=240": "chol_ge_240", "heavy alcohol": "alcohol_heavy", "occupation class 4": "occupation_class_4", "family history": "family_history"}
    return pd.DataFrame([{"factor": k, "loading": LOADINGS[keymap[k]] if v else 1.0} for k, v in fired.items()])


# ---------------- verification helpers ----------------
def consistency_percentile(score, sorted_scores):
    """Top-x%: share (in %) of the design cohort whose suspicion score is >= this applicant's."""
    return float(100.0 * np.mean(np.asarray(sorted_scores) >= score))


def p_conceal_given_rank(pct, prevalence, table):
    """P(concealer | suspicion rank bin) re-weighted from the design cohort to `prevalence` by Bayes (odds scaling)."""
    k = min(int(pct // 5), len(table["n_conc"]) - 1)
    nc, nh = table["n_conc"][k], table["n_hon"][k]
    s = table["share_conc"]
    odds = (nc / max(nh, 1e-9)) * (prevalence / (1 - prevalence)) / (s / (1 - s))
    return float(odds / (1 + odds))


def verify_decision(c, p_conc, v, dpi):
    """Verify iff cost < P(conceals | flagged) * v * premium shortfall recovered."""
    benefit = p_conc * v * dpi
    return bool(c < benefit), float(benefit)
