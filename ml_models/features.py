"""Model inputs. EVERYTHING goes through feature_view(df, 'true'|'declared'): no hidden column can leak in."""
import numpy as np
import pandas as pd
from data_generation.generate import feature_view


def xgb_matrix(df, view):
    """16 raw features for XGBoost. It must learn thresholds and interactions by itself."""
    X = feature_view(df, view)
    out = pd.DataFrame({
        "age": X["age"], "sex_male": (X["sex"] == "M").astype(int), "occupation_class": X["occupation_class"],
        "bmi": X["bmi"], "sbp": X["sbp"], "cholesterol": X["cholesterol"], "hba1c": X["hba1c"],
        "family_history": X["family_history"], "alcohol": X["alcohol"], "hazardous_hobby": X["hazardous_hobby"],
        "log10_sum_assured": np.log10(X["sum_assured"]), "smoker": X["smoker"], "pack_years": X["pack_years"],
        "diabetes": X["diabetes"], "hypertension": X["hypertension"], "cvd": X["cvd"]})
    return out.astype(float)


def glm_matrix(df, view):
    """Design for the logistic GLM and Cox PH: main effects only (NO interactions), plus the standard clinical-band
    flags an applied statistician would add (the same cut-offs Arm A uses). Centred/scaled for stable fitting."""
    X = xgb_matrix(df, view)
    return pd.DataFrame({
        "age_dec": (X.age - 50) / 10, "male": X.sex_male,
        "occ3": (X.occupation_class == 3).astype(float), "occ4": (X.occupation_class == 4).astype(float),
        "bmi_5": (X.bmi - 25) / 5, "bmi_ge_30": (X.bmi >= 30).astype(float),
        "sbp_10": (X.sbp - 120) / 10, "sbp_ge_140": (X.sbp >= 140).astype(float),
        "chol_30": (X.cholesterol - 200) / 30, "chol_ge_240": (X.cholesterol >= 240).astype(float),
        "hba1c_c": X.hba1c - 5.5, "hba1c_ge_6_5": (X.hba1c >= 6.5).astype(float),
        "family_history": X.family_history,
        "alc_moderate": (X.alcohol == 1).astype(float), "alc_heavy": (X.alcohol == 2).astype(float),
        "hobby": X.hazardous_hobby, "log10_sa_c": X.log10_sum_assured - 6.5,
        "smoker": X.smoker, "pack_years_10": X.pack_years / 10,
        "diabetes": X.diabetes, "hypertension": X.hypertension, "cvd": X.cvd})
