"""SHAP on the XGBoost model (descriptive post-processing, NOT a separate model; does not prove interaction recovery).
python -m scripts.shap_analysis      (seed 2026, validation split; gamma 0 vs 1)"""
import joblib, numpy as np, pandas as pd, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from ml_models.features import xgb_matrix
pd.set_option("display.width", 150)
res, glob, inter, shift = {}, {}, {}, []
for g in (0.0, 1.0):
    df = pd.read_csv(f"data/cohort_seed2026_gamma{g}.csv.gz"); va = df[df.split == "val"]
    B = joblib.load(f"models/seed2026_gamma{g}.joblib")["arm_b"]
    ex = shap.TreeExplainer(B.xgb)
    Xt = xgb_matrix(va, "true"); sv = ex.shap_values(Xt)
    glob[g] = pd.Series(np.abs(sv).mean(0), index=Xt.columns)
    sub = Xt.sample(2000, random_state=0)
    iv = ex.shap_interaction_values(sub); cols = list(Xt.columns)
    ia = lambda a, b: float(2 * np.abs(iv[:, cols.index(a), cols.index(b)]).mean())   # off-diagonal appears twice
    inter[g] = {"smoker x hypertension": ia("smoker", "hypertension"), "smoker x bmi": ia("smoker", "bmi"),
                "hypertension x bmi": ia("hypertension", "bmi"), "smoker x age": ia("smoker", "age")}
    for label, mask, feat in (("smoking concealers: SHAP(smoker)", va.concealed_fields.str.contains("smoker"), "smoker"),
                              ("hypertension concealers: SHAP(hypertension)", va.concealed_fields.str.contains("hypertension"), "hypertension"),
                              ("diabetes concealers: SHAP(diabetes)", va.concealed_fields.str.contains("diabetes"), "diabetes")):
        rows = va[mask]
        a = ex.shap_values(xgb_matrix(rows, "true"))[:, cols.index(feat)].mean()
        b = ex.shap_values(xgb_matrix(rows, "declared"))[:, cols.index(feat)].mean()
        mt = ex.expected_value + ex.shap_values(xgb_matrix(rows, "true")).sum(1); md = ex.expected_value + ex.shap_values(xgb_matrix(rows, "declared")).sum(1)
        shift.append({"gamma": g, "group": label, "n": len(rows), "mean SHAP truthful view": a, "mean SHAP declared view": b,
                      "mean log-odds drop from the lie": float((mt - md).mean())})
G = pd.DataFrame(glob).round(4); G.columns = ["gamma 0", "gamma 1"]
print("Mean |SHAP| (log-odds), validation, truthful view - top 8 at gamma 1:\n", G.sort_values("gamma 1", ascending=False).head(8).to_string())
I = pd.DataFrame(inter).round(4); I.columns = ["gamma 0", "gamma 1"]
print("\nMean |SHAP interaction| (log-odds), 2,000 val rows:\n", I.to_string())
S = pd.DataFrame(shift).round(3); print("\nExplanation shift under concealment:\n", S.to_string(index=False))
open("docs/shap_summary.txt", "w").write(G.sort_values("gamma 1", ascending=False).to_string() + "\n\n" + I.to_string() + "\n\n" + S.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
o = G.sort_values("gamma 1").tail(10); y = np.arange(len(o))
ax[0].barh(y - .2, o["gamma 0"], .4, label="gamma 0"); ax[0].barh(y + .2, o["gamma 1"], .4, label="gamma 1")
ax[0].set_yticks(y); ax[0].set_yticklabels(o.index); ax[0].set(title="XGBoost: mean |SHAP|, truthful view", xlabel="log-odds"); ax[0].legend()
I.plot.bar(ax=ax[1], rot=20); ax[1].set(title="Mean |SHAP interaction| (descriptive)", ylabel="log-odds")
plt.tight_layout(); plt.savefig("docs/fig_shap.png", dpi=110)
