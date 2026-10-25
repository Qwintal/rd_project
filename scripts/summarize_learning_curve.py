"""Aggregate the learning curve: mean RMSE vs true p10 by gamma and training size. python -m scripts.summarize_learning_curve"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pd.set_option("display.width", 160)
d = pd.DataFrame([json.loads(l) for l in open("docs/learning_curve.jsonl")])
m = ["rmse_arm_a", "rmse_glm", "rmse_cox", "rmse_xgb"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
txt = []
for i, g in enumerate(sorted(d.gamma.unique())):
    s = d[d.gamma == g]
    t = s.groupby("n")[m].mean(); sd = s.groupby("n")[m].std(); c = s.groupby("n").claims_train.mean()
    t.insert(0, "train_claims", c.round(0).astype(int)); txt.append(f"\n### gamma {g} (mean over replicate draws; 3 reps up to 90k, 1 rep at 270k)\n" + t.round(4).to_string())
    for col, lab in zip(m, ["Arm A", "GLM", "Cox", "XGBoost"]):
        ax[i].errorbar(t.index, t[col], yerr=sd[col].fillna(0), marker="o", capsize=3, label=lab)
    ax[i].set(xscale="log", title=f"gamma = {g}", xlabel="training applicants"); ax[i].grid(alpha=.3)
ax[0].set_ylabel("RMSE vs true p10 (truthful view)"); ax[0].legend()
plt.tight_layout(); plt.savefig("docs/fig_learning_curve.png", dpi=110)
print("\n".join(txt)); open("docs/learning_curve_summary.txt", "w").write("\n".join(txt))
