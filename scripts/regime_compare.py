"""Regime 1 (truthful-trained) vs Regime 2 (declared-trained), scored on the DECLARED view of the VALIDATION split
(what an insurer would face). Adds the 'innocent victims' measure: how much honest applicants are over/under-charged.
bias% = mean 100*(p - true_p10)/true_p10 (positive = overcharged); under% = mean 100*(true - p)/true (positive = underpriced).
portfolio_ratio = sum(p*SA)/sum(true_p10*SA), UNWEIGHTED (cohort has ~35% smokers / 35% concealers).
python -m scripts.regime_compare"""
import joblib, numpy as np, pandas as pd
from actuarial_engine.arm_a import TraditionalEngine
from actuarial_engine.arm_a_plus import TraditionalEnginePlus
pd.set_option("display.width", 180)
rows = []
for seed in (2026, 2027, 2028):
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        df = pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz"); va = df[df.split == "val"]
        P = {}
        for reg, suf in (("R1", ""), ("R2", "_declared")):
            m = joblib.load(f"models/seed{seed}_gamma{g}{suf}.joblib")
            E = TraditionalEngine(); E.k = m["arm_a_k"]
            P[("arm_a", reg)] = E.score(va, "declared").p10.to_numpy()
            b = m["arm_b"].predict(va, "declared")
            for k in ("glm", "cox", "xgb"):
                P[(k, reg)] = b[k].to_numpy()
        tr = df[df.split == "train"]
        P[("arm_a_plus", "R2")] = TraditionalEnginePlus().fit(tr, "declared").score(va, "declared").p10.to_numpy()   # experience-rated Arm A (Regime 2 only)
        t = va.true_p10.to_numpy(); sa = va.sum_assured.to_numpy(); ct = va.concealment_type.to_numpy()
        h0 = (ct == "truthful") & (va.true_smoker.to_numpy() == 0); h1 = (ct == "truthful") & (va.true_smoker.to_numpy() == 1)
        for (mod, reg), p in P.items():
            r = {"seed": seed, "gamma": g, "model": mod, "regime": reg, "rmse_all": np.sqrt(np.mean((p - t) ** 2)),
                 "bias%_honest_nonsmoker": 100 * np.mean((p[h0] - t[h0]) / t[h0]), "bias%_honest_smoker": 100 * np.mean((p[h1] - t[h1]) / t[h1]),
                 "portfolio_ratio": (p * sa).sum() / (t * sa).sum()}
            for c in ("diagnosis", "smoking", "multiple"):
                mk = ct == c
                r[f"under%_{c}"] = 100 * np.mean((t[mk] - p[mk]) / t[mk])
            rows.append(r)
d = pd.DataFrame(rows)
cols = ["rmse_all", "bias%_honest_nonsmoker", "bias%_honest_smoker", "under%_diagnosis", "under%_smoking", "under%_multiple", "portfolio_ratio"]
full = d.groupby(["gamma", "model", "regime"])[cols].mean()
full.round(4).to_csv("docs/regime_compare.csv")
for g in (0.0, 1.0):
    print(f"\n=== gamma {g} (mean of 3 seeds, declared view, validation) ===")
    print(full.loc[g].round(3).to_string())
# paired per-seed: does R2 beat R1 on rmse_all? and R2 models vs R2 Arm A
print("\nPaired per-seed rmse_all differences (negative = first is better):")
for g in (0.0, 1.0):
    for a, b in ((("glm", "R2"), ("glm", "R1")), (("glm", "R2"), ("arm_a_plus", "R2")), (("xgb", "R2"), ("arm_a_plus", "R2")), (("xgb", "R2"), ("arm_a", "R2")), (("glm", "R2"), ("arm_a", "R2")), (("xgb", "R2"), ("glm", "R2"))):
        da = d[(d.gamma == g) & (d.model == a[0]) & (d.regime == a[1])].set_index("seed").rmse_all
        db = d[(d.gamma == g) & (d.model == b[0]) & (d.regime == b[1])].set_index("seed").rmse_all
        print(f"gamma {g}: {a[0]}-{a[1]} minus {b[0]}-{b[1]}: " + ", ".join(f"{x:+.4f}" for x in (da - db).tolist()))
