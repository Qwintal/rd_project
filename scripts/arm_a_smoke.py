"""Arm A smoke test on TRAIN (fit) and VALIDATION only - the test split is not touched until Phase 7.
python -m scripts.arm_a_smoke"""
import numpy as np, pandas as pd
from actuarial_engine.arm_a import TraditionalEngine
from actuarial_engine.arm_a import BASE_TABLE

pd.set_option("display.width", 150)
print(BASE_TABLE.round(5).to_string(index=False), "\n")
rows, mix = [], []
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    df = pd.read_csv(f"data/cohort_seed2026_gamma{g}.csv.gz")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    eng = TraditionalEngine().fit(tr, "true")
    t, d = eng.score(va, "true"), eng.score(va, "declared")
    r = {"gamma": g, "k": round(eng.k, 3),
         "mean_p_val": t.p10.mean(), "mean_true_p_val": va.true_p10.mean(),
         "RMSE_true_view": np.sqrt(((t.p10 - va.true_p10) ** 2).mean()),
         "RMSE_declared_view": np.sqrt(((d.p10 - va.true_p10) ** 2).mean())}
    for ct in ("diagnosis", "smoking", "multiple"):
        m = va.concealment_type == ct
        r[f"underprice%_{ct}"] = 100 * ((va.true_p10[m] - d.p10[m]) / va.true_p10[m]).mean()
    rows.append(r)
    mix.append({"gamma": g, **t.decision.value_counts(normalize=True).round(3).to_dict()})
print(pd.DataFrame(rows).set_index("gamma").round(4).to_string())
print("\nDecision mix, truthful view, validation split:")
print(pd.DataFrame(mix).set_index("gamma").to_string())
