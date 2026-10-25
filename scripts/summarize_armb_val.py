"""Aggregate validation-split preview (mean over 3 seeds). python -m scripts.summarize_armb_val"""
import json
import pandas as pd
pd.set_option("display.width", 160)
R = [json.loads(l) for l in open("docs/armb_val_results.jsonl")]
rows = []
for r in R:
    for model, m in r["metrics"].items():
        rows.append({"seed": r["seed"], "gamma": r["gamma"], "model": model, **m})
d = pd.DataFrame(rows)
order = ["arm_a", "glm", "cox", "xgb", "oracle"]
out = []
for col in ["rmse_true", "rmse_declared", "auc_true_view", "brier_true_view", "underprice_diagnosis", "underprice_smoking", "underprice_multiple"]:
    t = d.pivot_table(index="gamma", columns="model", values=col, aggfunc="mean")[order]
    sd = d.pivot_table(index="gamma", columns="model", values=col, aggfunc="std")[order]
    s = f"\n### {col} (mean of 3 seeds)\n" + t.round(4).to_string()
    if col.startswith("rmse"):
        s += "\n(seed-to-seed SD, max over models): " + str(sd.max(axis=1).round(4).to_dict())
    out.append(s)
print("\n".join(out))
open("docs/armb_val_preview.txt", "w").write("\n".join(out))
