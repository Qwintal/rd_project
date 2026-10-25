"""Fit Arm A (recalibrate) and Arm B (GLM, Cox, XGBoost) on every requested (seed, gamma) dataset.
python -m scripts.train_arm_b 2026 2027 2028        (resumable: skips finished runs)
Saves models/seed{S}_gamma{G}.joblib, docs/glm_coef_seed{S}_gamma{G}.csv, appends docs/armb_val_results.jsonl.
Metrics here are on TRAIN/VALIDATION only; the test split is never touched in this phase."""
import json, os, sys, time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from actuarial_engine.arm_a import TraditionalEngine
from ml_models.arm_b import ArmB

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
REGIME = "declared" if "--declared" in sys.argv else "true"
SUFFIX = "_declared" if REGIME == "declared" else ""
OUT = f"docs/armb_val_results{SUFFIX}.jsonl"
os.makedirs("models", exist_ok=True)
done = set()
if os.path.exists(OUT):
    done = {(r["seed"], r["gamma"]) for r in map(json.loads, open(OUT))}

def metrics(p, va, d_view):
    r = {"rmse_true": float(np.sqrt(np.mean((p["true"] - va.true_p10) ** 2))),
         "rmse_declared": float(np.sqrt(np.mean((p["declared"] - va.true_p10) ** 2))),
         "auc_true_view": float(roc_auc_score(va.claim_10yr, p["true"])),
         "brier_true_view": float(brier_score_loss(va.claim_10yr, p["true"]))}
    for ct in ("diagnosis", "smoking", "multiple"):
        m = (va.concealment_type == ct).to_numpy()
        r[f"underprice_{ct}"] = float(100 * ((va.true_p10[m] - p["declared"][m]) / va.true_p10[m]).mean())
    return r

for seed in map(int, [a for a in sys.argv[1:] if not a.startswith('--')] or [2026]):
    for g in GAMMAS:
        if (seed, g) in done:
            continue
        t0 = time.time()
        df = pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz")
        tr, va = df[df.split == "train"], df[df.split == "val"]
        A = TraditionalEngine().fit(tr, REGIME)
        B = ArmB(seed, REGIME).fit(tr, va)
        P = {"arm_a": {v: A.score(va, v).p10 for v in ("true", "declared")}}
        for m in ("glm", "cox", "xgb"):
            P[m] = {v: B.predict(va, v)[m] for v in ("true", "declared")}
        P["oracle"] = {"true": va.true_p10, "declared": va.true_p10}
        row = {"seed": seed, "gamma": g, "k_arm_a": A.k, "xgb": B.info["xgb_params"], "calibration": B.info["calibration"],
               "metrics": {m: metrics(P[m], va, None) for m in P}, "seconds": round(time.time() - t0, 1)}
        joblib.dump({"arm_a_k": A.k, "arm_b": B}, f"models/seed{seed}_gamma{g}{SUFFIX}.joblib")
        B.glm_table.round(4).to_csv(f"docs/glm_coef_seed{seed}_gamma{g}{SUFFIX}.csv")
        open(OUT, "a").write(json.dumps(row) + "\n")
        print(f"seed {seed} gamma {g}: {row['seconds']}s | xgb {B.info['xgb_params']} | cal "
              f"{ {m: B.info['calibration'][m]['chosen'] for m in ('glm','cox','xgb')} }", flush=True)
