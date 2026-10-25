"""Learning curve: at what TRAINING size (if ever) does XGBoost overtake the GLM / Arm A?
Separate cohort (seed 3000, 335,000 applicants) so the project's test split is never touched.
Pools: train 270,000 | validation 45,000 | fixed evaluation set 20,000 (never trained or tuned on).
For each size n: train n, validation n/6 (the project's 60:10 ratio), Regime 1 (truthful-trained), preset Arm B protocol.
Metric: RMSE of predicted p10 vs true_p10 on the fixed evaluation set, truthful view.
python -m scripts.learning_curve   (resumable; writes docs/learning_curve.jsonl)"""
import json, os, time
import numpy as np, pandas as pd
from actuarial_engine.arm_a import TraditionalEngine
from data_generation.generate import load_config, generate_base, apply_gamma
from ml_models.arm_b import ArmB

OUT = "docs/learning_curve.jsonl"
SIZES = [5_000, 15_000, 30_000, 90_000, 270_000]
done = {(r["gamma"], r["n"], r["rep"]) for r in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
t0 = time.time()
cfg = load_config("config/default.yaml", n_applicants=335_000, seed=3000)
base, E, _ = generate_base(cfg)
print(f"cohort generated in {time.time()-t0:.0f}s", flush=True)
for g in (0.0, 1.0):
    df, _ = apply_gamma(base, E, cfg, g)
    tr_pool, va_pool, ev = df.iloc[:270_000], df.iloc[270_000:315_000], df.iloc[315_000:]
    for n in SIZES:
        for rep in range(1 if n == 270_000 else 3):
            if (g, n, rep) in done:
                continue
            t1 = time.time()
            tr = tr_pool.sample(n, random_state=100 + rep)
            va = va_pool.sample(n // 6, random_state=200 + rep)
            A = TraditionalEngine().fit(tr, "true")
            B = ArmB(3000 + rep).fit(tr, va)
            P = B.predict(ev, "true")
            r = {"gamma": g, "n": n, "rep": rep, "claims_train": int(tr.claim_10yr.sum()),
                 "rmse_arm_a": float(np.sqrt(np.mean((A.score(ev, "true").p10 - ev.true_p10) ** 2))),
                 **{f"rmse_{m}": float(np.sqrt(np.mean((P[m] - ev.true_p10) ** 2))) for m in ("glm", "cox", "xgb")},
                 "xgb_params": B.info["xgb_params"], "seconds": round(time.time() - t1, 1)}
            open(OUT, "a").write(json.dumps(r) + "\n")
            print(f"gamma {g} n {n} rep {rep}: A {r['rmse_arm_a']:.4f} glm {r['rmse_glm']:.4f} cox {r['rmse_cox']:.4f} xgb {r['rmse_xgb']:.4f} ({r['seconds']}s)", flush=True)
print("DONE", flush=True)
