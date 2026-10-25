"""EXPLORATORY (validation split only, seed 2026, gamma 0 and 1) - two questions raised by the Phase 5 preview.
Not part of the pre-set protocol; reported as such.
  Q1: is XGBoost's weakness an artefact of tuning on the small validation split? -> tune by 3-fold CV on TRAIN instead.
  Q2: does training on DECLARED records (what an insurer actually has) change robustness to concealment?"""
import json, itertools
import numpy as np, pandas as pd, xgboost as xgb, statsmodels.api as sm
from ml_models.features import xgb_matrix, glm_matrix
from ml_models.arm_b import GRID, XGB_FIXED

def rmse(p, t): return float(np.sqrt(np.mean((p - t) ** 2)))
def under(p, va, ct):
    m = (va.concealment_type == ct).to_numpy()
    return float(100 * ((va.true_p10[m] - p[m]) / va.true_p10[m]).mean())

def cv_tuned_xgb(Xtr, ytr, seed):
    d = xgb.DMatrix(Xtr, ytr); best = None
    for depth, mcw, lam in itertools.product(GRID["max_depth"], GRID["min_child_weight"], GRID["reg_lambda"]):
        prm = dict(objective="binary:logistic", eval_metric="logloss", eta=0.05, max_depth=depth, min_child_weight=mcw,
                   reg_lambda=lam, subsample=0.8, colsample_bytree=0.8, tree_method="hist", nthread=1, seed=seed)
        cv = xgb.cv(prm, d, num_boost_round=1000, nfold=3, stratified=True, early_stopping_rounds=50, seed=seed)
        ll, n = cv["test-logloss-mean"].min(), int(cv["test-logloss-mean"].idxmin()) + 1
        if best is None or ll < best[0]: best = (ll, prm, n, dict(depth=depth, mcw=mcw, lam=lam, rounds=n))
    return xgb.train(best[1], d, num_boost_round=best[2]), best[3]

rows = []
for g in (0.0, 1.0):
    df = pd.read_csv(f"data/cohort_seed2026_gamma{g}.csv.gz")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ytr = tr.claim_10yr.to_numpy()
    Xt_tr, Xt_va, Xd_va = xgb_matrix(tr, "true"), xgb_matrix(va, "true"), xgb_matrix(va, "declared")
    # Q1: CV-tuned XGB, trained on truthful records
    b, prm = cv_tuned_xgb(Xt_tr, ytr, 2026)
    pt, pd_ = b.predict(xgb.DMatrix(Xt_va)), b.predict(xgb.DMatrix(Xd_va))
    rows.append({"gamma": g, "variant": "XGB regime1 CV-tuned on train", "rmse_true_view": rmse(pt, va.true_p10), "rmse_declared": rmse(pd_, va.true_p10),
                 **{f"under_{c}": under(pd_, va, c) for c in ("diagnosis", "smoking", "multiple")}, "note": str(prm)})
    # Q2: regime 2 = train on DECLARED records of the train split (true labels), score declared val
    Xd_tr = xgb_matrix(tr, "declared")
    clf = xgb.XGBClassifier(max_depth=3, min_child_weight=40, reg_lambda=1.0, random_state=2026, **XGB_FIXED)
    clf.fit(Xd_tr, ytr, eval_set=[(Xd_va, va.claim_10yr.to_numpy())], verbose=False)
    p2 = clf.predict_proba(Xd_va)[:, 1]
    G_tr, G_va = glm_matrix(tr, "declared"), glm_matrix(va, "declared")
    glm = sm.GLM(ytr, sm.add_constant(G_tr), family=sm.families.Binomial()).fit()
    pg = glm.predict(sm.add_constant(G_va, has_constant="add")).to_numpy()
    for name, p in (("XGB regime2 (declared-trained)", p2), ("GLM regime2 (declared-trained)", pg)):
        rows.append({"gamma": g, "variant": name, "rmse_true_view": np.nan, "rmse_declared": rmse(p, va.true_p10),
                     **{f"under_{c}": under(p, va, c) for c in ("diagnosis", "smoking", "multiple")}, "note": ""})
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
print(pd.DataFrame(rows).round(4).to_string(index=False))

# paired per-seed differences (same applicants) from the pre-set run, truthful view RMSE vs true p
R = [json.loads(l) for l in open("docs/armb_val_results.jsonl")]
print("\nPaired per-seed differences in rmse_true (negative = first model better):")
for g in (0.0, 1.0):
    for a, b in (("glm", "arm_a"), ("xgb", "glm"), ("xgb", "arm_a")):
        d = [r["metrics"][a]["rmse_true"] - r["metrics"][b]["rmse_true"] for r in R if r["gamma"] == g]
        print(f"gamma {g}: {a} - {b}: " + ", ".join(f"{x:+.4f}" for x in d))
