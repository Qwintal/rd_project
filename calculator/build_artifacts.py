"""Export the light artifacts the app needs (seed 2026 models + pooled-test aggregates).  python -m calculator.build_artifacts
Needs: data/cohort_seed2026_gamma*.csv.gz, models/seed2026_gamma*.joblib (+_declared), data/test_predictions.csv.gz."""
import json
import joblib, numpy as np, pandas as pd, xgboost as xgb
from actuarial_engine.arm_a import premium
from actuarial_engine.arm_a_plus import TraditionalEnginePlus
from data_generation.generate import load_config, generate_base, apply_gamma, feature_view
from ml_models.features import xgb_matrix
from verification.triggers import ConsistencyTrigger
from calculator.logic import ART, GAMMAS

SEED = 2026
cfg = load_config("config/default.yaml", n_applicants=50_000, seed=SEED)
base, E, _ = generate_base(cfg)
P = pd.read_csv("data/test_predictions.csv.gz")
FRACS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
cons = None

for g in GAMMAS:
    _, meta = apply_gamma(base, E, cfg, g)
    df = pd.read_csv(f"data/cohort_seed{SEED}_gamma{g}.csv.gz"); tr, va = df[df.split == "train"], df[df.split == "val"]
    m = {"A0": meta["A0"], "b": meta["b"], "arm_a_k": {}, "xgb_best_iteration": {}}
    for reg, suf, view in (("R1", "", "true"), ("R2", "_declared", "declared")):
        f = joblib.load(f"models/seed{SEED}_gamma{g}{suf}.joblib"); B = f["arm_b"]
        m["arm_a_k"][reg] = f["arm_a_k"]; m["xgb_best_iteration"][reg] = B.info["xgb_params"]["best_iteration"]
        B.xgb.get_booster().save_model(f"{ART}/xgb_{reg}_gamma{g}.json")
        B.glm_table[["coef"]].to_csv(f"{ART}/glm_{reg}_gamma{g}.csv")
        bst = xgb.Booster(); bst.load_model(f"{ART}/xgb_{reg}_gamma{g}.json")
        X = xgb_matrix(va, view)
        diff = np.abs(bst.predict(xgb.DMatrix(X, feature_names=list(X.columns)), iteration_range=(0, m["xgb_best_iteration"][reg] + 1)) - B.xgb.predict_proba(X)[:, 1]).max()
        assert diff < 1e-6, f"XGBoost export parity failed ({reg}, gamma {g}): {diff}"
    e = TraditionalEnginePlus().fit(tr, "declared"); m["arm_a_plus"] = {"k": e.k, "k_cls": {str(k): v for k, v in e.k_cls.items()}}
    json.dump(m, open(f"{ART}/gamma{g}.json", "w"))
    if cons is None:                                   # consistency trigger is label-free and identical across gamma
        cons = ConsistencyTrigger(SEED).fit(tr); joblib.dump(cons, f"{ART}/consistency.joblib")
    print(f"gamma {g}: A0 {meta['A0']:.3e}, XGB parity ok", flush=True)

# ---- suspicion-rank table (design cohort = test split, seed 2026, gamma-independent) ----
te = df[df.split == "test"]
sc = np.sort(cons.score(te)); np.save(f"{ART}/cons_scores_sorted.npy", sc)
conc = (te.concealment_type != "truthful").to_numpy(); s_te = cons.score(te)
pct = 100 * np.array([(sc >= x).mean() for x in s_te]); bins = np.minimum((pct // 5).astype(int), 19)
json.dump({"n_conc": [int(((bins == k) & conc).sum()) for k in range(20)], "n_hon": [int(((bins == k) & ~conc).sum()) for k in range(20)], "share_conc": float(conc.mean())},
          open(f"{ART}/conceal_by_rank.json", "w"))

# ---- portfolio aggregates (3 seeds pooled, test split) ----
KEYS = {"arm_a|R1|T": "p_arm_a_R1_T", "arm_a|R1|D": "p_arm_a_R1_D", "glm|R1|T": "p_glm_R1_T", "glm|R1|D": "p_glm_R1_D", "xgb|R1|T": "p_xgb_R1_T", "xgb|R1|D": "p_xgb_R1_D",
        "arm_a|R2|D": "p_arm_a_R2_D", "arm_a_plus|R2|D": "p_arm_a_plus_R2_D", "glm|R2|D": "p_glm_R2_D", "xgb|R2|D": "p_xgb_R2_D"}
for g in GAMMAS:
    d = P[P.gamma == g].reset_index(drop=True); t = d.true_p10.to_numpy(); sa = d.sum_assured.to_numpy(); ct = d.concealment_type.to_numpy()
    grp = np.where(ct != "truthful", ct, np.where(d.true_smoker.to_numpy() == 1, "honest_smoker", "honest_nonsmoker"))
    pj = {"gamma": g, "groups": {k: {"n": int((grp == k).sum())} for k in ["honest_nonsmoker", "honest_smoker", "smoking", "diagnosis", "multiple"]}, "models": {}, "strategies": {}}
    loss = t * sa / 10
    for key, col in KEYS.items():
        p = d[col].to_numpy(); prem = premium(p, sa)
        pj["models"][key] = {k: {"sum_prem": float(prem[grp == k].sum()), "sum_loss": float(loss[grp == k].sum()), "sum_se": float(((p - t) ** 2)[grp == k].sum()),
                                 "sum_bias": float(((p - t) / t)[grp == k].sum())} for k in pj["groups"]}
    conc_ = ct != "truthful"; pj["n_conc"], pj["n_hon"], pj["design_share_conc"] = int(conc_.sum()), int((~conc_).sum()), float(conc_.mean())
    recA = 10 * (premium(d.p_arm_a_R1_T.to_numpy(), sa) - premium(d.p_arm_a_R1_D.to_numpy(), sa)); recB = 10 * (premium(d.p_xgb_R1_T.to_numpy(), sa) - premium(d.p_xgb_R1_D.to_numpy(), sa))
    ones = np.ones(len(d)); zeros = np.zeros(len(d))
    strat = {"Arm A: none": (zeros, recA), "Arm A: rule trigger": (d.flag_rule.to_numpy(), recA), "Arm A: blanket": (ones, recA), "Arm B: none": (zeros, recB),
             "Arm B: matched to rule": (d.flag_cons_matched.to_numpy(), recB), "Arm B: blanket": (ones, recB)}
    for f in FRACS:
        strat[f"Arm B: top {f:.1%} of ranking"] = (d[f"flag_cons_top{f}"].to_numpy(), recB)
    for name, (fl, rec) in strat.items():
        fl = fl.astype(float)
        pj["strategies"][name] = {"n_flag_conc": float(fl[conc_].sum()), "n_flag_hon": float(fl[~conc_].sum()), "sum_rec_conc": float((fl * rec)[conc_].sum())}
    rk = d.groupby("seed")["s_cons"].rank(ascending=False, method="first") / d.groupby("seed")["s_cons"].transform("size")
    b = np.minimum(((rk - 1e-12) * 100).astype(int), 99).to_numpy()
    pj["marginal"] = {"n_conc": [int(((b == k) & conc_).sum()) for k in range(100)], "n_hon": [int(((b == k) & ~conc_).sum()) for k in range(100)],
                      "sum_rec_conc": [float(recB[(b == k) & conc_].sum()) for k in range(100)]}
    json.dump(pj, open(f"{ART}/portfolio_gamma{g}.json", "w"))
print("artifacts written:", sorted(__import__("os").listdir(ART))[:6], "...")
