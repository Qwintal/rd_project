"""Score the TEST split once with every frozen model/trigger -> data/test_predictions.csv.gz  (python -m scripts.phase7_predict)
R1 = truthful-trained (scored on truthful 'T' and declared 'D' views); R2 = declared-trained (declared view only).
Arm A+ and the consistency trigger are re-fitted deterministically from TRAIN only."""
import joblib, numpy as np, pandas as pd
from actuarial_engine.arm_a import TraditionalEngine
from actuarial_engine.arm_a_plus import TraditionalEnginePlus
from data_generation.generate import load_config
from verification.triggers import RuleTrigger, ConsistencyTrigger, hidden_loss

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
FRACS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
cfg = load_config("config/default.yaml")
frames = []
for seed in (2026, 2027, 2028):
    for g in GAMMAS:
        df = pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz")
        tr, te = df[df.split == "train"], df[df.split == "test"].copy()
        m1, m2 = joblib.load(f"models/seed{seed}_gamma{g}.joblib"), joblib.load(f"models/seed{seed}_gamma{g}_declared.joblib")
        E1, E2 = TraditionalEngine(), TraditionalEngine(); E1.k, E2.k = m1["arm_a_k"], m2["arm_a_k"]
        Eplus = TraditionalEnginePlus().fit(tr, "declared")
        o = te[["applicant_id", "concealment_type", "concealed_fields", "true_smoker", "age", "sum_assured", "true_p10", "claim_10yr", "gamma"]].copy()
        o.insert(0, "seed", seed)
        for view, tag in (("true", "T"), ("declared", "D")):
            s = E1.score(te, view); o[f"p_arm_a_R1_{tag}"], o[f"dec_arm_a_R1_{tag}"] = s.p10, s.decision
            b = m1["arm_b"].predict(te, view)
            for k in ("glm", "cox", "xgb"):
                o[f"p_{k}_R1_{tag}"] = b[k]
        o["p_arm_a_R2_D"] = E2.score(te, "declared").p10
        o["p_arm_a_plus_R2_D"] = Eplus.score(te, "declared").p10
        b = m2["arm_b"].predict(te, "declared")
        for k in ("glm", "cox", "xgb"):
            o[f"p_{k}_R2_D"] = b[k]
        o["s_rule"], o["s_cons"] = RuleTrigger().score(te), ConsistencyTrigger(seed).fit(tr).score(te)
        o["p_decl_oracle"], o["shortfall"] = hidden_loss(te, cfg)
        order = np.argsort(-o["s_cons"].to_numpy())
        k = int((o["s_rule"] >= 1).sum()); o["flag_rule"] = (o["s_rule"] >= 1).astype(int)
        m_ = np.zeros(len(o), int); m_[order[:k]] = 1; o["flag_cons_matched"] = m_
        for f in FRACS:
            t_ = np.zeros(len(o), int); t_[order[:int(f * len(o))]] = 1; o[f"flag_cons_top{f}"] = t_
        frames.append(o)
        print(f"seed {seed} gamma {g}: {len(o)} test applicants scored", flush=True)
P = pd.concat(frames, ignore_index=True)
P.to_csv("data/test_predictions.csv.gz", index=False, compression="gzip")
print(P.shape, "| NaN:", int(P.isna().sum().sum()))
