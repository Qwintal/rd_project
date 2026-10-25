"""Trigger evaluation on the VALIDATION split (neither trigger has anything tuned on validation; the consistency model is fit on train only).
Compares the Arm A rule trigger and the Arm B consistency trigger AT THE SAME NUMBER VERIFIED, with a random-flagging baseline.
hidden loss = SA x (true_p10 - p10 the true hazard gives the declared record): what the lie hides, arm-independent.
python -m scripts.verification_eval"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from data_generation.generate import load_config
from verification.triggers import RuleTrigger, ConsistencyTrigger, hidden_loss
pd.set_option("display.width", 170)
cfg = load_config("config/default.yaml")
FRACS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
rows, aucs = [], []

def metrics(flag, conc, ctype, hl):
    n = len(flag); k = int(flag.sum()); honest = ~conc
    r = {"n_flagged": k, "share_flagged": k / n, "precision": conc[flag].mean() if k else np.nan,
         "recall": (flag & conc).sum() / conc.sum(), "false_verif_rate": (flag & honest).sum() / honest.sum(),
         "hidden_loss_caught_pct": 100 * hl[flag].sum() / hl.sum()}
    for t in ("smoking", "diagnosis", "multiple"):
        m = ctype == t
        r[f"recall_{t}"] = (flag & m).sum() / m.sum()
    return r

for seed in (2026, 2027, 2028):
    for g in (0.0, 0.5, 1.0):
        df = pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz")
        tr, va = df[df.split == "train"], df[df.split == "val"]
        conc = (va.concealment_type != "truthful").to_numpy(); ctype = va.concealment_type.to_numpy()
        hl = (hidden_loss(va, cfg)[1] * va.sum_assured.to_numpy())
        tr_r, tr_s = RuleTrigger(), ConsistencyTrigger(seed).fit(tr)
        sr, ss = tr_r.score(va), tr_s.score(va)
        base = {"seed": seed, "gamma": g}
        flag_r = sr >= 1; k = int(flag_r.sum())
        rows.append({**base, "trigger": "rule (Arm A)", "budget": "rule's own", **metrics(flag_r, conc, ctype, hl)})
        top = np.zeros(len(va), bool); top[np.argsort(-ss)[:k]] = True
        rows.append({**base, "trigger": "consistency (Arm B)", "budget": "matched to rule", **metrics(top, conc, ctype, hl)})
        rnd = {"n_flagged": k, "share_flagged": k / len(va), "precision": conc.mean(), "recall": k / len(va),
               "false_verif_rate": k / len(va), "hidden_loss_caught_pct": 100 * k / len(va),
               **{f"recall_{t}": k / len(va) for t in ("smoking", "diagnosis", "multiple")}}
        rows.append({**base, "trigger": "random baseline", "budget": "matched to rule", **rnd})
        for f in FRACS:
            kk = int(f * len(va)); tp = np.zeros(len(va), bool); tp[np.argsort(-ss)[:kk]] = True
            rows.append({**base, "trigger": "consistency (Arm B)", "budget": f"top {f:.1%}", **metrics(tp, conc, ctype, hl)})
        a = {**base, "auc_consistency_all": roc_auc_score(conc, ss), "auc_rule_all": roc_auc_score(conc, sr)}
        for t in ("smoking", "diagnosis", "multiple"):
            m = (ctype == t) | (~conc)
            a[f"auc_consistency_{t}"] = roc_auc_score(ctype[m] == t, ss[m])
        aucs.append(a)
R, A = pd.DataFrame(rows), pd.DataFrame(aucs)
R.to_csv("docs/verification_results.csv", index=False); A.to_csv("docs/verification_auc.csv", index=False)
cols = ["share_flagged", "precision", "recall", "false_verif_rate", "hidden_loss_caught_pct", "recall_smoking", "recall_diagnosis", "recall_multiple"]
for g in (0.5,):
    s = R[(R.gamma == g) & R.budget.isin(["rule's own", "matched to rule"])].groupby(["trigger"])[cols].mean()
    print(f"\n=== gamma {g}: triggers at the SAME number verified (mean of 3 seeds, validation) ===\n", s.round(3).to_string())
    s = R[(R.gamma == g) & R.budget.str.startswith("top")].groupby("budget")[cols].mean()
    print(f"\n=== gamma {g}: consistency-model budget curve ===\n", s.round(3).to_string())
print("\n=== AUC (concealed vs honest), mean of 3 seeds and 3 gamma ===\n", A.drop(columns=["seed", "gamma"]).mean().round(3).to_string())
print("\nhidden-loss caught by the consistency trigger at the rule's budget, by gamma:",
      R[(R.trigger == "consistency (Arm B)") & (R.budget == "matched to rule")].groupby("gamma").hidden_loss_caught_pct.mean().round(1).to_dict())
print("rule's own:", R[R.trigger == "rule (Arm A)"].groupby("gamma").hidden_loss_caught_pct.mean().round(1).to_dict())
