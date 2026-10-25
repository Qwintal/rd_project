"""PRELIMINARY game-theory inputs (validation split; test untouched). Pricing = Regime 1 (truthful-trained), each arm with its own trigger:
   Arm A = Arm A pricing + rule trigger ; Arm B = XGBoost pricing + consistency trigger at the SAME number verified.
Definitions (all money in INR over the 10-year term, no discounting):
  d_pi   = term premium saved by the lie = 10 x [premium(true record) - premium(declared record)] under the arm's own pricing
  r      = trigger recall on concealers (prob. a concealer is flagged) ; v = exam accuracy (base 1.0)
  detection probability = r*v
  P*     = d_pi / (r*v)   total loss on detection (forfeited saving + surcharge) that makes truthful disclosure a best response
  c*     = sum over flagged of w*v*d_pi*[concealer] / sum over flagged of w   = expected recovered shortfall per flagged applicant;
           verify a flagged applicant iff cost c < c*.   w re-weights the cohort to a target concealer share pi.
python -m scripts.game_prelim"""
import joblib, numpy as np, pandas as pd
from actuarial_engine.arm_a import TraditionalEngine, premium
from verification.triggers import RuleTrigger, ConsistencyTrigger
pd.set_option("display.width", 170); pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
rows = []
for seed in (2026, 2027, 2028):
    for g in (0.0, 0.5, 1.0):
        df = pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz")
        tr, va = df[df.split == "train"], df[df.split == "val"]
        m = joblib.load(f"models/seed{seed}_gamma{g}.joblib"); E = TraditionalEngine(); E.k = m["arm_a_k"]; B = m["arm_b"]
        sa = va.sum_assured.to_numpy(); conc = (va.concealment_type != "truthful").to_numpy(); ct = va.concealment_type.to_numpy()
        pA = {v: E.score(va, v).p10.to_numpy() for v in ("true", "declared")}
        pB = {v: B.predict(va, v)["xgb"].to_numpy() for v in ("true", "declared")}
        sr, ss = RuleTrigger().score(va), ConsistencyTrigger(seed).fit(tr).score(va)
        flagA = sr >= 1; k = int(flagA.sum()); flagB = np.zeros(len(va), bool); flagB[np.argsort(-ss)[:k]] = True
        s_obs = conc.mean()
        for arm, p, flag in (("Arm A (rule + loadings)", pA, flagA), ("Arm B (consistency + XGBoost)", pB, flagB)):
            dpi = 10 * (premium(p["true"], sa) - premium(p["declared"], sa)); dpi = np.where(conc, dpi, 0.0)
            r = {"seed": seed, "gamma": g, "arm": arm, "n_flagged": k, "mean_dpi_conc": dpi[conc].mean(), "median_dpi_conc": np.median(dpi[conc]),
                 "recall_all": (flag & conc).sum() / conc.sum()}
            for t in ("smoking", "diagnosis", "multiple"):
                mt = ct == t
                r[f"dpi_{t}"] = dpi[mt].mean(); r[f"recall_{t}"] = (flag & mt).sum() / mt.sum()
            for pi in (s_obs, 0.10):
                w = np.where(conc, pi / s_obs, (1 - pi) / (1 - s_obs))
                r[f"cstar_pi{'design' if pi == s_obs else '10'}"] = (w[flag] * dpi[flag]).sum() / w[flag].sum()    # v = 1
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv("docs/game_prelim.csv", index=False)
G = R.groupby(["gamma", "arm"]).mean(numeric_only=True)
print("=== mean over 3 seeds; INR over the 10-year term ===")
print(G[["mean_dpi_conc", "recall_all", "cstar_pidesign", "cstar_pi10"]].to_string())
print("\n=== by concealment type, gamma 0.5 ===")
g5 = G.loc[0.5]
print(g5[["dpi_smoking", "dpi_diagnosis", "dpi_multiple", "recall_smoking", "recall_diagnosis", "recall_multiple"]].to_string())
print("\n=== applicant deterrence threshold P* = d_pi / (r v), gamma 0.5, v = 1.0 and 0.8 (INR) ===")
for arm in g5.index:
    for t in ("diagnosis", "multiple", "smoking"):
        d, r = g5.loc[arm, f"dpi_{t}"], g5.loc[arm, f"recall_{t}"]
        print(f"{arm:32s} {t:10s} d_pi {d:10,.0f}  recall {r:5.3f}  P*(v=1) {d / r:14,.0f}   P*(v=.8) {d / (r * .8):14,.0f}")
