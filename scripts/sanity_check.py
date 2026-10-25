"""Phase 2 sanity run.  python -m scripts.sanity_check   (from the project root)

Part A: 1,000 applicants, same people at gamma=0 and gamma=1 (common random numbers).
Part B: 100,000 applicants, deterministic true_p10 only (no outcome noise) to check
        risk-factor ratios and the interaction gap, which 1,000 people cannot resolve.
"""
import numpy as np
import pandas as pd
from data_generation.generate import load_config, generate_base, apply_gamma, assign_split
from data_generation import hazard as hz

pd.set_option("display.width", 140)
cfg = load_config("config/default.yaml")
out = []
def say(s=""):
    print(s); out.append(s)

# ---------------- Part A: 1,000 applicants ----------------
base, E, pools = generate_base(cfg)
n = len(base)
say(f"## Part A - {n} applicants (seed {cfg['seed']})")
say(f"Gompertz slope b (fitted to CDC 2023 male q_x): {hz.fit_gompertz_b():.4f} per year")
say(f"Realized smoker share: {base.true_smoker.mean():.1%}")
cs = base.concealment_type.value_counts()
say(f"Concealment mix realized: " + ", ".join(f"{k} {v/n:.1%}" for k, v in cs.items()))
smokers = base.true_smoker.sum()
conc_smk = ((base.concealment_type == 'smoking') | (base.concealment_type == 'multiple')).sum()
say(f"Implied share of smokers who conceal smoking: {conc_smk/smokers:.0%}  (design consequence of oversampling)")
say(f"Eligibility pool sizes (multi / smoking / diagnosis): {pools}")
say(f"Prevalence: HTN {base.true_hypertension.mean():.1%}, diabetes {base.true_diabetes.mean():.1%}, "
    f"CVD {base.true_cvd.mean():.1%}, BMI>=30 {(base.bmi>=30).mean():.1%}")

res = {}
for g in (0.0, 1.0):
    df, meta = apply_gamma(base, E, cfg, g)
    res[g] = (assign_split(df, cfg["seed"]), meta)
    d = res[g][0]
    say(f"\n### gamma = {g}   (A0 = {meta['A0']:.3e})")
    say(f"mean true p10 = {d.true_p10.mean():.4f} (x-conditional target); mean p10 given hidden frailty = {d.p10_given_frailty.mean():.4f}; claims = {int(d.claim_10yr.sum())} of {n}  (~{d.claim_10yr.mean()*1000:.0f} per 1,000)")
    d = d.assign(age_band=np.where(d.age < 50, "40-49", "50-60"))
    t = d.groupby("age_band").agg(n=("claim_10yr", "size"), claims=("claim_10yr", "sum"),
                                   rate=("claim_10yr", "mean"), mean_true_p10=("true_p10", "mean"))
    say(t.round(4).to_string())
    t = d.groupby("concealment_type").agg(n=("claim_10yr", "size"), claims=("claim_10yr", "sum"),
                                           mean_true_p10=("true_p10", "mean"))
    t["expected_test_claims(30%)"] = (t["claims"] * 0.3).round(1)
    say(t.round(4).to_string())
    say(d.groupby("sex").agg(mean_true_p10=("true_p10", "mean"), claims=("claim_10yr", "sum")).round(4).to_string())

# ---------------- Part B: 100k deterministic check ----------------
cfgB = load_config("config/default.yaml", n_applicants=100_000)
bigB, EB, _ = generate_base(cfgB)
say(f"\n## Part B - {len(bigB):,} applicants, deterministic true_p10")
say("Marginal ratio = mean p10 with factor / mean p10 without (confounded by correlated factors, so can differ from the input HR).")
say("Input HRs: smoker 2.4, hypertension 1.30, BMI>=30 1.26, diabetes 1.80")
rows = []
for g in (0.0, 1.0):
    d, _ = apply_gamma(bigB, EB, cfgB, g)
    s, h, bm, dm = d.true_smoker == 1, d.true_hypertension == 1, d.bmi >= 30, d.true_diabetes == 1
    r = {"gamma": g, "mean_p10": d.true_p10.mean()}
    for name, m in [("smoker", s), ("hypertension", h), ("bmi>=30", bm), ("diabetes", dm)]:
        r[f"ratio_{name}"] = d.true_p10[m].mean() / d.true_p10[~m].mean()
    none = ~s & ~h & ~bm
    ref = d.true_p10[none].mean()
    single = {"S": s & ~h & ~bm, "H": h & ~s & ~bm, "B": bm & ~s & ~h}
    sr = {k: d.true_p10[m].mean() / ref for k, m in single.items()}
    shb = s & h & bm
    r["n_SHB"] = int(shb.sum())
    r["ratio_SHB_vs_none"] = d.true_p10[shb].mean() / ref
    r["product_of_singles"] = sr["S"] * sr["H"] * sr["B"]
    rows.append(r)
tb = pd.DataFrame(rows).set_index("gamma")
say(tb.round(3).T.to_string())
gap = tb.loc[1.0, "ratio_SHB_vs_none"] / tb.loc[0.0, "ratio_SHB_vs_none"]
say(f"\nInteraction gap for smoker+HTN+BMI>=30: ratio(gamma=1)/ratio(gamma=0) = {gap:.2f}   (design value exp(0.50)=1.65 before age/other effects)")

# CDC-implied general-population 10-yr rate for this age/sex mix (shape check, approximate)
ages = np.array([40, 49, 50, 59.0]); q = np.array([0.003176, 0.004859, 0.005182, 0.010581])
b_, c_ = np.polyfit(ages, np.log(q), 1)
a = base.age.to_numpy(float)[:, None] + np.arange(10)[None, :]
sexf = np.where(base.sex == "F", cfg["female_hazard_ratio"], 1.0)[:, None]
cdc10 = (1 - np.exp(-(np.exp(c_ + b_ * a) * sexf).sum(axis=1))).mean()
say(f"\nCDC-implied general-population 10-yr death probability for this age/sex mix (log-linear fit to 4 table points x assumed female ratio): {cdc10:.3f}")
say(f"DGP target mean p10: {cfg['target_p10']:.3f}  (cohort has {base.true_smoker.mean():.0%} smokers, so above general population is expected)")

# ---------------- Part C: what the hidden frailty does to attainable accuracy ----------------
from sklearn.metrics import roc_auc_score, brier_score_loss
say("\n## Part C - oracle ceiling with and without hidden frailty (100,000 applicants, gamma = 0.5)")
say("Oracle = the TRUE x-conditional probability. It is the best any model can do from the form's columns,")
say("so its outcome-based scores are the CEILING; every arm's gap to this ceiling is the honest measure.")
rowsC = []
for th in (0.0, 0.25):
    cfgC = load_config("config/default.yaml", n_applicants=100_000, frailty_theta=th)
    bC, EC, _ = generate_base(cfgC)
    dC, _ = apply_gamma(bC, EC, cfgC, 0.5)
    rowsC.append({"theta": th, "claims": int(dC.claim_10yr.sum()),
                  "oracle_AUC": roc_auc_score(dC.claim_10yr, dC.true_p10),
                  "oracle_Brier": brier_score_loss(dC.claim_10yr, dC.true_p10),
                  "sd(p_given_frailty)/sd(true_p10)": dC.p10_given_frailty.std() / dC.true_p10.std()})
say(pd.DataFrame(rowsC).set_index("theta").round(4).to_string())

res[0.0][0].to_csv("data/sanity_1000_gamma0.csv", index=False)
res[1.0][0].to_csv("data/sanity_1000_gamma1.csv", index=False)
with open("docs/phase2_sanity_report.md", "w") as f:
    f.write("\n".join(out))
