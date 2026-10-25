"""Validate the generated grid.  python -m scripts.validate_datasets   (after make_datasets)

Primary file: seed 2026, gamma 0.5.  Writes docs/validation_report.md, docs/fig_validation.png,
docs/fig_gamma_sweep.png.  Every PASS/FAIL line is an assertion about the data, not an opinion.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
load = lambda seed, g: pd.read_csv(f"data/cohort_seed{seed}_gamma{g}.csv.gz")
df = load(2026, 0.5)
man = pd.read_csv("docs/dataset_manifest.csv")
out, checks = [], []

def say(s=""):
    out.append(s)

def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))

# ---------------- integrity ----------------
check("No missing values in any column", df.isna().sum().sum() == 0)
check("applicant_id unique (one row per applicant => no pairing/leakage across splits)", df.applicant_id.is_unique)
check("train/val/test all present", set(df.split) == {"train", "val", "test"},
      str(df.split.value_counts(normalize=True).round(3).to_dict()))
for c in ["smoker", "pack_years", "diabetes", "hypertension", "cvd"]:
    check(f"declared_{c} never exceeds true_{c} (lies only hide, never invent)", (df[f"declared_{c}"] <= df[f"true_{c}"]).all())
tr = df.concealment_type == "truthful"
check("truthful rows: declared == true on every concealable field",
      all((df.loc[tr, f"declared_{c}"] == df.loc[tr, f"true_{c}"]).all() for c in ["smoker", "pack_years", "diabetes", "hypertension", "cvd"]))
sm = df.concealed_fields.str.contains("smoker")
check("smoking concealers declare smoker=0 and pack_years=0", ((df.loc[sm, "declared_smoker"] == 0) & (df.loc[sm, "declared_pack_years"] == 0)).all())
check("true_p10 in (0,1) and mean equals target 5.5% (+-0.1pp)", df.true_p10.between(0, 1).all() and abs(df.true_p10.mean() - 0.055) < 0.001,
      f"mean={df.true_p10.mean():.4f}")
check("realized claim rate within 3 SE of mean p10_given_frailty",
      abs(df.claim_10yr.mean() - df.p10_given_frailty.mean()) < 3 * np.sqrt(0.055 * 0.945 / len(df)),
      f"claim rate={df.claim_10yr.mean():.4f}, mean p given frailty={df.p10_given_frailty.mean():.4f}")
check("frailty Z has mean ~1 and variance ~0.25", abs(df.frailty_z.mean() - 1) < 0.02 and abs(df.frailty_z.var() - 0.25) < 0.02,
      f"mean={df.frailty_z.mean():.3f}, var={df.frailty_z.var():.3f}")
# split identical across gamma and seeds' shares
splits = [load(2026, g).split for g in (0.0, 1.0)]
check("split identical at gamma=0 and gamma=1 (same people in same split)", (splits[0] == splits[1]).all())

# ---------------- concealment mix ----------------
mix = df.concealment_type.value_counts(normalize=True).round(4)
say("## 1. Concealment mix (target 65 / 15 / 12 / 8)")
say(mix.to_frame("share").to_markdown())
mix_split = pd.crosstab(df.split, df.concealment_type, normalize="index").round(3)
say("\nShare by split (stratified, should match):")
say(mix_split.to_markdown())
check("concealment mix within 0.1pp of 65/15/12/8", abs(mix["truthful"] - .65) < .001 and abs(mix["smoking"] - .15) < .001
      and abs(mix["diagnosis"] - .12) < .001 and abs(mix["multiple"] - .08) < .001)
smk_total = df.true_smoker.sum()
say(f"\nSmoker share {df.true_smoker.mean():.1%}; share of smokers who conceal smoking = {sm.sum() / smk_total:.0%} (design consequence of oversampling).")
say(f"Honest smokers (matched control for smoking concealers): {int(((df.true_smoker == 1) & ~sm).sum()):,}")

# ---------------- prevalence & correlations ----------------
prev = pd.Series({"smoker": df.true_smoker.mean(), "hypertension": df.true_hypertension.mean(), "diabetes": df.true_diabetes.mean(),
                  "CVD": df.true_cvd.mean(), "BMI>=30": (df.bmi >= 30).mean(), "heavy alcohol": (df.alcohol == 2).mean(),
                  "hazardous hobby": df.hazardous_hobby.mean(), "family history": df.family_history.mean(),
                  "male": (df.sex == "M").mean()})
say("\n## 2. Prevalence (design choices; NOT fitted to NHANES microdata)")
say(prev.round(3).to_frame("share").to_markdown())

d = df.assign(male=(df.sex == "M").astype(int), htn=df.true_hypertension, dm=df.true_diabetes, smk=df.true_smoker)
corr = pd.Series({
    "age ~ SBP": d.age.corr(d.sbp), "BMI ~ SBP": d.bmi.corr(d.sbp), "BMI ~ HbA1c": d.bmi.corr(d.hba1c),
    "age ~ HbA1c": d.age.corr(d.hba1c), "smoker ~ pack_years": d.smk.corr(d.true_pack_years),
    "BMI ~ cholesterol": d.bmi.corr(d.cholesterol)})
say("\n## 3. Relationships the generator was meant to produce (Pearson r)")
say(corr.round(3).to_frame("r").to_markdown())
check("age~SBP, BMI~SBP, BMI~HbA1c, smoker~pack_years all positive", (corr[["age ~ SBP", "BMI ~ SBP", "BMI ~ HbA1c", "smoker ~ pack_years"]] > 0.1).all())
cond = pd.DataFrame({
    "HTN rate": d.groupby(d.bmi >= 30).htn.mean(), "diabetes rate": d.groupby(d.bmi >= 30).dm.mean()})
cond.index = ["BMI<30", "BMI>=30"]
say("\nCondition rates by BMI (should rise with BMI):")
say(cond.round(3).to_markdown())
check("HTN and diabetes more common at BMI>=30", (cond.loc["BMI>=30"] > cond.loc["BMI<30"]).all())
foot = pd.DataFrame({
    "SBP | HTN no / yes": [d[d.htn == 0].sbp.mean(), d[d.htn == 1].sbp.mean()],
    "HbA1c | diabetes no / yes": [d[d.dm == 0].hba1c.mean(), d[d.dm == 1].hba1c.mean()],
    "cholesterol | smoker no / yes": [d[d.smk == 0].cholesterol.mean(), d[d.smk == 1].cholesterol.mean()]}, index=["no", "yes"])
say("\nFootprint (insurer-measured values by TRUE condition) - the settings locked in this phase:")
say(foot.round(2).to_markdown())

# ---------------- risk & outcomes ----------------
df["age_band"] = np.where(df.age < 50, "40-49", "50-60")
rb = df.groupby("age_band").agg(n=("claim_10yr", "size"), claim_rate=("claim_10yr", "mean"), mean_true_p10=("true_p10", "mean"))
say("\n## 4. Risk gradients (seed 2026, gamma 0.5)")
say(rb.round(4).to_markdown())
check("claim rate rises with age", rb.loc["50-60", "claim_rate"] > rb.loc["40-49", "claim_rate"])
sx = df.groupby("sex").agg(claim_rate=("claim_10yr", "mean"), mean_true_p10=("true_p10", "mean"))
say(sx.round(4).to_markdown())
bt = df.groupby("concealment_type").agg(n=("true_p10", "size"), mean_true_p10=("true_p10", "mean"), claim_rate=("claim_10yr", "mean"))
say("\nBaseline risk by concealment type - NOTE the confound that motivated amendment A9:")
say(bt.round(4).to_markdown())

# ---------------- power check (all 15 files) ----------------
say("\n## 5. Power check - test-set claim counts per concealment subgroup (min / max over 15 files)")
cols = ["test_claims_truthful", "test_claims_smoking", "test_claims_diagnosis", "test_claims_multiple", "test_claims"]
say(man[cols].agg(["min", "max"]).T.to_markdown())
check("every test subgroup has >= 80 claims in every file", (man[cols[:4]].min() >= 80).all(), str(man[cols[:4]].min().to_dict()))

# ---------------- gamma sweep ----------------
rows = []
for g in GAMMAS:
    x = load(2026, g)
    s, h, bm = x.true_smoker == 1, x.true_hypertension == 1, x.bmi >= 30
    ref = x.true_p10[~s & ~h & ~bm].mean()
    sr = {k: x.true_p10[m].mean() / ref for k, m in {"S": s & ~h & ~bm, "H": h & ~s & ~bm, "B": bm & ~s & ~h}.items()}
    shb = s & h & bm
    rows.append({"gamma": g, "mean_true_p10": x.true_p10.mean(), "n_SHB": int(shb.sum()),
                 "SHB_vs_none": x.true_p10[shb].mean() / ref, "product_of_singles": sr["S"] * sr["H"] * sr["B"]})
gs = pd.DataFrame(rows).set_index("gamma")
gs["excess_over_product"] = gs["SHB_vs_none"] / gs["product_of_singles"]
say("\n## 6. Does gamma open the interaction gap? (smoker + hypertension + BMI>=30, seed 2026)")
say(gs.round(3).to_markdown())
check("gap widens monotonically with gamma", gs.excess_over_product.is_monotonic_increasing)
check("overall level held constant across gamma", gs.mean_true_p10.sub(0.055).abs().max() < 0.001)

# ---------------- figures ----------------
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for sx_, c in [("M", "tab:blue"), ("F", "tab:red")]:
    ax[0, 0].hist(df[df.sex == sx_].age, bins=21, alpha=.5, label=sx_, color=c)
ax[0, 0].set(title="Age by sex", xlabel="age"); ax[0, 0].legend()
ax[0, 1].hist(df.bmi, bins=50, color="tab:gray"); ax[0, 1].axvline(30, color="k", ls="--"); ax[0, 1].set(title="BMI (dashed = 30)", xlabel="BMI")
for v, c in [(0, "tab:green"), (1, "tab:orange")]:
    ax[0, 2].hist(df[df.true_hypertension == v].sbp, bins=50, alpha=.5, color=c, label=f"true HTN={v}")
ax[0, 2].set(title="SBP by true hypertension (footprint)", xlabel="mmHg"); ax[0, 2].legend()
for v, c in [(0, "tab:green"), (1, "tab:orange")]:
    ax[1, 0].hist(df[df.true_diabetes == v].hba1c, bins=50, alpha=.5, color=c, label=f"true diabetes={v}", density=True)
ax[1, 0].set(title="HbA1c by true diabetes (footprint)", xlabel="HbA1c %"); ax[1, 0].legend()
order = ["truthful", "diagnosis", "smoking", "multiple"]
ax[1, 1].boxplot([df[df.concealment_type == t].true_p10 for t in order], labels=order, showfliers=False)
ax[1, 1].set(title="true_p10 by concealment type (baseline-risk confound)", ylabel="true 10-yr probability")
ab = df.groupby("age").claim_10yr.mean().rolling(3, center=True, min_periods=1).mean()
ax[1, 2].plot(ab.index, ab.values, label="realized claim rate (3-yr smooth)")
ax[1, 2].plot(df.groupby("age").true_p10.mean().index, df.groupby("age").true_p10.mean().values, label="mean true_p10")
ax[1, 2].set(title="Risk by age", xlabel="age"); ax[1, 2].legend()
plt.tight_layout(); plt.savefig("docs/fig_validation.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(gs.index, gs.SHB_vs_none, "o-", label="smoker+HTN+BMI>=30 vs no-factor group")
ax[0].plot(gs.index, gs.product_of_singles, "s--", label="product of single-factor ratios")
ax[0].set(title="Interaction gap opens with gamma", xlabel="gamma", ylabel="relative risk"); ax[0].legend()
ax[1].plot(gs.index, gs.mean_true_p10, "o-"); ax[1].set(title="Overall level held constant", xlabel="gamma", ylabel="mean true_p10", ylim=(0.05, 0.06))
plt.tight_layout(); plt.savefig("docs/fig_gamma_sweep.png", dpi=110); plt.close()

# ---------------- report ----------------
head = ["# Validation report (seed 2026, gamma 0.5; grid = 3 seeds x 5 gamma x 50,000)\n", "## 0. Assertions\n",
        "| Check | Result | Detail |\n|---|---|---|"]
for n_, ok, det in checks:
    head.append(f"| {n_} | {'PASS' if ok else '**FAIL**'} | {det} |")
open("docs/validation_report.md", "w").write("\n".join(head) + "\n\n" + "\n".join(out) + "\n\n![](fig_validation.png)\n![](fig_gamma_sweep.png)\n")
print("\n".join(head)); print("\n".join(out))
print("\nFAILED:", [n for n, ok, _ in checks if not ok])
