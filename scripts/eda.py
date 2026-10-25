"""EDA of the simulated data (seed 2026, gamma 0.5; the gamma-gap uses all five gamma files of seed 2026).
Run from the project folder:   python -m scripts.eda
Writes 7 figures and eda_results.txt into docs/eda/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

SEED, GAMMA = 2026, 0.5
OUT = "docs/eda"
os.makedirs(OUT, exist_ok=True)
NAVY, STEEL, RED, TEAL, AMBER, GRAY = "#1B2535", "#4C6A92", "#C8553D", "#2A9D8F", "#F2A33A", "#8A94A6"
df = pd.read_csv(f"data/cohort_seed{SEED}_gamma{GAMMA}.csv.gz")
lines = []


def say(s=""):
    print(s)
    lines.append(str(s))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=120)
    plt.close()


say(f"EDA of the simulated data: {len(df):,} applicants, {int(df.claim_10yr.sum()):,} claims within 10 years (seed {SEED}, gamma {GAMMA})")

# ------------------------------------------------------------------ 1. age, BMI, systolic BP
say("\n1. Age, BMI and systolic BP distributions")
say(df[["age", "bmi", "sbp"]].describe().loc[["mean", "std", "min", "50%", "max"]].round(1).to_string())
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for sx, col in (("M", STEEL), ("F", RED)):
    ax[0].hist(df[df.sex == sx].age, bins=21, alpha=.6, color=col, label="Male" if sx == "M" else "Female")
ax[0].set(title="Age", xlabel="years", ylabel="applicants", xticks=range(40, 61, 5)); ax[0].legend()
ax[1].hist(df.bmi, bins=50, color=GRAY); ax[1].axvline(30, color=RED, ls="--"); ax[1].set(title="BMI (red line = 30)", xlabel="kg/m2")
ax[2].hist(df.sbp, bins=np.arange(84, 182, 2), color=GRAY); ax[2].axvline(140, color=RED, ls="--"); ax[2].set(title="Systolic blood pressure (red line = 140)", xlabel="mmHg")
save("1_distributions")

# ------------------------------------------------------------------ 2. smoker share by age
say("\n2. Smoker share by age")
by = df.groupby("age").true_smoker.mean() * 100
bys = df.groupby(["age", "sex"]).true_smoker.mean().unstack() * 100
df["age_band"] = pd.cut(df.age, [39, 44, 49, 54, 60], labels=["40-44", "45-49", "50-54", "55-60"])
say((df.groupby("age_band", observed=True).true_smoker.mean() * 100).round(1).rename("smokers %").to_string())
say(f"Overall smoker share: {df.true_smoker.mean():.1%} (male {df[df.sex == 'M'].true_smoker.mean():.1%}, female {df[df.sex == 'F'].true_smoker.mean():.1%}). Oversampled on purpose so the 65/15/12/8 mix is possible.")
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(by.index, by.values, color=NAVY, lw=2.5, label="All")
ax.plot(bys.index, bys["M"], color=STEEL, ls="--", label="Male"); ax.plot(bys.index, bys["F"], color=RED, ls="--", label="Female")
ax.set(title="Share of smokers by age", xlabel="age", ylabel="% smokers", ylim=(0, 60), xticks=range(40, 61, 5)); ax.legend()
save("2_smoker_share_by_age")

# ------------------------------------------------------------------ 3. correlation heatmap
say("\n3. Correlations (Pearson)")
cols = {"age": df.age, "BMI": df.bmi, "systolic BP": df.sbp, "cholesterol": df.cholesterol, "HbA1c": df.hba1c, "pack-years": df.true_pack_years,
        "smoker": df.true_smoker, "diabetes": df.true_diabetes, "hypertension": df.true_hypertension, "heart disease": df.true_cvd}
cm = pd.DataFrame(cols).corr()
for a, b in (("age", "systolic BP"), ("BMI", "HbA1c"), ("BMI", "systolic BP"), ("smoker", "pack-years")):
    say(f"  {a} ~ {b}: r = {cm.loc[a, b]:.2f}")
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right"); ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(cm.values[i, j]) > .6 else "black")
plt.colorbar(im, fraction=.046); ax.set_title("Correlation heatmap")
save("3_correlation_heatmap")

# ------------------------------------------------------------------ 4. claim rate by age band and by risk factor
say("\n4. Claim rate within 10 years (realised claims; not adjusted for other factors)")
ab = df.groupby("age_band", observed=True).claim_10yr.agg(["sum", "count"])
ab["rate"] = ab["sum"] / ab["count"] * 100
ab[["lo", "hi"]] = [(wilson(k, n)[0] * 100, wilson(k, n)[1] * 100) for k, n in zip(ab["sum"], ab["count"])]
say(ab.round(2).rename(columns={"sum": "claims", "count": "applicants", "rate": "claim rate %", "lo": "95% low", "hi": "95% high"}).to_string())
factors = {"Smoker": df.true_smoker == 1, "Hypertension": df.true_hypertension == 1, "Diabetes": df.true_diabetes == 1, "Heart disease": df.true_cvd == 1,
           "BMI 30 or more": df.bmi >= 30, "Cholesterol 240+": df.cholesterol >= 240, "Heavy alcohol": df.alcohol == 2, "Family history": df.family_history == 1,
           "Hazardous hobby": df.hazardous_hobby == 1, "Occupation class 4": df.occupation_class == 4}
rows = []
for name, m in factors.items():
    k1, n1, k0, n0 = df.claim_10yr[m].sum(), m.sum(), df.claim_10yr[~m].sum(), (~m).sum()
    rows.append({"factor": name, "n_with": int(n1), "rate_with_%": 100 * k1 / n1, "rate_without_%": 100 * k0 / n0, "ratio": (k1 / n1) / (k0 / n0)})
fr = pd.DataFrame(rows).set_index("factor")
say(fr.round(2).to_string())
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
ax[0].bar(ab.index.astype(str), ab.rate, yerr=[ab.rate - ab.lo, ab.hi - ab.rate], color=STEEL, capsize=4)
for i, (r, n) in enumerate(zip(ab.rate, ab["count"])):
    ax[0].text(i, 0.2, f"n={n:,}", ha="center", color="white", fontsize=9)
ax[0].set(title="Claim rate by age band (95% interval)", ylabel="% claiming within 10 years")
y = np.arange(len(fr))
ax[1].barh(y + .2, fr["rate_without_%"], .4, color=GRAY, label="without"); ax[1].barh(y - .2, fr["rate_with_%"], .4, color=RED, label="with")
for i, r in enumerate(fr["ratio"]):
    ax[1].text(max(fr["rate_with_%"].iloc[i], fr["rate_without_%"].iloc[i]) + .15, i, f"x{r:.2f}", va="center", fontsize=9)
ax[1].set_yticks(y); ax[1].set_yticklabels(fr.index); ax[1].invert_yaxis(); ax[1].set(title="Claim rate by risk factor (x = ratio with / without)", xlabel="% claiming within 10 years"); ax[1].legend()
save("4_claim_rate_by_age_and_risk_factor")

# ------------------------------------------------------------------ 5. combined-risk gap vs gamma
say("\n5. Combined-risk gap vs gamma (smoker + hypertension + BMI 30+, compared with none of them; true 10-year probability)")
g_rows = []
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    x = pd.read_csv(f"data/cohort_seed{SEED}_gamma{g}.csv.gz")
    s, h, b = x.true_smoker == 1, x.true_hypertension == 1, x.bmi >= 30
    ref = x.true_p10[~s & ~h & ~b].mean()
    r1 = {k: x.true_p10[m].mean() / ref for k, m in {"S": s & ~h & ~b, "H": h & ~s & ~b, "B": b & ~s & ~h}.items()}
    shb = s & h & b
    g_rows.append({"gamma": g, "n_with_all_three": int(shb.sum()), "combined risk (x)": x.true_p10[shb].mean() / ref, "if risks only multiplied (x)": r1["S"] * r1["H"] * r1["B"]})
gp = pd.DataFrame(g_rows).set_index("gamma")
gp["extra from interaction (x)"] = gp["combined risk (x)"] / gp["if risks only multiplied (x)"]
say(gp.round(2).to_string())
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(gp.index, gp["combined risk (x)"], "o-", color=RED, lw=2.5, label="combined group (smoker + hypertension + BMI 30+)")
ax.plot(gp.index, gp["if risks only multiplied (x)"], "s--", color=GRAY, label="if single-factor risks only multiplied")
for g_, v in zip(gp.index, gp["combined risk (x)"]):
    ax.text(g_, v + .45, f"{v:.1f}x", ha="center", fontsize=9)
ax.set(title="The gap opens as gamma grows", xlabel="interaction strength gamma", ylabel="risk compared with none of the three", ylim=(0, 10)); ax.legend(fontsize=8, loc="upper left")
save("5_combined_risk_gap_vs_gamma")

# ------------------------------------------------------------------ 6. group sizes
say("\n6. Group sizes")
names = {"truthful": "Honest", "smoking": "Hides smoking", "diagnosis": "Hides a disease", "multiple": "Hides both"}
gs = df.concealment_type.map(names).value_counts().reindex(list(names.values()))
say(pd.DataFrame({"applicants": gs, "share %": (gs / len(df) * 100).round(1)}).to_string())
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(gs.index, gs.values, color=[TEAL, RED, AMBER, "#7B5EA7"])
for i, v in enumerate(gs.values):
    ax.text(i, v + 300, f"{v:,}\n({v / len(df):.0%})", ha="center", fontsize=9)
ax.set(title="Applicants by honesty group", ylabel="applicants", ylim=(0, gs.max() * 1.18))
save("6_group_sizes")

# ------------------------------------------------------------------ 7. footprint plot
say("\n7. Footprints: why disease lies are catchable and smoking lies are not")
hid_dm = df.concealed_fields.str.contains("diabetes")
hon_nodm = (df.concealment_type == "truthful") & (df.true_diabetes == 0)
hid_sm = df.concealed_fields.str.contains("smoker")
hon_nosm = (df.concealment_type == "truthful") & (df.true_smoker == 0)


def compare(name, col, a, b):
    xa, xb = df[col][a], df[col][b]
    pooled = np.sqrt((xa.var() + xb.var()) / 2)
    auc = roc_auc_score(np.r_[np.ones(len(xa)), np.zeros(len(xb))], np.r_[xa, xb])
    say(f"  {name}: hidden n={len(xa):,}, median {xa.median():.1f}; honest n={len(xb):,}, median {xb.median():.1f}; difference in means = {(xa.mean() - xb.mean()) / pooled:.2f} standard deviations; one-number AUC = {auc:.2f}")
    return xa, xb


xa, xb = compare("HbA1c, hidden diabetics vs honest non-diabetics", "hba1c", hid_dm, hon_nodm)
share = (xa >= 6.5).mean()
say(f"  Share of hidden diabetics with HbA1c of 6.5 or more: {share:.0%}; share of honest non-diabetics: {(xb >= 6.5).mean():.1%}")
xc, xd = compare("Cholesterol, hidden smokers vs honest non-smokers", "cholesterol", hid_sm, hon_nosm)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
bins = np.linspace(4, 12, 60)
ax[0].hist(xb, bins=bins, density=True, alpha=.65, color=TEAL, label=f"honest non-diabetics (n={len(xb):,})")
ax[0].hist(xa, bins=bins, density=True, alpha=.65, color=RED, label=f"hidden diabetics (n={len(xa):,})")
ax[0].axvline(6.5, color=NAVY, ls="--"); ax[0].text(6.55, ax[0].get_ylim()[1] * .9, "6.5 = diabetes cut-off", fontsize=9)
ax[0].set(title="Disease lies leave a trace: HbA1c", xlabel="HbA1c (%)", ylabel="density"); ax[0].legend(fontsize=8, loc="upper right")
bins = np.linspace(110, 330, 60)
ax[1].hist(xd, bins=bins, density=True, alpha=.65, color=TEAL, label=f"honest non-smokers (n={len(xd):,})")
ax[1].hist(xc, bins=bins, density=True, alpha=.65, color=RED, label=f"hidden smokers (n={len(xc):,})")
ax[1].set(title="Smoking lies leave almost none: cholesterol", xlabel="cholesterol (mg/dL)", ylabel="density"); ax[1].legend(fontsize=8, loc="upper right")
save("7_footprints")

# ------------------------------------------------------------------ 8. adjusted logistic regression
say("\n8. Adjusted logistic regression: claim within 10 years ~ true smoker + age + sex")
X = pd.DataFrame({"true smoker (yes vs no)": df.true_smoker, "age (per year)": df.age, "male (vs female)": (df.sex == "M").astype(int)})
fit = sm.Logit(df.claim_10yr, sm.add_constant(X)).fit(disp=0)
ci = fit.conf_int()
tab = pd.DataFrame({"odds ratio": np.exp(fit.params), "95% low": np.exp(ci[0]), "95% high": np.exp(ci[1]), "p-value": fit.pvalues}).drop("const")
tab["p-value"] = tab["p-value"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
say(f"  {len(df):,} applicants, {int(df.claim_10yr.sum()):,} claims")
say(tab.round(4).to_string())
say(f"  Age per 10 years: odds ratio {np.exp(10 * fit.params['age (per year)']):.2f}")
say("  The simulated smoker hazard ratio is 2.4. The odds ratio here is adjusted only for age and sex (not for the other risk factors),")
say("  so it need not equal 2.4 exactly, but it should be clearly above 1.")
open(f"{OUT}/eda_results.txt", "w", encoding="utf-8").write("\n".join(lines))
print(f"\nSaved 7 figures and eda_results.txt in {OUT}/")
