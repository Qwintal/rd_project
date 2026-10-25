"""Phase 7 analysis exactly as frozen in docs/phase7_protocol.md.   python -m scripts.phase7_analyze"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from actuarial_engine.arm_a import premium
from evaluation.stats import Boot, summarize, holm, delong, calibration

P = pd.read_csv("data/test_predictions.csv.gz")
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
R1 = ["arm_a", "glm", "cox", "xgb"]
R2 = ["arm_a", "arm_a_plus", "glm", "cox", "xgb"]
COMP = [("xgb", "arm_a"), ("glm", "arm_a"), ("cox", "arm_a"), ("xgb", "glm")]
CS = [5_000, 25_000, 100_000, 250_000]
FRACS = [0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
boot = Boot(45_000, B=1000)
tests, levels, secondary, delong_rows, desc = [], [], [], [], {k: [] for k in ("bands", "rates", "portfolio", "expC", "expC_dec", "expD", "expE")}

for g in GAMMAS:
    d = P[P.gamma == g].reset_index(drop=True)
    assert len(d) == 45_000
    t, sa = d.true_p10.to_numpy(), d.sum_assured.to_numpy()
    conc = (d.concealment_type != "truthful").to_numpy().astype(float)
    seed = d.seed.to_numpy()
    prem_t = premium(t, sa)
    cols = {"conc": conc}
    for m in R1:
        for v in ("T", "D"):
            p = d[f"p_{m}_R1_{v}"].to_numpy()
            cols[f"se_{m}_R1_{v}"] = (p - t) ** 2
            cols[f"ape_{m}_R1_{v}"] = np.abs(premium(p, sa) - prem_t)
        cols[f"under_{m}_R1_D"] = conc * 100 * (t - d[f"p_{m}_R1_D"].to_numpy()) / t
    for m in R2:
        cols[f"se_{m}_R2_D"] = (d[f"p_{m}_R2_D"].to_numpy() - t) ** 2
    hl = d.shortfall.to_numpy() * sa
    cols["hl"] = hl
    for nm in ("rule", "cons_matched"):
        fl = d[f"flag_{nm}"].to_numpy()
        cols[f"hlf_{nm}"], cols[f"cf_{nm}"] = hl * fl, conc * fl
    names = list(cols); col = {n: i for i, n in enumerate(names)}
    M = np.column_stack([cols[n] for n in names]); mu = M.mean(0); BM = boot.means(M)
    mus = {s: M[seed == s].mean(0) for s in np.unique(seed)}

    def run(family, label, f, per_seed=False, alt_p=None, **extra):
        r = summarize(f(mu), f(BM)); r.update(family=family, gamma=g, label=label, **extra)
        if per_seed:
            r["seeds_negative"] = int(sum(f(mus[s]) < 0 for s in mus))
        if alt_p is not None:
            r["p_alt"] = alt_p
        tests.append(r)

    rmse = lambda n: (lambda m_: np.sqrt(m_[..., col[n]]))
    # ---------------- F1: RQ1 RMSE, R1 truthful ----------------
    for a, b in COMP:
        run("F1 RQ1 RMSE (truthful view, R1)", f"{a} - {b}", lambda m_, a=a, b=b: np.sqrt(m_[..., col[f"se_{a}_R1_T"]]) - np.sqrt(m_[..., col[f"se_{b}_R1_T"]]),
            per_seed=True, rmse_a=float(np.sqrt(mu[col[f"se_{a}_R1_T"]])), rmse_b=float(np.sqrt(mu[col[f"se_{b}_R1_T"]])))
    # ---------------- F2: RQ2 premium MAE, R1 truthful ----------------
    for a, b in COMP:
        pw = stats.wilcoxon(cols[f"ape_{a}_R1_T"] - cols[f"ape_{b}_R1_T"]).pvalue
        run("F2 RQ2 premium MAE (truthful view, R1)", f"{a} - {b}", lambda m_, a=a, b=b: m_[..., col[f"ape_{a}_R1_T"]] - m_[..., col[f"ape_{b}_R1_T"]],
            per_seed=True, alt_p=float(pw), mae_a=float(mu[col[f"ape_{a}_R1_T"]]), mae_b=float(mu[col[f"ape_{b}_R1_T"]]))
    # ---------------- F3a: RQ3 ----------------
    for m in ("xgb", "glm"):
        run("F3a RQ3 difference-in-differences (R1)", f"DiD RMSE: (ArmA-{m} declared) - (ArmA-{m} truthful)",
            lambda m_, m=m: (np.sqrt(m_[..., col["se_arm_a_R1_D"]]) - np.sqrt(m_[..., col[f"se_{m}_R1_D"]])) - (np.sqrt(m_[..., col["se_arm_a_R1_T"]]) - np.sqrt(m_[..., col[f"se_{m}_R1_T"]])))
    for m in ("xgb", "glm"):
        run("F3a RQ3 underpricing of concealers (R1, declared)", f"under% {m} - arm_a",
            lambda m_, m=m: m_[..., col[f"under_{m}_R1_D"]] / m_[..., col["conc"]] - m_[..., col["under_arm_a_R1_D"]] / m_[..., col["conc"]])
    # ---------------- F3b: declared-view RMSE comparisons ----------------
    f3b = [(("glm", "R2"), ("arm_a_plus", "R2")), (("xgb", "R2"), ("arm_a_plus", "R2")), (("xgb", "R2"), ("glm", "R2"))]
    f3b += [((m, "R2"), (m, "R1")) for m in R1] + [((a, "R1"), (b, "R1")) for a, b in COMP]
    for (a, ra), (b, rb) in f3b:
        run("F3b RQ3 declared-view RMSE", f"{a}[{ra}] - {b}[{rb}]",
            lambda m_, a=a, ra=ra, b=b, rb=rb: np.sqrt(m_[..., col[f"se_{a}_{ra}_D"]]) - np.sqrt(m_[..., col[f"se_{b}_{rb}_D"]]))
    # ---------------- F4: detection at matched budget ----------------
    run("F4 RQ3b detection (matched budget)", "hidden-loss caught %: consistency - rule",
        lambda m_: 100 * (m_[..., col["hlf_cons_matched"]] - m_[..., col["hlf_rule"]]) / m_[..., col["hl"]])
    run("F4 RQ3b detection (matched budget)", "recall %: consistency - rule",
        lambda m_: 100 * (m_[..., col["cf_cons_matched"]] - m_[..., col["cf_rule"]]) / m_[..., col["conc"]])
    # level CIs for figures
    for m in R1:
        for v in ("T", "D"):
            n_ = f"se_{m}_R1_{v}"; r = summarize(np.sqrt(mu[col[n_]]), np.sqrt(BM[:, col[n_]])); levels.append({"gamma": g, "model": m, "regime": "R1", "view": v, **r})
    for m in R2:
        n_ = f"se_{m}_R2_D"; r = summarize(np.sqrt(mu[col[n_]]), np.sqrt(BM[:, col[n_]])); levels.append({"gamma": g, "model": m, "regime": "R2", "view": "D", **r})

    # ---------------- secondary: outcome-based metrics + DeLong (R1 truthful) ----------------
    y = d.claim_10yr.to_numpy()
    for m in R1 + ["oracle"]:
        p = t if m == "oracle" else d[f"p_{m}_R1_T"].to_numpy()
        secondary.append({"gamma": g, "model": m, "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), **calibration(y, p)})
    for a, b in COMP:
        a1, a2, df_, pv = delong(y, d[f"p_{a}_R1_T"].to_numpy(), d[f"p_{b}_R1_T"].to_numpy())
        delong_rows.append({"family": "S1 DeLong AUC (truthful view, R1)", "gamma": g, "label": f"{a} - {b}", "auc_a": a1, "auc_b": a2, "est": df_, "p": pv})

    # ---------------- descriptive: bands, rates, portfolio ----------------
    band = np.where(t < 0.03, "low <3%", np.where(t < 0.08, "medium 3-8%", "high >=8%"))
    s_obs = conc.mean(); w = np.where(conc == 1, 0.10 / s_obs, 0.90 / (1 - s_obs))
    for m in R1:
        p = d[f"p_{m}_R1_T"].to_numpy()
        for b_ in ("low <3%", "medium 3-8%", "high >=8%"):
            k = band == b_
            desc["bands"].append({"gamma": g, "model": m, "band": b_, "n": int(k.sum()), "rmse": float(np.sqrt(np.mean((p[k] - t[k]) ** 2))), "bias_pct": float(100 * np.mean((p[k] - t[k]) / t[k]))})
        desc["rates"].append({"gamma": g, "model": m, "overpriced_pct": float(100 * np.mean(p > 1.10 * t)), "underpriced_pct": float(100 * np.mean(p < 0.90 * t)), "premium_bias_inr": float(np.mean(premium(p, sa) - prem_t))})
    loss_yr = t * sa / 10
    for reg, models, view in (("R1", R1, "T"), ("R1", R1, "D"), ("R2", R2, "D")):
        for m in models:
            prm = premium(d[f"p_{m}_{reg}_{view}"].to_numpy(), sa)
            desc["portfolio"].append({"gamma": g, "model": m, "regime": reg, "view": view, "loss_ratio_unweighted": float(loss_yr.sum() / prm.sum()),
                                      "loss_ratio_10pct_concealers": float((w * loss_yr).sum() / (w * prm).sum())})
    # ---------------- Exp C: within-method concealment effect (R1, concealers) ----------------
    ct = d.concealment_type.to_numpy()
    for typ in ("smoking", "diagnosis", "multiple"):
        k = ct == typ
        for m in R1:
            pt_, pd_ = d[f"p_{m}_R1_T"].to_numpy()[k], d[f"p_{m}_R1_D"].to_numpy()[k]
            dprem = premium(pd_, sa[k]) - premium(pt_, sa[k]); derr = np.abs(pd_ - t[k]) * sa[k] - np.abs(pt_ - t[k]) * sa[k]
            desc["expC"].append({"gamma": g, "type": typ, "model": m, "n": int(k.sum()), "mean_dp": float((pd_ - pt_).mean()), "mean_dp_pct_of_truthful": float(100 * np.mean((pd_ - pt_) / pt_)),
                                 "mean_dpremium_inr_yr": float(dprem.mean()), "se_dpremium": float(dprem.std(ddof=1) / np.sqrt(k.sum())), "mean_change_abs_expected_loss_error_inr": float(derr.mean())})
    kc = conc == 1
    x = pd.crosstab(d.dec_arm_a_R1_T[kc], d.dec_arm_a_R1_D[kc], normalize="index").round(3)
    for a_ in x.index:
        for b_ in x.columns:
            desc["expC_dec"].append({"gamma": g, "truthful_decision": a_, "declared_decision": b_, "share_of_concealers_with_that_truthful_decision": float(x.loc[a_, b_])})
    # ---------------- Exp D: subgroups (declared view) ----------------
    honest = conc == 0; smk = d.true_smoker.to_numpy() == 1
    groups = {"honest non-smokers": honest & ~smk, "honest smokers (matched control)": honest & smk, "smoking concealers": ct == "smoking", "diagnosis concealers": ct == "diagnosis", "multiple concealers": ct == "multiple"}
    for reg, models in (("R1", R1), ("R2", R2)):
        for m in models:
            p = d[f"p_{m}_{reg}_D"].to_numpy()
            for gname, k in groups.items():
                desc["expD"].append({"gamma": g, "regime": reg, "model": m, "group": gname, "n": int(k.sum()), "rmse": float(np.sqrt(np.mean((p[k] - t[k]) ** 2))), "bias_pct": float(100 * np.mean((p[k] - t[k]) / t[k]))})
    # ---------------- Exp E: verification scenarios (10% concealers, R1 pricing) ----------------
    for arm, mdl, strategies in (("Arm A (rule)", "arm_a", [("none", None), ("rule trigger", "flag_rule"), ("blanket", "all")]),
                                 ("Arm B (consistency)", "xgb", [("none", None)] + [(f"top {f:.1%}", f"flag_cons_top{f}") for f in FRACS] + [("matched to rule", "flag_cons_matched"), ("blanket", "all")])):
        rec = 10 * (premium(d[f"p_{mdl}_R1_T"].to_numpy(), sa) - premium(d[f"p_{mdl}_R1_D"].to_numpy(), sa))
        for sname, fcol in strategies:
            fl = np.zeros(len(d)) if fcol is None else (np.ones(len(d)) if fcol == "all" else d[fcol].to_numpy().astype(float))
            nv, rc = (w * fl).sum() / w.sum() * 1000, (w * fl * rec).sum() / w.sum() * 1000
            for c in CS:
                desc["expE"].append({"gamma": g, "arm": arm, "strategy": sname, "c_inr": c, "verified_per_1000": nv, "recovered_inr_per_1000": rc, "cost_inr_per_1000": c * nv, "net_inr_per_1000": rc - c * nv})

# ---------------- Holm within each family ----------------
T = pd.DataFrame(tests)
T["p_holm"] = T.groupby("family")["p"].transform(lambda s: holm(s.to_numpy()))
if "p_alt" in T:
    T["p_alt_holm"] = T.groupby("family")["p_alt"].transform(lambda s: holm(s.fillna(1).to_numpy()) if s.notna().any() else s)
DL = pd.DataFrame(delong_rows); DL["p_holm"] = holm(DL.p.to_numpy())
T.to_csv("docs/phase7_tests.csv", index=False); DL.to_csv("docs/phase7_delong.csv", index=False)
pd.DataFrame(levels).to_csv("docs/phase7_levels.csv", index=False); pd.DataFrame(secondary).to_csv("docs/phase7_secondary.csv", index=False)
D = {k: pd.DataFrame(v) for k, v in desc.items()}
for k, v in D.items():
    v.to_csv(f"docs/phase7_{k}.csv", index=False)

# ---------------- figures ----------------
L = pd.DataFrame(levels); col_ = {"arm_a": "tab:gray", "arm_a_plus": "tab:brown", "glm": "tab:blue", "cox": "tab:green", "xgb": "tab:red"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
for a_, (reg, view, ttl) in zip(ax, (("R1", "T", "R1 truthful-trained, truthful view (Exp A)"), ("R1", "D", "R1 truthful-trained, declared view (Exp B)"), ("R2", "D", "R2 declared-trained, declared view"))):
    for m in (R2 if reg == "R2" else R1):
        s = L[(L.model == m) & (L.regime == reg) & (L.view == view)].sort_values("gamma")
        a_.plot(s.gamma, s.est, marker="o", color=col_[m], label=m); a_.fill_between(s.gamma, s.lo, s.hi, color=col_[m], alpha=.18)
    a_.set(title=ttl, xlabel="gamma", ylabel="RMSE vs true p10"); a_.grid(alpha=.3); a_.legend(fontsize=8)
plt.tight_layout(); plt.savefig("docs/fig7_rmse_gamma.png", dpi=110); plt.close()

e = D["expD"][(D["expD"].gamma == 0.5)]
fig, ax = plt.subplots(figsize=(11, 4.6)); gl = list(e.group.unique()); wdt = 0.8 / 9
combos = [(r, m) for r, ms in (("R1", R1), ("R2", R2)) for m in ms]
for i, (r, m) in enumerate(combos):
    s = e[(e.regime == r) & (e.model == m)].set_index("group").loc[gl]
    ax.bar(np.arange(len(gl)) + i * wdt, s.bias_pct, wdt, label=f"{m} [{r}]", color=col_[m], alpha=1 if r == "R1" else .55, hatch="" if r == "R1" else "//")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(np.arange(len(gl)) + .4); ax.set_xticklabels(gl, rotation=12, fontsize=8)
ax.set(ylabel="mean (price - true)/true, %   (+ = overcharged)", title="Who is mispriced? declared view, gamma 0.5, test split"); ax.legend(fontsize=7, ncol=3)
plt.tight_layout(); plt.savefig("docs/fig7_bias_by_group.png", dpi=110); plt.close()

ee = D["expE"][(D["expE"].gamma == 0.5)]
fig, ax = plt.subplots(figsize=(9, 4.6))
for (arm, st), s in ee.groupby(["arm", "strategy"]):
    if st in ("none", "top 2.5%", "top 15.0%", "top 30.0%"):
        continue
    ax.plot(s.c_inr, s.net_inr_per_1000 / 1e6, marker="o", label=f"{arm}: {st}")
ax.set(xscale="log", xlabel="verification cost per applicant verified (INR)", ylabel="net recovered premium per 1,000 applicants (INR million, term)",
       title="Experiment E: verification scenarios (gamma 0.5, 10% concealers)"); ax.axhline(0, color="k", lw=.8); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("docs/fig7_expE.png", dpi=110); plt.close()

# ---------------- console summary ----------------
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 60); pd.set_option("display.float_format", lambda x: f"{x:.4f}")
for fam in T.family.unique():
    s = T[T.family == fam]
    cols_ = ["gamma", "label", "est", "lo", "hi", "p", "p_holm"] + [c for c in ("seeds_negative",) if c in s]
    if fam.startswith("F3b"):
        s = s[s.gamma.isin([0.0, 1.0])]
    print(f"\n##### {fam}  (n tests in family: {int((T.family == fam).sum())}; Holm-significant: {int((T[T.family == fam].p_holm < 0.05).sum())})")
    print(s[cols_].to_string(index=False))
print("\n##### DeLong AUC differences (secondary)\n", DL[["gamma", "label", "auc_a", "auc_b", "est", "p", "p_holm"]].to_string(index=False))
print("\n##### Outcome-based metrics (R1, truthful view), gamma 0 and 1\n", pd.DataFrame(secondary).query("gamma in [0.0, 1.0]").to_string(index=False))
