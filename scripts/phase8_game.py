"""Phase 8: game-theory sweeps from the TEST-split predictions (no new fitting).   python -m scripts.phase8_game
Pairings: Arm A = Arm A pricing + rule trigger ; Arm B = XGBoost pricing + consistency trigger. Regime 1 pricing, v = 1 unless stated.
d_pi_i = 10 x [premium(true record) - premium(declared record)] under the arm's own pricing (zero for honest applicants).
Prevalence re-weighting: concealers get weight pi/s, honest 1-pi over 1-s (s = cohort concealer share)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from actuarial_engine.arm_a import premium
from evaluation.stats import Boot, summarize
from game_theory.model import deterrence_threshold, deterrence_multiple, sigma_star, theta_star

P = pd.read_csv("data/test_predictions.csv.gz")
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SA_VALUES = [1_000_000, 2_500_000, 5_000_000, 10_000_000, 20_000_000]
PIS = [0.02, 0.05, 0.10, 0.20, 0.35]
CS = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
TYPES = ["diagnosis", "multiple", "smoking"]
boot = Boot(45_000, B=1000, seed=11)
T1, T2, T2opt, T3, T4, T5, T6 = [], [], [], [], [], [], []
curve = {}

for g in GAMMAS:
    d = P[P.gamma == g].reset_index(drop=True)
    sa = d.sum_assured.to_numpy(); ct = d.concealment_type.to_numpy(); conc = ct != "truthful"; n = len(d)
    rec = {"A": 10 * (premium(d.p_arm_a_R1_T.to_numpy(), sa) - premium(d.p_arm_a_R1_D.to_numpy(), sa)),
           "B": 10 * (premium(d.p_xgb_R1_T.to_numpy(), sa) - premium(d.p_xgb_R1_D.to_numpy(), sa))}
    s_obs = conc.mean()
    W = lambda pi: np.where(conc, pi / s_obs, (1 - pi) / (1 - s_obs))
    ones = np.ones(n)
    flags = {"Arm A blanket": ("A", ones), "Arm A rule": ("A", d.flag_rule.to_numpy().astype(float)),
             "Arm B blanket": ("B", ones), "Arm B matched to rule": ("B", d.flag_cons_matched.to_numpy().astype(float)),
             "Arm B top 5%": ("B", d["flag_cons_top0.05"].to_numpy().astype(float)),
             "Arm B top 10%": ("B", d["flag_cons_top0.1"].to_numpy().astype(float)),
             "Arm B top 20%": ("B", d["flag_cons_top0.2"].to_numpy().astype(float))}
    # ---- T1: headline c* with bootstrap CI (10% concealers, v = 1) ----
    w10 = W(0.10); cols, idx = [], {}
    for name, (arm, fl) in flags.items():
        idx[name] = (len(cols), len(cols) + 1); cols += [w10 * fl * rec[arm], w10 * fl]
    M = np.column_stack(cols); mu, BM = M.mean(0), boot.means(M)
    for name, (i, j) in idx.items():
        r = summarize(mu[i] / mu[j], BM[:, i] / BM[:, j])
        T1.append({"gamma": g, "strategy": name, "share_verified_pct": 100 * mu[j] / w10.mean(), "cstar_inr": r["est"], "lo": r["lo"], "hi": r["hi"]})
    # ---- T2: marginal benefit by suspicion rank (Arm B pricing, consistency ranking) ----
    rk = d.groupby("seed")["s_cons"].rank(ascending=False, method="first") / d.groupby("seed")["s_cons"].transform("size")
    b = np.minimum(((rk - 1e-12) * 100).astype(int), 99).to_numpy()
    Wb = np.array([w10[b == k].sum() for k in range(100)]); Rb = np.array([(w10 * rec["B"])[b == k].sum() for k in range(100)])
    marg = Rb / Wb; curve[g] = marg
    for k in range(100):
        T2.append({"gamma": g, "verified_share_pct": k + 1, "marginal_cstar_inr": marg[k], "cumulative_avg_cstar_inr": Rb[:k + 1].sum() / Wb[:k + 1].sum()})
    for c in CS:
        net = np.concatenate([[0.0], np.cumsum(Rb - c * Wb) / Wb.sum() * 1000]); k = int(net.argmax())
        cumW = np.concatenate([[0.0], np.cumsum(Wb) / Wb.sum()])
        T2opt.append({"gamma": g, "c_inr": c, "optimal_share_verified_pct_of_cohort": k, "optimal_share_verified_pct_at_10pct_prevalence": 100 * cumW[k],
                      "net_inr_per_1000": net[k], "blanket_net_inr_per_1000": net[-1]})
    # ---- T3: by sum assured (c* per verified applicant) ----
    for sv in SA_VALUES:
        band = sa == sv
        for name in ("Arm A blanket", "Arm A rule", "Arm B top 10%"):
            arm, fl = flags[name]; m_ = band & (fl > 0)
            T3.append({"gamma": g, "sum_assured": sv, "strategy": name, "n_flagged": int(m_.sum()),
                       "cstar_inr": (w10 * rec[arm])[m_].sum() / w10[m_].sum()})
    # ---- T4: prevalence sweep (v scales c* linearly) ----
    for pi in PIS:
        w = W(pi)
        for name in ("Arm A blanket", "Arm A rule", "Arm B blanket", "Arm B top 10%"):
            arm, fl = flags[name]
            T4.append({"gamma": g, "prevalence": pi, "strategy": name, "cstar_v1.0": (w * fl * rec[arm]).sum() / (w * fl).sum()})
    # ---- T5: applicant-side deterrence by trigger operating point and concealment type ----
    honest = ~conc
    for name in ("Arm A rule", "Arm B matched to rule", "Arm B top 10%", "Arm B top 20%"):
        arm, fl = flags[name]; fvr = (fl[honest] > 0).mean()
        for t in TYPES:
            m_ = ct == t; r_ = (fl[m_] > 0).mean(); dpi = rec[arm][m_].mean()
            for v in (1.0, 0.8):
                T5.append({"gamma": g, "operating_point": name, "type": t, "n": int(m_.sum()), "mean_dpi_inr": dpi, "median_dpi_inr": float(np.median(rec[arm][m_])),
                           "recall_r": r_, "false_verif_rate_f": fvr, "v": v, "penalty_multiple_1_over_rv": deterrence_multiple(r_, v) if r_ > 0 else np.inf,
                           "P_star_inr": deterrence_threshold(dpi, r_, v) if r_ > 0 else np.inf, "c_max_inr_v_dpi": v * dpi})
    if g == 0.5:                                  # illustrative equilibrium, Arm B top-10% operating point
        arm, fl = flags["Arm B top 10%"]; fvr = (fl[honest] > 0).mean()
        for t in TYPES:
            m_ = ct == t; r_ = (fl[m_] > 0).mean(); dpi = rec[arm][m_].mean()
            for c in (10_000, 50_000):
                for mult in (3, 10):
                    L = mult * dpi; det = L >= deterrence_threshold(dpi, r_, 1.0); th = theta_star(dpi, r_, fvr, 1.0, c)
                    if det and th is not None:
                        eq, sg, tt = "mixed", sigma_star(dpi, L, r_, 1.0), th
                    elif not det:
                        eq, sg, tt = ("pure: applicant conceals; insurer verifies" if th is not None else "pure: applicant conceals; insurer does not verify"), np.nan, 1.0
                    else:
                        eq, sg, tt = "pure: insurer never verifies, applicant conceals", np.nan, 1.0
                    T6.append({"type": t, "mean_dpi_inr": dpi, "r": r_, "f": fvr, "c_inr": c, "penalty_multiple_L_over_dpi": mult,
                               "equilibrium": eq, "sigma_star_verify_prob": sg, "theta_star_conceal_prob": tt})

T1, T2, T2opt, T3, T4, T5, T6 = map(pd.DataFrame, (T1, T2, T2opt, T3, T4, T5, T6))
for k, v in zip(("t1_headline", "t2_marginal", "t2_optimal", "t3_sa", "t4_prevalence", "t5_deterrence", "t6_equilibrium"), (T1, T2, T2opt, T3, T4, T5, T6)):
    v.to_csv(f"docs/phase8_{k}.csv", index=False)

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(1, 101)
for g, c_ in ((0.0, "tab:blue"), (0.5, "tab:orange"), (1.0, "tab:red")):
    ax.plot(x, np.clip(curve[g], 1e2, None), color=c_, label=f"gamma {g}")
for c in (10_000, 50_000, 100_000, 250_000):
    ax.axhline(c, color="k", ls=":", lw=.8); ax.text(100, c, f" c = {c:,}", va="bottom", ha="right", fontsize=7)
ax.set(yscale="log", xlabel="share of applicants verified, ranked by suspicion (%)", ylabel="marginal break-even cost c* per verified applicant (INR)",
       title="Verify the next applicant iff cost < marginal c*  (Arm B, 10% concealers, v = 1)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("docs/fig8_marginal_cstar.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
s = T3[T3.gamma == 0.5].pivot(index="sum_assured", columns="strategy", values="cstar_inr")
s.index = [f"{v/1e5:,.0f} lakh" if v < 1e7 else f"{v/1e7:,.0f} crore" for v in s.index]; s.plot.bar(ax=ax[0], rot=0)
ax[0].set(yscale="log", ylabel="c* per verified applicant (INR)", xlabel="sum assured", title="Break-even exam cost by policy size (gamma 0.5, 10% concealers)"); ax[0].grid(alpha=.3, axis="y")
for st, s in T4[T4.gamma == 0.5].groupby("strategy"):
    ax[1].plot(s.prevalence * 100, s["cstar_v1.0"], marker="o", label=st)
ax[1].set(xscale="log", yscale="log", xlabel="share of applicants who conceal (%)", ylabel="c* (INR)", title="Break-even cost vs concealment prevalence (gamma 0.5)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("docs/fig8_sa_prevalence.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(9, 4.6)); s = T5[(T5.gamma == 0.5) & (T5.v == 1.0)]
pv = s.pivot(index="type", columns="operating_point", values="penalty_multiple_1_over_rv").loc[TYPES]; pv.plot.bar(ax=ax, rot=0)
ax.set(yscale="log", ylabel="required total loss on detection, as a multiple of the premium saved (1 / r v)", title="Deterrence: how large must the penalty be? (v = 1)"); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("docs/fig8_deterrence.png", dpi=110); plt.close()

# ---------------- console ----------------
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("== T1 headline c* (INR per verified applicant, 10% concealers, v=1) ==")
print(T1.pivot_table(index="strategy", columns="gamma", values="cstar_inr").round(0).to_string())
print("\n95% CI at gamma 0.5:\n", T1[T1.gamma == 0.5][["strategy", "share_verified_pct", "cstar_inr", "lo", "hi"]].round(0).to_string(index=False))
print("\n== T2 marginal c*: top-1% bin and cumulative average at 5/10/20% (Arm B) ==")
for g in GAMMAS:
    t = T2[T2.gamma == g]
    print(f"gamma {g}: marginal@1% {t.marginal_cstar_inr.iloc[0]:,.0f} | @5% {t.marginal_cstar_inr.iloc[4]:,.0f} | @10% {t.marginal_cstar_inr.iloc[9]:,.0f} | @20% {t.marginal_cstar_inr.iloc[19]:,.0f} | @50% {t.marginal_cstar_inr.iloc[49]:,.0f}")
print("\n== T2 optimal share verified at gamma 0.5 ==\n", T2opt[T2opt.gamma == 0.5].round(1).to_string(index=False))
print("\n== T3 c* by sum assured, gamma 0.5 ==\n", T3[T3.gamma == 0.5].pivot(index="sum_assured", columns="strategy", values="cstar_inr").round(0).to_string())
print("\n== T4 prevalence sweep, gamma 0.5 ==\n", T4[T4.gamma == 0.5].pivot(index="prevalence", columns="strategy", values="cstar_v1.0").round(0).to_string())
print("\n== T5 deterrence, gamma 0.5, v=1 ==\n", T5[(T5.gamma == 0.5) & (T5.v == 1.0)][["operating_point", "type", "mean_dpi_inr", "recall_r", "false_verif_rate_f", "penalty_multiple_1_over_rv", "P_star_inr"]].round(3).to_string(index=False))
pd.set_option("display.float_format", lambda x: f"{x:,.4f}"); print("\n== T6 illustrative equilibrium (Arm B top 10%, gamma 0.5) ==\n", T6.to_string(index=False))
