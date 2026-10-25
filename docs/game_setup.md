# Game-theory setup (costly-state-verification / audit game) - PRELIMINARY numbers

**Players / strategies:** applicant: disclose truthfully or conceal; insurer: do not verify, or verify a flagged applicant at cost c. If a concealer is flagged and verified, the lie is detected.
**Definitions (INR, 10-year term, no discounting):**
- d_pi = term premium saved by the lie = 10 x [premium(true record) - premium(declared record)] under the arm's own pricing (Regime 1).
- r = trigger recall on concealers; v = exam accuracy (base 1.0); detection probability = r v.
- **Deterrence threshold:** P* = d_pi / (r v), the total loss on detection (forfeited saving plus surcharge) at which truthful disclosure is a best response. Best-response condition, not an equilibrium claim.
- **Insurer rule:** verify a flagged applicant iff c < c*, c* = expected recovered premium shortfall per flagged applicant = sum_flagged w v d_pi [concealer] / sum_flagged w, with w re-weighting to a target concealer share.
Pairings: Arm A = loadings + rule trigger; Arm B = XGBoost + consistency trigger at the same number verified.

## Preliminary results (validation, mean of 3 seeds, INR over the term)
| gamma | Arm | mean d_pi per concealer | c* at design prevalence (35%) | c* at 10% prevalence |
|---|---|---|---|---|
| 0 | A | 3,67,000 | 2,61,000 | 1,69,000 |
| 0 | B | 3,49,000 | 3,53,000 | 2,31,000 |
| 0.5 | A | 3,67,000 | 2,61,000 | 1,69,000 |
| 0.5 | B | 3,72,000 | 4,05,000 | 2,65,000 |
| 1.0 | A | 3,66,000 | 2,60,000 | 1,68,000 |
| 1.0 | B | 3,99,000 | 4,50,000 | 2,94,000 |

d_pi by type at gamma 0.5 (Arm A / Arm B): smoking 3.97 / 3.72 lakh; diagnosis 0.79 / 1.02 lakh; multiple 7.43 / 7.75 lakh.
P* at gamma 0.5, v = 1 (Arm A / Arm B): diagnosis 2.2 / 3.0 lakh; multiple 21.9 / 21.9 lakh; **smoking 2.1 crore / 1.2 crore** (recall only 2-3%: deterrence is not achievable with these triggers).

## Reading and caveats
- c* is far above any realistic exam cost because these are large policies (sum assured up to 2 crore). The real question is the MARGINAL one: how far down the suspicion ranking verification still pays, by sum-assured band. That is the Phase 8 analysis.
- c*(gamma) rises for Arm B (3.5 -> 4.5 lakh) and is flat for Arm A: XGBoost prices part of the interaction-amplified difference, Arm A cannot. This links RQ3 and RQ4, but it is one pairing, validation only.
- Inputs are assumptions: 10% realistic concealer share, v = 1, no discounting, Regime 1 pricing (Regime 2 pooling would change d_pi), no strategic repricing by the insurer.
