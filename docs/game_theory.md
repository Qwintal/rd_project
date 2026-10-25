# Game theory (Phase 8): when is verification worth it, and what deters concealment?

All inputs come from the test-split predictions (no new fitting). Pairings: **Arm A** = Arm A pricing + rule trigger; **Arm B** = XGBoost pricing + consistency trigger. Regime 1 pricing, exam accuracy v = 1, **10% concealers** (re-weighted from the 35% design cohort) unless stated. Money in INR over the 10-year term, no discounting.
Closed forms and tests: `game_theory/model.py`, `tests/test_game.py` (the indifference conditions hold exactly).

## 1. The model in plain words
A **candidate** is an applicant with a concealable risk. They either disclose truthfully or conceal. The insurer either does nothing or runs a *flag-and-verify policy* (an exam for flagged applicants, cost c each).
- dpi = premium saved by lying; r = trigger recall (P flagged | concealer); f = false-flag rate (P flagged | honest); v = exam accuracy; delta = r v = detection probability.
- L = **total loss on detection** = forfeited saving + surcharge F (so F = L - dpi).

**Applicant.** If the insurer runs the policy with probability sigma, concealing beats disclosing by `dpi - sigma r v L`.
Truthful disclosure is a best response iff `sigma r v L >= dpi`. Even with sigma = 1 this needs **L >= P* = dpi / (r v)**, i.e. the penalty must be **1/(r v) times the premium saved** (scale-free). The extra surcharge is F* = P* - dpi.
**Insurer.** Running the policy beats not running it by `theta r v dpi - c [theta r + (1 - theta) f]`, where theta is the share of candidates who conceal. For one flagged applicant: verify iff **c < c* = P(conceals | flagged) v dpi**.
**Equilibrium (when L >= P* and v dpi > c):** both sides are indifferent at
- sigma* = dpi / (r v L) = P* / L   (insurer verifies flagged applicants with this probability),
- theta* = c f / ( r (v dpi - c) + c f )   (share of candidates who conceal).
Comparative statics: a bigger penalty lowers the verification intensity needed; a cheaper exam, higher recall or fewer false flags lowers equilibrium concealment. If L < P* the applicant conceals whatever the insurer does (pure equilibrium).
Note: the 10% prevalence in sections 2-4 is an exogenous assumption about the book; theta* in section 5 is endogenous. They answer different questions and are not mixed.

## 2. Insurer: break-even exam cost c* per verified applicant (gamma 0.5; 95% CI by applicant bootstrap)
| Strategy | Share of book verified | c* | 95% CI |
|---|---|---|---|
| Arm A blanket | 100.0% | ₹36,855 | ₹35,734 - ₹38,026 |
| Arm A rule | 3.7% | ₹1.6 lakh | ₹1.5 lakh - ₹1.7 lakh |
| Arm B blanket | 100.0% | ₹37,154 | ₹36,053 - ₹38,247 |
| Arm B matched to rule | 3.7% | ₹2.5 lakh | ₹2.3 lakh - ₹2.7 lakh |
| Arm B top 5% | 1.9% | ₹3.3 lakh | ₹3.0 lakh - ₹3.6 lakh |
| Arm B top 10% | 4.7% | ₹2.2 lakh | ₹2.0 lakh - ₹2.3 lakh |
| Arm B top 20% | 12.5% | ₹1.3 lakh | ₹1.2 lakh - ₹1.4 lakh |

**Policy statement.** At 10% concealers and v = 1, verifying **everyone** pays only if the exam costs less than about **₹37,154** per applicant. Targeted verification with the statistical trigger (top 10% of the design-cohort suspicion ranking, about 5% of a realistic book) pays while the cost per verified applicant is below **₹2.2 lakh** (95% CI ₹2.0 lakh - ₹2.3 lakh); the rule trigger (about 4% flagged) pays below **₹1.6 lakh**. At the same number verified the statistical trigger's threshold is ₹2.5 lakh.
"Top x%" refers to ranks in the 35%-concealer design cohort; at 10% prevalence the same score cut-off flags fewer applicants (the weighted share column).

**c* by gamma (RQ3-RQ4 link):**
| gamma | Arm A rule | Arm B top 10% (CI) | Arm B blanket |
|---|---|---|---|
| 0.0 | ₹1.6 lakh | ₹1.9 lakh (₹1.8 lakh - ₹2.1 lakh) | ₹35,145 |
| 0.25 | ₹1.6 lakh | ₹2.1 lakh (₹2.0 lakh - ₹2.2 lakh) | ₹36,241 |
| 0.5 | ₹1.6 lakh | ₹2.2 lakh (₹2.0 lakh - ₹2.3 lakh) | ₹37,154 |
| 0.75 | ₹1.6 lakh | ₹2.1 lakh (₹2.0 lakh - ₹2.3 lakh) | ₹36,906 |
| 1.0 | ₹1.6 lakh | ₹2.4 lakh (₹2.3 lakh - ₹2.6 lakh) | ₹39,870 |
c* rises with gamma for Arm B (XGBoost prices part of the interaction-amplified shortfall) and is flat for Arm A, whose loadings cannot see interactions.

## 3. Which applicant do I verify next? Marginal c* by suspicion rank (Arm B, 10% concealers)
Verify the next-ranked applicant iff the exam cost is below the marginal c* at that rank.
| gamma | top 1% | 5% | 10% | 20% | 50% |
|---|---|---|---|---|---|
| 0.0 | ₹4.5 lakh | ₹2.1 lakh | ₹91,931 | ₹51,442 | ₹32,503 |
| 0.5 | ₹5.0 lakh | ₹2.3 lakh | ₹1.0 lakh | ₹53,064 | ₹31,118 |
| 1.0 | ₹5.5 lakh | ₹2.6 lakh | ₹1.2 lakh | ₹55,954 | ₹31,633 |

Optimal share of a realistic (10%-concealer) book to verify, gamma 0.5:
| Exam cost c | Optimal share verified | Net recovered per 1,000 applicants | Blanket net per 1,000 |
|---|---|---|---|
| ₹2,000 | 91.8% | ₹3.52 crore | ₹3.52 crore |
| ₹5,000 | 90.5% | ₹3.25 crore | ₹3.22 crore |
| ₹10,000 | 87.9% | ₹2.81 crore | ₹2.72 crore |
| ₹25,000 | 46.0% | ₹1.75 crore | ₹1.22 crore |
| ₹50,000 | 17.0% | ₹99.6 lakh | -₹1.28 crore |
| ₹1.0 lakh | 5.4% | ₹57.1 lakh | -₹6.28 crore |
| ₹2.5 lakh | 1.4% | ₹15.8 lakh | -₹21.28 crore |
| ₹5.0 lakh | 0.3% | ₹3,903 | -₹46.28 crore |

## 4. Policy size and prevalence (gamma 0.5)
Break-even cost scales almost linearly with sum assured because the premium saved does.
| Sum assured | Blanket (Arm A pricing) | Arm A rule | Arm B top 10% |
|---|---|---|---|
| 10 lakh | ₹5,537 | ₹22,843 | ₹34,862 |
| 25 lakh | ₹14,432 | ₹62,054 | ₹88,686 |
| 50 lakh | ₹28,019 | ₹1.2 lakh | ₹1.7 lakh |
| 1 crore | ₹54,841 | ₹2.5 lakh | ₹3.5 lakh |
| 2 crore | ₹1.1 lakh | ₹4.6 lakh | ₹6.4 lakh |

| Concealer share of book | Arm A blanket | Arm A rule | Arm B top 10% |
|---|---|---|---|
| 2% | ₹7,371 | ₹52,123 | ₹68,038 |
| 5% | ₹18,428 | ₹1.0 lakh | ₹1.4 lakh |
| 10% | ₹36,855 | ₹1.6 lakh | ₹2.2 lakh |
| 20% | ₹73,711 | ₹2.1 lakh | ₹3.0 lakh |
| 35% | ₹1.3 lakh | ₹2.5 lakh | ₹3.6 lakh |
Exam accuracy scales c* linearly (v = 0.8 gives 80% of the values; v = 0.6 gives 60%) and the deterrence multiple by 1/v.

## 5. Applicant: how big must the penalty be? (gamma 0.5, v = 1)
| Trigger operating point | Concealment type | Recall r | False-flag rate f | Required penalty (multiple of premium saved) | P* |
|---|---|---|---|---|---|
| Arm A rule | diagnosis | 0.35 | 0.019 | 2.9x | ₹2.5 lakh |
| Arm A rule | multiple | 0.33 | 0.019 | 3.1x | ₹22.7 lakh |
| Arm A rule | smoking | 0.02 | 0.019 | 54.0x | ₹2.13 crore |
| Arm B matched to rule | diagnosis | 0.33 | 0.018 | 3.0x | ₹3.4 lakh |
| Arm B matched to rule | multiple | 0.35 | 0.018 | 2.9x | ₹21.9 lakh |
| Arm B matched to rule | smoking | 0.03 | 0.018 | 39.9x | ₹1.48 crore |
| Arm B top 10% | diagnosis | 0.38 | 0.026 | 2.6x | ₹3.0 lakh |
| Arm B top 10% | multiple | 0.40 | 0.026 | 2.5x | ₹19.3 lakh |
| Arm B top 10% | smoking | 0.04 | 0.026 | 27.0x | ₹99.8 lakh |
| Arm B top 20% | diagnosis | 0.58 | 0.095 | 1.7x | ₹2.0 lakh |
| Arm B top 20% | multiple | 0.62 | 0.095 | 1.6x | ₹12.3 lakh |
| Arm B top 20% | smoking | 0.13 | 0.095 | 7.7x | ₹28.3 lakh |

**Policy statement.** To make truthful disclosure a best response, the total loss on detection must be at least about **2.5-3x the premium saved for hidden diagnoses and multi-field concealment**, but **27-54x for hidden smoking**, because the triggers catch only 2-4% of smoking concealers. Doubling the share verified (top 20%) cuts the smoking multiple to about 7.7x but raises the false-flag rate from 3% to 10%. Smoking concealment is not realistically deterrable without a better detector (for example a nicotine test).

**Illustrative equilibrium** (Arm B top 10%, gamma 0.5, v = 1; L = penalty multiple x premium saved):
| Type | Exam cost | Penalty | Equilibrium | sigma* (verify prob.) | theta* (conceal prob.) |
|---|---|---|---|---|---|
| diagnosis | ₹10,000 | 3x | mixed | 0.87 | 0.0066 |
| diagnosis | ₹10,000 | 10x | mixed | 0.26 | 0.0066 |
| diagnosis | ₹50,000 | 3x | mixed | 0.87 | 0.0514 |
| diagnosis | ₹50,000 | 10x | mixed | 0.26 | 0.0514 |
| multiple | ₹10,000 | 3x | mixed | 0.84 | 0.0009 |
| multiple | ₹10,000 | 10x | mixed | 0.25 | 0.0009 |
| multiple | ₹50,000 | 3x | mixed | 0.84 | 0.0046 |
| multiple | ₹50,000 | 10x | mixed | 0.25 | 0.0046 |
| smoking | ₹10,000 | 3x | pure: applicant conceals; insurer verifies |  | 1.0000 |
| smoking | ₹10,000 | 10x | pure: applicant conceals; insurer verifies |  | 1.0000 |
| smoking | ₹50,000 | 3x | pure: applicant conceals; insurer verifies |  | 1.0000 |
| smoking | ₹50,000 | 10x | pure: applicant conceals; insurer verifies |  | 1.0000 |

## 6. Limitations
- Premium saved comes from Regime 1 (truthful-trained) pricing; declared-trained pricing would pool risk and change dpi.
- v = 1 in the base case; no discounting; the surcharge F is a deterrent only (not counted as insurer revenue).
- The equilibrium illustration uses type-average dpi; real policies vary (see the sum-assured table).
- The 10% concealer share is an assumption (the project notes cite 7-15% non-disclosure; I did not verify that range).
- CIs condition on the trained models and triggers; model-fitting variance (3 seeds) is not included.
- This is a one-shot, complete-information audit game; no reputation, learning or repeated play.
