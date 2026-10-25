# Verification module (Phase 6) - both arms get a trigger; compared at the SAME number verified

**Purity rule:** both triggers read only `feature_view(df, 'declared')` columns. Enforced by tests that scramble every hidden column and check scores do not change.
**Verification action:** a medical exam / records check reveals the true record (accuracy v = 1.0 in the base case; sensitivity 0.8-1.0 in the game).

| | Arm A trigger | Arm B trigger |
|---|---|---|
| Type | explicit rules | label-free statistical consistency model |
| Logic | flag if (declared no hypertension and SBP >= 140) or (declared no diabetes and HbA1c >= 6.5) | per field (smoker, hypertension, diabetes, CVD) a shallow gradient-boosted model predicts the DECLARED status from the other declared columns; suspicion = sum over fields declared 'no' of P(field = yes given the rest) |
| Training | none | train split, declared view only, no concealment labels |
| Smoking | no rule (no marker in the data) | weak signal via correlated columns |

**Value measure (evaluation only):** hidden loss = sum assured x (true_p10 - p10 the TRUE hazard assigns to the declared record). Arm-independent; zero for honest applicants.

## Results (validation split, mean of 3 seeds, gamma 0.5; nothing tuned on validation)
At the same number verified (8.6% of applicants):

| Trigger | Precision | Recall | False-verif. rate | Hidden loss caught | Recall: smoking / diagnosis / multiple |
|---|---|---|---|---|---|
| Rule (Arm A) | 0.857 | 0.211 | 0.019 | 25.3% | 1.9% / 36.4% / 34.0% |
| Consistency model (Arm B) | 0.859 | 0.211 | 0.019 | 27.9% | 3.0% / 34.3% / 35.3% |
| Random flagging | 0.350 | 0.086 | 0.086 | 8.6% | 8.6% each |

Consistency-model budget curve (share flagged -> precision / recall / hidden loss caught): 2.5% -> 0.95/0.07/9.8% ; 5% -> 0.92/0.13/18.9% ;
10% -> 0.83/0.24/30.3% ; 15% -> 0.76/0.32/38.0% ; 20% -> 0.70/0.40/45.2% ; 30% -> 0.61/0.52/56.6%.
AUC (concealed vs honest): consistency 0.752 (smoking 0.63, diagnosis 0.83, multiple 0.87); rule score 0.596 (coarse 0/1/2 score).

## Reading and caveats
- At a matched budget the two triggers are close: the statistical one catches ~2.6 points more hidden loss (25.3% vs 27.9%); significance is untested (3 seeds) - Phase 7.
- Its practical advantage is a continuous score, so the budget can be chosen. Smoking concealment is nearly undetectable for both (recall 2-3% at 8.6% flagged; 26% even at 30% flagged).
- Precision is inflated by the design prevalence (35% of cohort conceals; random precision 0.35). Re-weight to realistic prevalence before quoting it (game_prelim does this).
- The rule trigger has one operating point; budget-matched comparison uses top-k of the statistical score at the rule's flag count.
