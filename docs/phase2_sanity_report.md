## Part A - 1000 applicants (seed 2026)
Gompertz slope b (fitted to CDC 2023 male q_x): 0.0633 per year
Realized smoker share: 35.4%
Concealment mix realized: truthful 65.0%, smoking 15.0%, diagnosis 12.0%, multiple 8.0%
Implied share of smokers who conceal smoking: 65%  (design consequence of oversampling)
Eligibility pool sizes (multi / smoking / diagnosis): {'multiple': 112, 'smoking': 274, 'diagnosis': 192}
Prevalence: HTN 21.9%, diabetes 7.4%, CVD 3.4%, BMI>=30 17.5%

### gamma = 0.0   (A0 = 1.169e-03)
mean true p10 = 0.0550 (x-conditional target); mean p10 given hidden frailty = 0.0548; claims = 50 of 1000  (~50 per 1,000)
            n  claims    rate  mean_true_p10
age_band                                    
40-49     494      15  0.0304         0.0353
50-60     506      35  0.0692         0.0742
                    n  claims  mean_true_p10  expected_test_claims(30%)
concealment_type                                                       
diagnosis         120       6         0.0530                        1.8
multiple           80      12         0.1434                        3.6
smoking           150      11         0.0822                        3.3
truthful          650      21         0.0382                        6.3
     mean_true_p10  claims
sex                       
F           0.0358      21
M           0.0736      29

### gamma = 1.0   (A0 = 1.080e-03)
mean true p10 = 0.0550 (x-conditional target); mean p10 given hidden frailty = 0.0550; claims = 51 of 1000  (~51 per 1,000)
            n  claims    rate  mean_true_p10
age_band                                    
40-49     494      15  0.0304         0.0331
50-60     506      36  0.0711         0.0764
                    n  claims  mean_true_p10  expected_test_claims(30%)
concealment_type                                                       
diagnosis         120       6         0.0508                        1.8
multiple           80      13         0.1660                        3.9
smoking           150      12         0.0813                        3.6
truthful          650      20         0.0361                        6.0
     mean_true_p10  claims
sex                       
F           0.0355      21
M           0.0740      30

## Part B - 100,000 applicants, deterministic true_p10
Marginal ratio = mean p10 with factor / mean p10 without (confounded by correlated factors, so can differ from the input HR).
Input HRs: smoker 2.4, hypertension 1.30, BMI>=30 1.26, diabetes 1.80
gamma                    0.0       1.0
mean_p10               0.055     0.055
ratio_smoker           2.622     2.892
ratio_hypertension     1.660     1.921
ratio_bmi>=30          1.467     1.712
ratio_diabetes         2.111     2.179
n_SHB               1938.000  1938.000
ratio_SHB_vs_none      5.448     8.531
product_of_singles     5.613     5.707

Interaction gap for smoker+HTN+BMI>=30: ratio(gamma=1)/ratio(gamma=0) = 1.57   (design value exp(0.50)=1.65 before age/other effects)

CDC-implied general-population 10-yr death probability for this age/sex mix (log-linear fit to 4 table points x assumed female ratio): 0.062
DGP target mean p10: 0.055  (cohort has 35% smokers, so above general population is expected)

## Part C - oracle ceiling with and without hidden frailty (100,000 applicants, gamma = 0.5)
Oracle = the TRUE x-conditional probability. It is the best any model can do from the form's columns,
so its outcome-based scores are the CEILING; every arm's gap to this ceiling is the honest measure.
       claims  oracle_AUC  oracle_Brier  sd(p_given_frailty)/sd(true_p10)
theta                                                                    
0.00     5499      0.7316        0.0490                            1.0000
0.25     5491      0.7274        0.0492                            1.1857