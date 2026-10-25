# Phase 7 results - TEST split, run once under the frozen protocol (docs/phase7_protocol.md)

Pooled over 3 seeds = 45,000 test applicants per gamma. est = first model minus second (negative = first is better for error metrics); 95% paired bootstrap CI by applicant; p_holm = Holm-adjusted within family.
**Limitation that matters:** the applicant bootstrap treats the trained models as fixed, so CIs ignore model-fitting variance (only 3 seeds). `seeds_negative` (how many of 3 seeds agree in sign) is the guard; effects whose seeds disagree should be read as inconclusive.
The test split has now been used: any further tuning on it would invalidate these results.

## RQ1 - RMSE vs true p10, truthful view, truthful-trained models (F1; 19 of 20 Holm-significant)
|   gamma | label       |     est |      lo |      hi |   p_holm |   seeds_negative |
|--------:|:------------|--------:|--------:|--------:|---------:|-----------------:|
|    0    | xgb - arm_a |  0.0055 |  0.0052 |  0.0059 |   0      |                0 |
|    0    | glm - arm_a | -0.0006 | -0.0008 | -0.0004 |   0      |                1 |
|    0    | cox - arm_a | -0.0007 | -0.0009 | -0.0005 |   0      |                1 |
|    0    | xgb - glm   |  0.0061 |  0.0058 |  0.0064 |   0      |                0 |
|    0.25 | xgb - arm_a |  0.004  |  0.0037 |  0.0043 |   0      |                0 |
|    0.25 | glm - arm_a | -0.0006 | -0.0008 | -0.0004 |   0      |                1 |
|    0.25 | cox - arm_a | -0.0008 | -0.001  | -0.0006 |   0      |                1 |
|    0.25 | xgb - glm   |  0.0046 |  0.0044 |  0.0048 |   0      |                0 |
|    0.5  | xgb - arm_a |  0.0041 |  0.0037 |  0.0044 |   0      |                0 |
|    0.5  | glm - arm_a | -0.0012 | -0.0014 | -0.001  |   0      |                2 |
|    0.5  | cox - arm_a | -0.0016 | -0.0018 | -0.0013 |   0      |                2 |
|    0.5  | xgb - glm   |  0.0053 |  0.005  |  0.0055 |   0      |                0 |
|    0.75 | xgb - arm_a |  0.0026 |  0.0023 |  0.003  |   0      |                0 |
|    0.75 | glm - arm_a | -0.0023 | -0.0025 | -0.002  |   0      |                3 |
|    0.75 | cox - arm_a | -0.0028 | -0.003  | -0.0025 |   0      |                3 |
|    0.75 | xgb - glm   |  0.0049 |  0.0046 |  0.0052 |   0      |                0 |
|    1    | xgb - arm_a | -0.0002 | -0.0006 |  0.0002 |   0.4282 |                2 |
|    1    | glm - arm_a | -0.0036 | -0.0039 | -0.0032 |   0      |                3 |
|    1    | cox - arm_a | -0.0042 | -0.0045 | -0.0039 |   0      |                3 |
|    1    | xgb - glm   |  0.0034 |  0.0031 |  0.0037 |   0      |                0 |

## RQ2 - premium MAE (INR/year) vs true-p premium, truthful view (F2; 20 of 20 significant)
|   gamma | label       |       est |        lo |        hi |   p_holm |   seeds_negative |
|--------:|:------------|----------:|----------:|----------:|---------:|-----------------:|
|    0    | xgb - arm_a |  2679.03  |  2563.72  |  2787.14  |        0 |                0 |
|    0    | glm - arm_a |  -377.409 |  -452.437 |  -301.048 |        0 |                2 |
|    0    | cox - arm_a |  -536.746 |  -612.659 |  -457.196 |        0 |                2 |
|    0    | xgb - glm   |  3056.44  |  2955.64  |  3161.13  |        0 |                0 |
|    0.25 | xgb - arm_a |  1838.32  |  1735.64  |  1939.59  |        0 |                0 |
|    0.25 | glm - arm_a |  -477.23  |  -554.505 |  -400.168 |        0 |                2 |
|    0.25 | cox - arm_a |  -662.929 |  -741.39  |  -586.338 |        0 |                2 |
|    0.25 | xgb - glm   |  2315.55  |  2222.33  |  2407.76  |        0 |                0 |
|    0.5  | xgb - arm_a |  2013.57  |  1904.37  |  2121.8   |        0 |                0 |
|    0.5  | glm - arm_a |  -605.141 |  -689.012 |  -522.247 |        0 |                2 |
|    0.5  | cox - arm_a |  -850.338 |  -934.619 |  -767.796 |        0 |                2 |
|    0.5  | xgb - glm   |  2618.71  |  2515.87  |  2724.13  |        0 |                0 |
|    0.75 | xgb - arm_a |  1540.99  |  1424.7   |  1647.73  |        0 |                0 |
|    0.75 | glm - arm_a |  -874.621 |  -968.673 |  -785.993 |        0 |                2 |
|    0.75 | cox - arm_a | -1165.78  | -1258.21  | -1075.5   |        0 |                3 |
|    0.75 | xgb - glm   |  2415.61  |  2312.84  |  2523.64  |        0 |                0 |
|    1    | xgb - arm_a |   528.37  |   406.958 |   641.916 |        0 |                0 |
|    1    | glm - arm_a | -1036.7   | -1139.27  |  -938.433 |        0 |                3 |
|    1    | cox - arm_a | -1353.53  | -1456.53  | -1254.24  |        0 |                3 |
|    1    | xgb - glm   |  1565.07  |  1467.6   |  1669.27  |        0 |                0 |

## RQ3 - what concealment does to ML's advantage (F3a)
Difference-in-differences: positive = ML's advantage over Arm A GROWS under concealment, negative = shrinks/reverses. Underpricing: model minus Arm A, percentage points (negative = model underprices concealers less).
|   gamma | label                                               |     est |      lo |      hi |   p_holm |
|--------:|:----------------------------------------------------|--------:|--------:|--------:|---------:|
|    0    | DiD RMSE: (ArmA-xgb declared) - (ArmA-xgb truthful) |  0.0034 |  0.003  |  0.0037 |   0      |
|    0    | DiD RMSE: (ArmA-glm declared) - (ArmA-glm truthful) | -0.0026 | -0.0029 | -0.0023 |   0      |
|    0    | under% xgb - arm_a                                  | -0.9455 | -1.1831 | -0.7134 |   0      |
|    0    | under% glm - arm_a                                  |  5.1229 |  4.8941 |  5.3299 |   0      |
|    0.25 | DiD RMSE: (ArmA-xgb declared) - (ArmA-xgb truthful) |  0.0019 |  0.0016 |  0.0022 |   0      |
|    0.25 | DiD RMSE: (ArmA-glm declared) - (ArmA-glm truthful) | -0.0027 | -0.003  | -0.0024 |   0      |
|    0.25 | under% xgb - arm_a                                  | -0.3355 | -0.5642 | -0.0995 |   0.0075 |
|    0.25 | under% glm - arm_a                                  |  6.0351 |  5.8125 |  6.2446 |   0      |
|    0.5  | DiD RMSE: (ArmA-xgb declared) - (ArmA-xgb truthful) |  0.0017 |  0.0014 |  0.002  |   0      |
|    0.5  | DiD RMSE: (ArmA-glm declared) - (ArmA-glm truthful) | -0.0035 | -0.0038 | -0.0032 |   0      |
|    0.5  | under% xgb - arm_a                                  |  0.3896 |  0.1689 |  0.6197 |   0.0025 |
|    0.5  | under% glm - arm_a                                  |  7.2299 |  7.0124 |  7.4391 |   0      |
|    0.75 | DiD RMSE: (ArmA-xgb declared) - (ArmA-xgb truthful) |  0.0002 | -0.0001 |  0.0005 |   0.2116 |
|    0.75 | DiD RMSE: (ArmA-glm declared) - (ArmA-glm truthful) | -0.0045 | -0.0048 | -0.0042 |   0      |
|    0.75 | under% xgb - arm_a                                  |  0.2207 |  0.0101 |  0.4586 |   0.0599 |
|    0.75 | under% glm - arm_a                                  |  8.106  |  7.8905 |  8.3245 |   0      |
|    1    | DiD RMSE: (ArmA-xgb declared) - (ArmA-xgb truthful) | -0.0026 | -0.003  | -0.0022 |   0      |
|    1    | DiD RMSE: (ArmA-glm declared) - (ArmA-glm truthful) | -0.0059 | -0.0063 | -0.0056 |   0      |
|    1    | under% xgb - arm_a                                  |  2.7907 |  2.5815 |  3.0088 |   0      |
|    1    | under% glm - arm_a                                  |  9.4604 |  9.2357 |  9.677  |   0      |

## RQ3 - declared view, both training regimes (F3b; gamma 0 and 1 shown, full table in phase7_tests.csv)
|   gamma | label                    |     est |      lo |      hi |   p_holm |
|--------:|:-------------------------|--------:|--------:|--------:|---------:|
|       0 | glm[R2] - arm_a_plus[R2] | -0.0002 | -0.0004 | -0.0001 |   0.0994 |
|       0 | xgb[R2] - arm_a_plus[R2] |  0.0014 |  0.0011 |  0.0016 |   0      |
|       0 | xgb[R2] - glm[R2]        |  0.0016 |  0.0014 |  0.0018 |   0      |
|       0 | arm_a[R2] - arm_a[R1]    | -0.0032 | -0.0035 | -0.003  |   0      |
|       0 | glm[R2] - glm[R1]        | -0.0082 | -0.0085 | -0.0078 |   0      |
|       0 | cox[R2] - cox[R1]        | -0.0081 | -0.0085 | -0.0077 |   0      |
|       0 | xgb[R2] - xgb[R1]        | -0.0067 | -0.007  | -0.0063 |   0      |
|       0 | xgb[R1] - arm_a[R1]      |  0.0022 |  0.002  |  0.0024 |   0      |
|       0 | glm[R1] - arm_a[R1]      |  0.0021 |  0.0019 |  0.0023 |   0      |
|       0 | cox[R1] - arm_a[R1]      |  0.002  |  0.0018 |  0.0022 |   0      |
|       0 | xgb[R1] - glm[R1]        |  0.0001 | -0.0001 |  0.0003 |   0.6999 |
|       1 | glm[R2] - arm_a_plus[R2] | -0.0006 | -0.0008 | -0.0003 |   0.0001 |
|       1 | xgb[R2] - arm_a_plus[R2] |  0.0009 |  0.0005 |  0.0012 |   0      |
|       1 | xgb[R2] - glm[R2]        |  0.0014 |  0.0012 |  0.0017 |   0      |
|       1 | arm_a[R2] - arm_a[R1]    | -0.0036 | -0.0039 | -0.0034 |   0      |
|       1 | glm[R2] - glm[R1]        | -0.0089 | -0.0094 | -0.0085 |   0      |
|       1 | cox[R2] - cox[R1]        | -0.0089 | -0.0094 | -0.0084 |   0      |
|       1 | xgb[R2] - xgb[R1]        | -0.0076 | -0.0081 | -0.0071 |   0      |
|       1 | xgb[R1] - arm_a[R1]      |  0.0025 |  0.0022 |  0.0027 |   0      |
|       1 | glm[R1] - arm_a[R1]      |  0.0024 |  0.0022 |  0.0026 |   0      |
|       1 | cox[R1] - arm_a[R1]      |  0.0023 |  0.0021 |  0.0025 |   0      |
|       1 | xgb[R1] - glm[R1]        |  0.0001 | -0.0001 |  0.0003 |   1      |

## RQ3b - detection at the same number verified (F4)
|   gamma | label                                    |    est |      lo |     hi |   p_holm |
|--------:|:-----------------------------------------|-------:|--------:|-------:|---------:|
|    0    | hidden-loss caught %: consistency - rule | 2.7273 |  2.0291 | 3.4841 |        0 |
|    0    | recall %: consistency - rule             | 0.146  | -0.2428 | 0.5315 |        1 |
|    0.25 | hidden-loss caught %: consistency - rule | 2.7112 |  1.9819 | 3.5013 |        0 |
|    0.25 | recall %: consistency - rule             | 0.146  | -0.2428 | 0.5315 |        1 |
|    0.5  | hidden-loss caught %: consistency - rule | 2.6917 |  1.9272 | 3.5246 |        0 |
|    0.5  | recall %: consistency - rule             | 0.146  | -0.2428 | 0.5315 |        1 |
|    0.75 | hidden-loss caught %: consistency - rule | 2.6685 |  1.8918 | 3.5408 |        0 |
|    0.75 | recall %: consistency - rule             | 0.146  | -0.2428 | 0.5315 |        1 |
|    1    | hidden-loss caught %: consistency - rule | 2.6414 |  1.8276 | 3.5544 |        0 |
|    1    | recall %: consistency - rule             | 0.146  | -0.2428 | 0.5315 |        1 |
Recall is identical across gamma because the declared data (hence the triggers) do not depend on gamma.

## Secondary - AUC (DeLong) and outcome-based metrics
All AUC differences are non-significant after Holm (ceiling = AUC of the true probability: 0.714 at gamma 0, 0.731 at gamma 1).
|   gamma | model   |    auc |   brier |   log_loss |   cal_slope |   cal_intercept |    ece |   hl_p |
|--------:|:--------|-------:|--------:|-----------:|------------:|----------------:|-------:|-------:|
|       0 | arm_a   | 0.7097 |  0.0517 |     0.2029 |      0.9329 |          0.0298 | 0.0035 | 0.0038 |
|       0 | glm     | 0.7097 |  0.0517 |     0.2029 |      0.9468 |          0.0316 | 0.0026 | 0.0466 |
|       0 | cox     | 0.7099 |  0.0517 |     0.2029 |      0.9615 |          0.0315 | 0.0025 | 0.1559 |
|       0 | xgb     | 0.7065 |  0.0519 |     0.2036 |      0.9991 |          0.0359 | 0.0031 | 0.0793 |
|       0 | oracle  | 0.7136 |  0.0516 |     0.2022 |      0.9705 |          0.0412 | 0.0027 | 0.0266 |
|       1 | arm_a   | 0.7259 |  0.0508 |     0.1987 |      1.024  |          0.026  | 0.0034 | 0.0722 |
|       1 | glm     | 0.7261 |  0.0508 |     0.1987 |      0.949  |          0.0281 | 0.0022 | 0.0837 |
|       1 | cox     | 0.7263 |  0.0508 |     0.1986 |      0.9604 |          0.0268 | 0.0018 | 0.2437 |
|       1 | xgb     | 0.7267 |  0.0509 |     0.1989 |      1.0146 |          0.0291 | 0.0032 | 0.0591 |
|       1 | oracle  | 0.7305 |  0.0507 |     0.1979 |      0.9685 |          0.0346 | 0.0024 | 0.1734 |

## Descriptive tables (CSV in docs/)
phase7_portfolio.csv (loss ratio; correct = 0.80), phase7_expC.csv / phase7_expC_dec.csv (within-method effect, Arm A decision changes), phase7_expD.csv (subgroups), phase7_expE.csv (verification scenarios), phase7_bands.csv, phase7_rates.csv.

Figures: fig7_rmse_gamma.png, fig7_bias_by_group.png, fig7_expE.png.
