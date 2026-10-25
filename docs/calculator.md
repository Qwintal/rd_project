# Premium calculator (Phase 9)

Run from the project root:
    pip install -r requirements.txt        # exact versions used: requirements-lock.txt
    streamlit run calculator/app.py
The app needs only `calculator/artifacts/` (2.3 MB, included), `config/default.yaml` and the code. If the pickled consistency trigger fails to load
(scikit-learn version mismatch) install the locked versions, or rebuild: `python -m scripts.make_datasets; python -m scripts.train_arm_b 2026 2027 2028;
python -m scripts.train_arm_b --declared 2026 2027 2028; python -m scripts.phase7_predict; python -m calculator.build_artifacts`.

## What it shows
1. **Premium** - for a hand-entered applicant: Arm A, GLM, XGBoost (and Arm A+ under R2) next to the exact simulated ground truth, for the DECLARED record (what the insurer sees) and the TRUE record; Arm A decision and loading breakdown; XGBoost TreeSHAP contributions; an extrapolation panel comparing the exact term probability with the constant-hazard extrapolation.
2. **Concealment & verification** - premium saved by the lie, rule trigger, consistency-model suspicion rank, P(conceals | rank, book share) by Bayes re-weighting, the insurer's verify/do-not-verify rule, and the deterrence threshold dpi / (r v) for the concealment type.
3. **Portfolio dashboard** - pooled test split (3 seeds) re-weighted to any concealer share: RMSE, loss ratio, who is over/under-charged, net recovered premium by verification strategy, marginal break-even cost curve.
4. **Notes & limitations** - including the mandatory extrapolation disclosure.

## Verification done
- Unit tests (tests/test_calculator.py): extrapolation formula; exact-DGP function reproduces the dataset's `true_p10` to 1e-10; exported GLM coefficients and XGBoost JSON reproduce the trained models (<1e-6) in both regimes; portfolio aggregates reproduce the Phase 7 loss-ratio, Experiment E and Phase 8 marginal-curve tables.
- Headless app tests (tests/test_app.py, Streamlit AppTest): all 6 example applicants, both regimes x gamma {0, 1} x term {5, 30} run without exceptions; a concealed smoker is cheaper on the declared record under every model; warnings appear for ages outside 40-60 and for expiry age > 85.
- **Not done:** the visual layout was not inspected in a real browser (none available here); check it on your machine.

## Extrapolation finding (important for the demo)
Hazard rises with age, so the constant-hazard extrapolation understates longer terms. Example (concealed-smoker example, true record, gamma 0.5):
exact 20-year claim probability 17.8% vs 12.9% from a constant hazard fitted on the exact 10-year value (4.9 points low; the annual premium is understated by about 28%);
at 30 years 34.7% vs 18.7%. Beyond ~20 years applicants pass age 80, outside the ages the baseline hazard was fitted on, so even the "exact" line is an extrapolation of the simulation.
The 20-year demo quote therefore carries a material, disclosed model-assumption error.
