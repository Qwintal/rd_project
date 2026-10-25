# ML vs Traditional Actuarial Underwriting - project repo (Phase 2 state)

Run from this folder (full grid takes ~2 min):
    pip install -r requirements.txt
    python -m scripts.sanity_check          # 1,000-applicant check + 100k deterministic check
    python -m scripts.oracle_underpricing   # how much concealment hides, by type

Layout: config/default.yaml (every DGP knob, each tagged V verified / A assumed / D design choice),
data_generation/ (hazard.py, covariates.py, concealment.py, generate.py), scripts/, docs/, data/.
Later phases add actuarial_engine/, ml_models/, verification/, experiments/, evaluation/, game_theory/, calculator/.

Phase 3 scripts (in order): `python -m scripts.make_datasets` (15 files into data/), `python -m scripts.validate_datasets` (docs/validation_report.md + 2 figures), `python -m scripts.footprint_probe`.
Docs: docs/data_dictionary.md, docs/amendments.md, docs/validation_report.md, docs/dataset_manifest.csv.

Phase 7-8 scripts: `scripts.phase7_predict`, `scripts.phase7_analyze` (frozen protocol in docs/phase7_protocol.md), `scripts.phase8_game`; docs/phase7_results.md, docs/game_theory.md.

Pipeline order: make_datasets -> validate_datasets -> train_arm_b (and --declared) -> phase7_predict -> phase7_analyze -> phase8_game ->streamlit run calculator/app.py