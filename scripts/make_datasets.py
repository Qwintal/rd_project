"""Build the full dataset grid.  python -m scripts.make_datasets
3 seeds x 5 gamma values x 50,000 applicants -> data/cohort_seed{S}_gamma{G}.csv.gz + docs/dataset_manifest.csv
Within a seed the SAME applicants appear at every gamma (common random numbers) and the split is IDENTICAL."""
import os
import pandas as pd
from data_generation.generate import load_config, generate_base, apply_gamma, assign_split

SEEDS = [2026, 2027, 2028]
GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N = 50_000
os.makedirs("data", exist_ok=True)

rows = []
for seed in SEEDS:
    cfg = load_config("config/default.yaml", n_applicants=N, seed=seed)
    base, E, pools = generate_base(cfg)
    ref_split = None
    for g in GAMMAS:
        df, meta = apply_gamma(base, E, cfg, g)
        df = assign_split(df, seed)
        if ref_split is None:
            ref_split = df["split"].copy()
        else:
            assert (df["split"] == ref_split).all(), "split must be identical across gamma"
        assert df.applicant_id.is_unique and df.isna().sum().sum() == 0
        path = f"data/cohort_seed{seed}_gamma{g}.csv.gz"
        df.to_csv(path, index=False, compression="gzip")
        te = df[df.split == "test"]
        r = {"seed": seed, "gamma": g, "n": len(df), "claims": int(df.claim_10yr.sum()),
             "mean_true_p10": round(df.true_p10.mean(), 4), "A0": meta["A0"],
             "test_n": len(te), "test_claims": int(te.claim_10yr.sum()),
             "size_MB": round(os.path.getsize(path) / 1e6, 1)}
        for t in ["truthful", "smoking", "diagnosis", "multiple"]:
            r[f"test_claims_{t}"] = int(te[te.concealment_type == t].claim_10yr.sum())
            r[f"test_n_{t}"] = int((te.concealment_type == t).sum())
        rows.append(r)
        print(f"seed {seed} gamma {g}: claims {r['claims']}, test claims {r['test_claims']}, {r['size_MB']} MB", flush=True)
pd.DataFrame(rows).to_csv("docs/dataset_manifest.csv", index=False)
