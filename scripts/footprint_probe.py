"""Footprint detectability probe.  python -m scripts.footprint_probe

Question: given ONLY what the insurer sees (declared view + measured biomarkers), how well can a
flexible classifier separate applicants who concealed field F from honest applicants who also
declared 'no F'?  AUC ~0.5 => undetectable by construction; ~1.0 => trivial.

This probe is a diagnostic for the DGP's footprint settings (config: footprint:). It is NOT the
verification module (Phase 6), and it uses the true concealment labels, which an insurer would not have.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from data_generation.generate import load_config, generate_base, apply_gamma, feature_view

cfg = load_config("config/default.yaml", n_applicants=20_000)
base, E, _ = generate_base(cfg)
df, _ = apply_gamma(base, E, cfg, 0.0)          # footprint does not depend on gamma
X = feature_view(df, "declared")
X["sex"] = (X["sex"] == "M").astype(int)

probes = {   # field -> (declared column that defines the candidate set, single-biomarker comparison)
    "smoker":       ("declared_smoker",       "cholesterol"),
    "hypertension": ("declared_hypertension", "sbp"),
    "diabetes":     ("declared_diabetes",     "hba1c"),
    "cvd":          ("declared_cvd",          None),
}
rows = []
for field, (dcol, marker) in probes.items():
    cand = df[dcol] == 0                                   # applicants who declare "no <field>"
    y = df["concealed_fields"].str.contains(field).astype(int)[cand]
    Xc = X[cand]
    Xtr, Xte, ytr, yte = train_test_split(Xc, y, test_size=0.3, stratify=y, random_state=1)
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.08, random_state=1).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    auc_single = roc_auc_score(y, Xc[marker]) if marker else np.nan
    rows.append({"concealed_field": field, "candidates(declare no F)": int(cand.sum()),
                 "concealers": int(y.sum()), "share_of_candidates": y.mean(),
                 "AUC_all_declared_features": auc, "AUC_single_biomarker": auc_single})
tb = pd.DataFrame(rows).set_index("concealed_field")
print(tb.round(3).to_string())
with open("docs/footprint_probe.md", "w") as f:
    f.write("# Footprint detectability probe (N=20,000)\n\n" + tb.round(3).to_markdown() + "\n")
