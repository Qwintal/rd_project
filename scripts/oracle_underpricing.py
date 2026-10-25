"""How much does concealment hide? An oracle that knows the TRUE hazard function but sees only
DECLARED flags. python -m scripts.oracle_underpricing   (run from project root)
Gives the floor on pricing error attributable purely to the lie (no model-learning error)."""
import pandas as pd
from data_generation.generate import load_config, generate_base, apply_gamma
from data_generation import hazard as hz

cfg = load_config("config/default.yaml")
base, E, _ = generate_base(cfg)
for g in (0.0, 1.0):
    df, meta = apply_gamma(base, E, cfg, g)
    dd = df.copy()
    for c in ["smoker", "diabetes", "hypertension", "cvd"]:
        dd[f"true_{c}"] = dd[f"declared_{c}"]
    p = hz.prob_within(dd, cfg, meta["b"], meta["A0"], hz.linear_predictor(dd, cfg, g), 10)
    df["p_declared_oracle"] = p
    df["underprice_pct"] = 100 * (df.true_p10 - p) / df.true_p10
    print(f"\ngamma={g}")
    print(df.groupby("concealment_type").agg(n=("true_p10", "size"), mean_true=("true_p10", "mean"),
          mean_if_declared=("p_declared_oracle", "mean"), mean_underprice_pct=("underprice_pct", "mean")).round(4))
