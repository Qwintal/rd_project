"""Concealment layer.

Rules (locked in the Phase 1 spec):
  * concealment = FALSE DECLARED VALUE, never missing/NaN
  * eligibility: smoking concealers must be smokers; diagnosis concealers must
    hold >=1 diagnosis; 'multiple' = smoking + ONE diagnosis (needs both)
  * assignment is random among eligible applicants (conservative: no adverse
    selection into concealment); insurer-measured biomarkers stay true
"""
import numpy as np

DX = ["diabetes", "hypertension", "cvd"]


def assign_concealment(df, cfg, rng):
    n = len(df)
    mix = cfg["concealment_mix"]
    n_multi = int(round(mix["multiple"] * n))
    n_smoke = int(round(mix["smoking"] * n))
    n_dx = int(round(mix["diagnosis"] * n))

    smoker = df["true_smoker"].to_numpy() == 1
    has_dx = (df[[f"true_{d}" for d in DX]].sum(axis=1) > 0).to_numpy()

    pools = {"multiple": int((smoker & has_dx).sum())}
    if pools["multiple"] < n_multi:
        raise ValueError(f"Infeasible: need {n_multi} smoker+diagnosis applicants, have {pools['multiple']}")
    multi = rng.choice(np.where(smoker & has_dx)[0], n_multi, replace=False)

    taken = np.zeros(n, bool)
    taken[multi] = True
    pool_s = np.where(smoker & ~taken)[0]
    pools["smoking"] = len(pool_s)
    if len(pool_s) < n_smoke:
        raise ValueError(f"Infeasible: need {n_smoke} smokers for smoking concealment, have {len(pool_s)}")
    smk = rng.choice(pool_s, n_smoke, replace=False)
    taken[smk] = True

    pool_d = np.where(has_dx & ~taken)[0]
    pools["diagnosis"] = len(pool_d)
    if len(pool_d) < n_dx:
        raise ValueError(f"Infeasible: need {n_dx} diagnosed applicants for diagnosis concealment, have {len(pool_d)}")
    dxc = rng.choice(pool_d, n_dx, replace=False)

    out = df.copy()
    out["declared_smoker"] = out["true_smoker"]
    out["declared_pack_years"] = out["true_pack_years"]
    for d in DX:
        out[f"declared_{d}"] = out[f"true_{d}"]
    ctype = np.array(["truthful"] * n, dtype=object)
    cfields = np.array(["none"] * n, dtype=object)

    def hide_smoking(idx):
        out.loc[out.index[idx], "declared_smoker"] = 0
        out.loc[out.index[idx], "declared_pack_years"] = 0.0

    def hide_one_dx(i):
        held = [d for d in DX if out.iloc[i][f"true_{d}"] == 1]
        pick = held[rng.integers(len(held))]
        out.iloc[i, out.columns.get_loc(f"declared_{pick}")] = 0
        return pick

    hide_smoking(smk)
    ctype[smk], cfields[smk] = "smoking", "smoker"
    hide_smoking(multi)
    for i in multi:
        cfields[i] = "smoker|" + hide_one_dx(i)
    ctype[multi] = "multiple"
    for i in dxc:
        cfields[i] = hide_one_dx(i)
    ctype[dxc] = "diagnosis"

    out["concealment_type"] = ctype
    out["concealed_fields"] = cfields
    return out, pools
