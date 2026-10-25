"""Arm A+ : Arm A with EXPERIENCE-RATED class scalars (Regime 2 only).
Same base table, same fixed loadings, same decision table; the single recalibration scalar k is replaced by one scalar per
declared class (declared smoker x any declared diagnosis, using the same measured-value override as Arm A), each set so that
mean predicted p10 equals the observed claim rate in that class on TRAIN (actual-over-expected by class, standard actuarial practice)."""
import numpy as np
from data_generation.generate import feature_view
from .arm_a import TraditionalEngine, TERM


class TraditionalEnginePlus(TraditionalEngine):
    def __init__(self):
        super().__init__()
        self.k_cls = {}

    def _cls(self, X):
        from .arm_a import SBP_HTN, HBA1C_DM
        dx = (X["hypertension"] == 1) | (X["diabetes"] == 1) | (X["cvd"] == 1) | (X["sbp"] >= SBP_HTN) | (X["hba1c"] >= HBA1C_DM)
        return (2 * (X["smoker"] == 1).astype(int) + dx.astype(int)).to_numpy()

    def _kvec(self, X):
        c = self._cls(X)
        return np.array([self.k_cls.get(int(i), self.k) for i in c])

    def p10(self, X, k=None):
        kv = self.k if k is not None else self._kvec(X)
        kv = k if k is not None else kv
        return 1.0 - np.exp(-TERM * kv * self._base_hazard(X) * self._multiplier(X))

    @staticmethod
    def _bisect(base, target):
        lo, hi = 1e-3, 50.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if (1 - np.exp(-TERM * mid * base)).mean() < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def fit(self, train_df, view="declared"):
        X = feature_view(train_df, view)
        base = self._base_hazard(X) * self._multiplier(X)
        y = train_df["claim_10yr"].to_numpy()
        self.k = self._bisect(base, y.mean())                      # fallback for empty classes
        cls = self._cls(X)
        self.k_cls = {int(c): self._bisect(base[cls == c], y[cls == c].mean()) for c in np.unique(cls) if (cls == c).sum() >= 50}
        return self
