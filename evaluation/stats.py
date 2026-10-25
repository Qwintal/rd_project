"""Statistics helpers: paired applicant bootstrap, Holm, DeLong, calibration metrics."""
import numpy as np
import statsmodels.api as sm
from scipy import stats


class Boot:
    """Bootstrap by APPLICANT. One resampling-count matrix C (B x n) is shared by every statistic."""
    def __init__(self, n, B=1000, seed=7):
        rng = np.random.default_rng(seed)
        self.n, self.B = n, B
        self.C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.float32)

    def means(self, M):
        return (self.C @ np.asarray(M, np.float32)) / self.n


def summarize(est, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5]); se = float(np.std(boot, ddof=1))
    p = float(2 * (1 - stats.norm.cdf(abs(est) / se))) if se > 0 else np.nan
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "se": se, "p": p}


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T
    return out


def delong(y, s1, s2):
    """Paired DeLong test for AUC(s1) - AUC(s2). Returns auc1, auc2, diff, p."""
    y = np.asarray(y); order = np.argsort(-y, kind="stable"); m = int(y.sum())
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]]); n = preds.shape[1] - m
    tx = np.array([_midrank(p[:m]) for p in preds]); ty = np.array([_midrank(p[m:]) for p in preds]); tz = np.array([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(aucs[0] - aucs[1]), float(2 * (1 - stats.norm.cdf(abs(z))))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calibration(y, p):
    lp = _logit(np.asarray(p)); y = np.asarray(y)
    slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    citl = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    o = np.argsort(p); g = np.array_split(o, 10)
    ece = sum(len(i) * abs(y[i].mean() - np.asarray(p)[i].mean()) for i in g) / len(y)
    chi = sum((y[i].sum() - len(i) * np.asarray(p)[i].mean()) ** 2 / (len(i) * np.asarray(p)[i].mean() * (1 - np.asarray(p)[i].mean())) for i in g)
    return {"cal_slope": float(slope), "cal_intercept": float(citl), "ece": float(ece), "hl_p": float(1 - stats.chi2.cdf(chi, 8))}
