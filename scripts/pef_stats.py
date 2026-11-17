"""Shared statistics for the peak-flow (Paper 2) scripts. numpy/pandas only.

Two-stage design: per-person OLS (HC1, or clustered by workday when >= 20 clusters),
then DerSimonian-Laird pooling with Hartung-Knapp-Sidik-Jonkman intervals (IntHout floor).
The helper functions follow scripts/analysis_health_v3.py so results are comparable.
"""
import math
import numpy as np


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))

    def cf(a, b, x):
        tiny = 1e-300; qab = a + b; qap = a + 1; qam = a - 1
        c = 1.0; d = 1 - qab * x / qap
        d = tiny if abs(d) < tiny else d; d = 1 / d; h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d
            c = 1 + aa / c; c = tiny if abs(c) < tiny else c
            d = 1 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 + aa * d; d = tiny if abs(d) < tiny else d
            c = 1 + aa / c; c = tiny if abs(c) < tiny else c
            d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-12: break
        return h
    if x < (a + 1) / (a + b + 2): return front * cf(a, b, x) / a
    return 1 - front * cf(b, a, 1 - x) / b


def p_t(t, df):
    """Two-sided t p-value."""
    return betainc(df / 2, 0.5, df / (df + t * t))


def p_z(z):
    return math.erfc(abs(z) / math.sqrt(2))


def t_crit(df, q=0.975):
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if 1 - 0.5 * p_t(mid, df) < q: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def ols(y, X, groups=None, min_g=20):
    n, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    g = None if groups is None else np.asarray(groups)
    if g is not None and len(np.unique(g)) >= min_g:
        meat = np.zeros((p, p)); ug = np.unique(g)
        for k in ug:
            m = g == k; s = X[m].T @ u[m]; meat += np.outer(s, s)
        G = len(ug); c = G / (G - 1) * (n - 1) / max(n - p, 1)
    else:
        meat = X.T @ (X * (u ** 2)[:, None]); c = n / max(n - p, 1)
    return b, np.sqrt(np.diag(c * XtXi @ meat @ XtXi)), u


def pool(betas, ses):
    b = np.asarray(betas, float); s = np.asarray(ses, float); k = len(b)
    w = 1 / s ** 2; mu_f = (w * b).sum() / w.sum(); q = (w * (b - mu_f) ** 2).sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (s ** 2 + tau2); mu = (ws * b).sum() / ws.sum(); se_dl = math.sqrt(1 / ws.sum())
    var_hk = (ws * (b - mu) ** 2).sum() / ((k - 1) * ws.sum())
    se_hk = math.sqrt(max(var_hk, 1 / ws.sum()))  # IntHout floor
    tc = t_crit(k - 1); tc90 = t_crit(k - 1, 0.95)
    return dict(k=k, mu=mu, se_dl=se_dl, lo_dl=mu - 1.96 * se_dl, hi_dl=mu + 1.96 * se_dl, p_dl=p_z(mu / se_dl),
                se_hk=se_hk, lo_hk=mu - tc * se_hk, hi_hk=mu + tc * se_hk, p_hk=p_t(mu / se_hk, k - 1),
                lo90_hk=mu - tc90 * se_hk, hi90_hk=mu + tc90 * se_hk,
                tau2=tau2, i2=max(0.0, (q - (k - 1)) / q) * 100 if q > 0 else 0.0, share_neg=float((b < 0).mean()))


def bayes(betaz, sez, sd_mu=0.1, sd_tau=0.1):
    """Grid posterior used by v3 (kept only to reproduce v3's P(harm))."""
    b = np.asarray(betaz); s2 = np.asarray(sez) ** 2
    mus = np.linspace(-5 * sd_mu, 5 * sd_mu, 801); taus = np.linspace(0, 5 * sd_tau, 301)
    ll = np.zeros((len(taus), len(mus)))
    for i in range(len(b)):
        v = s2[i] + taus[:, None] ** 2
        ll += -0.5 * np.log(v) - 0.5 * (b[i] - mus[None, :]) ** 2 / v
    lp = ll - 0.5 * (mus[None, :] / sd_mu) ** 2 - 0.5 * (taus[:, None] / sd_tau) ** 2
    pr = np.exp(lp - lp.max()); pr /= pr.sum(); pm = pr.sum(0)
    return dict(p_harm=float(pm[mus < 0].sum()))


def rider_fit(d, y, x, controls, cluster=None, min_n=12, min_g=20):
    """Per-person OLS of y on x + controls. Returns None if too few rows or rank-deficient."""
    cols = [y, x] + list(controls)
    d = d.dropna(subset=cols)
    if len(d) < min_n or d[x].std() < 1e-9: return None
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in [x] + list(controls)])
    if np.linalg.matrix_rank(X) < X.shape[1]: return None
    yy = d[y].to_numpy(float)
    b, se, _ = ols(yy, X, d[cluster].to_numpy() if cluster else None, min_g=min_g)
    return dict(beta=b[1], se=se[1], n=len(d), sd_y=yy.std(ddof=1))
