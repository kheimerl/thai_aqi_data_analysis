"""Shared helpers for S1 foundation scripts (numpy/pandas only).

Time convention: PVT / peak-flow timestamps are naive Asia/Bangkok local time.
AQ `Timestamp` is Unix seconds. Conversion is explicit: unix = (local - epoch) - 7 h.
Never use datetime.timestamp() on naive datetimes.
"""
import os, re, zipfile, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FOUND = os.path.join(ROOT, "foundation")
BKK_OFFSET_S = 7 * 3600
EPOCH = pd.Timestamp("1970-01-01")


def local_to_unix(ts):
    """Naive Bangkok-local pandas datetimes -> Unix seconds (float), explicit -7 h."""
    ts = pd.to_datetime(ts)
    return (ts - EPOCH).dt.total_seconds() - BKK_OFFSET_S if isinstance(ts, pd.Series) \
        else (ts - EPOCH).total_seconds() - BKK_OFFSET_S


def unix_to_local(u):
    return EPOCH + pd.to_timedelta(np.asarray(u, dtype=float) + BKK_OFFSET_S, unit="s")


def read_mapping():
    m = pd.read_csv(os.path.join(DATA, "Rider-Sensors-mapping.csv"), skiprows=1)
    m["sensor"] = m["Sensor no."].astype(str).str.replace(",", "").str.strip().astype(int)
    m["user"] = m["PVT user"].astype(str).str.strip()
    m["city"] = m["City"].str.strip().map({"BKK": "BKK", "Chiangmai": "CMI"})
    return m


def read_xlsx(path):
    """Minimal xlsx reader: returns {sheet_name: DataFrame of raw cell values (str)}."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    out = {}
    with zipfile.ZipFile(path) as z:
        ss = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("a:si", ns):
                ss.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["a"])))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.get("Id"): r.get("Target") for r in rels}
        for sh in wb.find("a:sheets", ns):
            tgt = rmap[sh.get(rns)].lstrip("/")
            tgt = tgt if tgt.startswith("xl/") else "xl/" + tgt
            root = ET.fromstring(z.read(tgt))
            cells = {}
            for c in root.iter("{%s}c" % ns["a"]):
                ref = c.get("r"); col = re.match(r"[A-Z]+", ref).group(); row = int(ref[len(col):])
                v = c.find("a:v", ns); isv = c.find("a:is", ns)
                if c.get("t") == "s" and v is not None:
                    val = ss[int(v.text)]
                elif c.get("t") == "inlineStr" and isv is not None:
                    val = "".join(t.text or "" for t in isv.iter("{%s}t" % ns["a"]))
                else:
                    val = v.text if v is not None else None
                cells[(row, col)] = val
            if not cells:
                out[sh.get("name")] = pd.DataFrame(); continue
            rows = sorted({r for r, _ in cells}); 
            cols = sorted({c for _, c in cells}, key=lambda s: (len(s), s))
            out[sh.get("name")] = pd.DataFrame([[cells.get((r, c)) for c in cols] for r in rows],
                                               index=rows, columns=cols)
    return out


def excel_serial_to_date(x):
    try:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(x), unit="D")
    except (TypeError, ValueError):
        return pd.NaT


# ---------------- statistics (no scipy) ----------------
from math import lgamma, exp, log, sqrt, erf


def _betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c; c = c if abs(c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def t_sf2(t, df):
    """Two-sided p for Student t."""
    return betainc(df / 2, 0.5, df / (df + t * t))


def t_ppf975(df):
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_sf2(mid, df) > 0.05: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def norm_sf2(z):
    return 1 - erf(abs(z) / sqrt(2))


def ols(y, X):
    """OLS with intercept already in X. Returns beta, se (classical)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    r = y - X @ b
    s2 = (r @ r) / (n - k)
    return b, np.sqrt(np.diag(XtX_inv) * s2)


def pool_dl_hksj(b, se):
    """DerSimonian-Laird random-effects pooling + Hartung-Knapp-Sidik-Jonkman CI."""
    b = np.asarray(b, float); v = np.asarray(se, float) ** 2; k = len(b)
    w = 1 / v; mu_fe = (w * b).sum() / w.sum()
    Q = (w * (b - mu_fe) ** 2).sum()
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1 / (v + tau2); mu = (ws * b).sum() / ws.sum()
    se_dl = sqrt(1 / ws.sum())
    q = (ws * (b - mu) ** 2).sum() / (k - 1)
    se_hk = sqrt(q / ws.sum())
    tcrit = t_ppf975(k - 1)
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    return dict(k=k, beta=mu, se_dl=se_dl, p_dl=norm_sf2(mu / se_dl),
                ci_dl=(mu - 1.959964 * se_dl, mu + 1.959964 * se_dl),
                se_hksj=se_hk, p_hksj=t_sf2(mu / se_hk, k - 1) if se_hk > 0 else np.nan,
                ci_hksj=(mu - tcrit * se_hk, mu + tcrit * se_hk), tau2=tau2, I2=I2,
                n_neg=int((b < 0).sum()))
