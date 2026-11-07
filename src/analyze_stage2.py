"""Stage 2 analysis: pre-specified criteria S1/S2 and the error decomposition.
Usage: python -I analyze_stage2.py <stage2_dir>
"""
import sys, pathlib, json
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = pathlib.Path(sys.argv[1]); RNG = np.random.default_rng(0); NB = 2000
D = pd.read_csv(R / "alarm_days.csv"); E = pd.read_csv(R / "decomposition.csv")
SIGS = ["sib", "ks", "zdrift"]


def unit_stats(g, cov, clock):
    fail = (g[cov] < 0.85).astype(int).values
    out = dict(n_days=len(g), fail_rate=fail.mean())
    ok_cls = 5 <= fail.sum() <= len(fail) - 5
    out["auc_clock"] = roc_auc_score(fail, g[clock]) if ok_cls and g[clock].nunique() > 1 else np.nan
    rc, ry = rankdata(g[clock]), rankdata(g[cov])
    for s in SIGS:
        v = g[s]
        if v.notna().sum() < 30 or not ok_cls:
            out[f"auc_{s}"] = np.nan; out[f"pr_{s}"] = np.nan; continue
        m = v.notna().values
        out[f"auc_{s}"] = roc_auc_score(fail[m], v[m])
        rs = rankdata(v[m]); X = np.c_[np.ones(m.sum()), rc[m]]
        res_s = rs - X @ np.linalg.lstsq(X, rs, rcond=None)[0]
        res_y = ry[m] - X @ np.linalg.lstsq(X, ry[m], rcond=None)[0]
        out[f"pr_{s}"] = np.corrcoef(res_s, res_y)[0, 1]
    return out


def boot_mean(df, col):
    v = df.dropna(subset=[col]).groupby("device")[col].apply(list)
    if len(v) < 2:
        return (np.nan, np.nan), len(v)
    devs = list(v.index); bs = []
    for _ in range(NB):
        pick = RNG.choice(len(devs), len(devs), replace=True)
        bs.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return tuple(np.round(np.percentile(bs, [2.5, 97.5]), 3)), len(v)


res = {}
for setting, cov, clock in [("B_rw3", "cov7_rw", "clock_rw"), ("A_static", "cov7_static", "clock_static")]:
    U = []
    for key, g in D.groupby(["series", "device", "dataset", "site", "target", "origin"]):
        U.append(dict(zip(["series", "device", "dataset", "site", "target", "origin"], key), **unit_stats(g, cov, clock)))
    U = pd.DataFrame(U); U.to_csv(R / f"alarm_units_{setting}.csv", index=False)
    rows = []
    for sig in SIGS:
        for scope, sub in [("ALL", U), ("QUANT", U[U.dataset == "QUANT"]), ("CAPTOR", U[U.dataset == "CAPTOR"]),
                           ("QM", U[U.dataset == "QM"])]:
            sub = sub.dropna(subset=[f"auc_{sig}", "auc_clock"]).copy()
            if sub.empty:
                continue
            sub["d"] = sub[f"auc_{sig}"] - sub["auc_clock"]
            ci_d, nd = boot_mean(sub, "d"); ci_a, _ = boot_mean(sub, f"auc_{sig}"); ci_p, _ = boot_mean(sub, f"pr_{sig}")
            rows.append(dict(setting=setting, signal=sig, scope=scope, n_units=len(sub), n_devices=nd,
                             auc_signal=round(sub[f"auc_{sig}"].mean(), 3), auc_signal_ci=ci_a,
                             auc_clock=round(sub["auc_clock"].mean(), 3),
                             auc_diff=round(sub["d"].mean(), 3), auc_diff_ci=ci_d,
                             partial_rho=round(sub[f"pr_{sig}"].mean(), 3), partial_rho_ci=ci_p,
                             S1_pass=bool(ci_d[0] > 0 and sub[f"auc_{sig}"].mean() >= 0.60),
                             S2_pass=bool(ci_p[1] < 0)))
    res[setting] = pd.DataFrame(rows)
out = pd.concat(res.values()); out.to_csv(R / "alarm_summary.csv", index=False)

# error decomposition
E["shortfall"] = 0.9 - E.cov_static
E["closed_by_bias"] = np.where(E.shortfall > 0.02, (E.cov_oracle_bias - E.cov_static) / E.shortfall, np.nan)
dec = E.groupby(["dataset", "target"]).agg(n=("series", "size"), cov_static=("cov_static", "mean"),
                                           cov_oracle_bias=("cov_oracle_bias", "mean"), cov_rw3=("cov_rw3", "mean"),
                                           frac_closed=("closed_by_bias", "median"), r2=("r2", "median")).round(3)
dec.to_csv(R / "decomposition_summary.csv")
allc = E.dropna(subset=["closed_by_bias"]).copy(); allc["d"] = allc.closed_by_bias
ci, nd = boot_mean(allc, "d")
pd.set_option("display.width", 250)
print(out.to_string()); print(dec.to_string())
print("ALL frac shortfall closed by oracle bias: mean", round(allc.d.mean(), 3), ci, "median", round(allc.d.median(), 3), "devices", nd)
print("static ALL", round(E.cov_static.mean(), 3), "oracle", round(E.cov_oracle_bias.mean(), 3), "rw3", round(E.cov_rw3.mean(), 3))
