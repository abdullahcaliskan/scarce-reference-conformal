"""Stage 3 analysis: criteria P1/P2, budget curves, device-cluster bootstrap.
Usage: python -I analyze_stage3.py <stage3_dir>
"""
import sys, pathlib
import numpy as np, pandas as pd

R = pathlib.Path(sys.argv[1]); RNG = np.random.default_rng(0); NB = 2000
r = pd.read_csv(R / "recenter_runs.csv")
KEY = ["series", "device", "dataset", "site", "target", "origin"]
# average over phases -> one value per series x pollutant x window
u = r.groupby(KEY + ["method", "schedule", "budget"])[["coverage", "worst_month", "width", "winkler", "frac_inf"]].mean().reset_index()


def boot(df, col):
    v = df.dropna(subset=[col]).groupby("device")[col].apply(list)
    if len(v) < 2:
        return (np.nan, np.nan)
    devs = list(v.index); bs = []
    for _ in range(NB):
        pick = RNG.choice(len(devs), len(devs), replace=True)
        bs.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return tuple(np.round(np.percentile(bs, [2.5, 97.5]), 4))


def sel(m, s, b):
    return u[(u.method == m) & (u.schedule == s) & (u.budget == b)].set_index(KEY)


# 1) summary table
summ = u.groupby(["method", "schedule", "budget"]).agg(n=("coverage", "size"), coverage=("coverage", "mean"),
                                                      worst_month=("worst_month", "mean"), width=("width", "mean"),
                                                      winkler=("winkler", "mean")).round(3).reset_index()
summ["cov_ci"] = [boot(u[(u.method == x.method) & (u.schedule == x.schedule) & (u.budget == x.budget)], "coverage")
                  for x in summ.itertuples()]
summ.to_csv(R / "recenter_summary.csv", index=False)
by_ds = u[(u.schedule.isin(["none", "CONSEC"])) & (u.budget.isin([0, 3]))].groupby(
    ["dataset", "target", "method", "budget"])[["coverage", "worst_month", "width", "winkler"]].mean().round(3)
by_ds.to_csv(R / "recenter_by_dataset.csv")


# 2) paired comparisons
def paired(a, b, label):
    out = []
    for col in ["coverage", "worst_month", "width", "winkler"]:
        d = (sel(*b)[col] - sel(*a)[col]).dropna().reset_index(name="d")
        for scope, sub in [("ALL", d), ("QUANT", d[d.dataset == "QUANT"]), ("CAPTOR", d[d.dataset == "CAPTOR"]), ("QM", d[d.dataset == "QM"])]:
            if len(sub) == 0:
                continue
            out.append(dict(comparison=label, metric=col, scope=scope, n_units=len(sub), n_devices=sub.device.nunique(),
                            mean_diff=round(sub.d.mean(), 4), ci=boot(sub, "d"), frac_positive=round((sub.d > 0).mean(), 3)))
    return out


C = []
C += paired(("RW", "CONSEC", 3), ("RC-RW", "CONSEC", 3), "P1: RC-RW(3) - RW(3)")
C += paired(("RW", "CONSEC", 7), ("RC-RW", "CONSEC", 3), "P2: RC-RW(3) - RW(7)")
C += paired(("RW", "CONSEC", 30), ("RC-RW", "CONSEC", 3), "RC-RW(3) - RW(30)")
C += paired(("STATIC", "none", 0), ("RC-RW", "CONSEC", 3), "RC-RW(3) - STATIC")
C += paired(("RWT", "CONSEC", 3), ("RC-RWT", "CONSEC", 3), "RC-RWT(3) - RWT(3)")
C += paired(("RC-RW", "CONSEC", 3), ("RC-RW", "DIST", 3), "RC-RW: DIST(3) - CONSEC(3)")
C += paired(("RW", "CONSEC", 3), ("RW", "DIST", 3), "RW: DIST(3) - CONSEC(3)")
C += paired(("RC-RW", "CONSEC", 3), ("ORACLE", "none", 0), "ORACLE - RC-RW(3)")
C = pd.DataFrame(C); C.to_csv(R / "recenter_paired.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(summ.to_string())
print(C[C.scope == "ALL"].to_string())
print(C[(C.metric.isin(["coverage", "winkler"])) & (C.comparison.str.startswith("P1"))].to_string())
print(by_ds.to_string())
P1 = C[(C.comparison.str.startswith("P1")) & (C.scope == "ALL")].set_index("metric")
P2 = C[(C.comparison.str.startswith("P2")) & (C.scope == "ALL")].set_index("metric")
print("P1 met:", bool(P1.loc["coverage", "ci"][0] > 0 and P1.loc["winkler", "ci"][1] < 0))
print("P2 met:", bool(P2.loc["coverage", "ci"][0] > -0.01))
