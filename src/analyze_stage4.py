"""Stage 4 analysis: pre-specified criteria K1/K2 (KC vs RC-RW), device-cluster bootstrap.
Usage: python -I analyze_stage4.py <stage4_dir>
"""
import sys, pathlib
import numpy as np, pandas as pd

R = pathlib.Path(sys.argv[1]); RNG = np.random.default_rng(0); NB = 2000
r = pd.read_csv(R / "kalman_runs.csv")
KEY = ["series", "device", "dataset", "site", "target", "origin"]
u = r.groupby(KEY + ["method", "schedule", "budget"])[["coverage", "worst_month", "width", "winkler"]].mean().reset_index()


def boot(df, col):
    v = df.dropna(subset=[col]).groupby("device")[col].apply(list)
    if len(v) < 2:
        return (np.nan, np.nan)
    bs = []
    for _ in range(NB):
        pick = RNG.choice(len(v), len(v), replace=True)
        bs.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return tuple(np.round(np.percentile(bs, [2.5, 97.5]), 4))


def sel(m, s, b):
    return u[(u.method == m) & (u.schedule == s) & (u.budget == b)].set_index(KEY)


summ = u.groupby(["method", "schedule", "budget"]).agg(n=("coverage", "size"), coverage=("coverage", "mean"),
                                                      worst_month=("worst_month", "mean"), width=("width", "mean"),
                                                      winkler=("winkler", "mean")).round(3).reset_index()
summ.to_csv(R / "kalman_summary.csv", index=False)
by_ds = u[u.budget == 3].groupby(["dataset", "target", "method", "schedule"])[["coverage", "worst_month", "width", "winkler"]].mean().round(3)
by_ds.to_csv(R / "kalman_by_dataset.csv")

C = []
for a, b, label in [(("RC-RW", "DIST", 3), ("KC", "DIST", 3), "K1: KC - RC-RW (DIST3)"),
                    (("RC-RW", "CONSEC", 3), ("KC", "CONSEC", 3), "K2: KC - RC-RW (CONSEC3)"),
                    (("RC-RW", "DIST", 3), ("KC-noGrowth", "DIST", 3), "Ablation: KC-noGrowth - RC-RW (DIST3)"),
                    (("KC-noGrowth", "DIST", 3), ("KC", "DIST", 3), "Ablation: KC - KC-noGrowth (DIST3)"),
                    (("RC-RW", "CONSEC", 30), ("KC", "DIST", 3), "KC(DIST3) - RC-RW(30)")]:
    for col in ["coverage", "worst_month", "width", "winkler"]:
        d = (sel(*b)[col] - sel(*a)[col]).dropna().reset_index(name="d")
        for scope, sub in [("ALL", d)] + [(x, d[d.dataset == x]) for x in ["QUANT", "CAPTOR", "QM"]]:
            if len(sub):
                C.append(dict(comparison=label, metric=col, scope=scope, n_units=len(sub), n_devices=sub.device.nunique(),
                              mean_diff=round(sub.d.mean(), 4), ci=boot(sub, "d"), frac_positive=round((sub.d > 0).mean(), 3)))
C = pd.DataFrame(C); C.to_csv(R / "kalman_paired.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
print(summ.to_string()); print(C[C.scope == "ALL"].to_string()); print(by_ds.to_string())
for k in ["K1", "K2"]:
    X = C[(C.comparison.str.startswith(k)) & (C.scope == "ALL")].set_index("metric")
    print(k, "met:", bool(X.loc["coverage", "ci"][0] > -0.01 and X.loc["worst_month", "ci"][0] > 0),
          "| coverage CI", X.loc["coverage", "ci"], "| worst month", X.loc["worst_month", "mean_diff"], X.loc["worst_month", "ci"])
