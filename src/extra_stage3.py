"""Stage 3 additional comparisons for the planned 'spread schedule' arm (not part of the criteria).
Usage: python -I extra_stage3.py <stage3_dir>
"""
import sys, pathlib
import numpy as np, pandas as pd
R = pathlib.Path(sys.argv[1]); RNG = np.random.default_rng(1); NB = 2000
r = pd.read_csv(R / "recenter_runs.csv")
KEY = ["series", "device", "dataset", "site", "target", "origin"]
u = r.groupby(KEY + ["method", "schedule", "budget"])[["coverage", "worst_month", "width", "winkler"]].mean().reset_index()
def boot(df, col):
    v = df.groupby("device")[col].apply(list); devs = list(v.index); bs = []
    for _ in range(NB):
        pick = RNG.choice(len(devs), len(devs), replace=True); bs.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return tuple(np.round(np.percentile(bs, [2.5, 97.5]), 4))
def sel(m, s, b): return u[(u.method == m) & (u.schedule == s) & (u.budget == b)].set_index(KEY)
rows = []
for lab, a, b in [("RC-RW DIST3 - RW CONSEC3", ("RW", "CONSEC", 3), ("RC-RW", "DIST", 3)),
                  ("RC-RW DIST3 - RW DIST3", ("RW", "DIST", 3), ("RC-RW", "DIST", 3)),
                  ("RC-RW DIST3 - RW CONSEC30", ("RW", "CONSEC", 30), ("RC-RW", "DIST", 3)),
                  ("RC-RW DIST3 - RC-RW CONSEC30", ("RC-RW", "CONSEC", 30), ("RC-RW", "DIST", 3))]:
    for col in ["coverage", "worst_month", "width", "winkler"]:
        d = (sel(*b)[col] - sel(*a)[col]).dropna().reset_index(name="d")
        for scope, sub in [("ALL", d), ("QUANT", d[d.dataset == "QUANT"]), ("CAPTOR", d[d.dataset == "CAPTOR"])]:
            rows.append(dict(comparison=lab, metric=col, scope=scope, n=len(sub), mean_diff=round(sub.d.mean(), 4),
                             ci=boot(sub, "d") if sub.device.nunique() > 1 else None, frac_pos=round((sub.d > 0).mean(), 3)))
out = pd.DataFrame(rows); out.to_csv(R / "recenter_paired_extra.csv", index=False)
pd.set_option("display.width", 250); print(out.to_string())
x = u[(u.method == "RC-RW") & (u.schedule == "DIST")]
print("RC-RW DIST3 by dataset/target:"); print(x.groupby(["dataset", "target"])[["coverage", "worst_month", "width", "winkler"]].mean().round(3).to_string())
