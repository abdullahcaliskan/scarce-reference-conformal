"""Generate numbers.tex (LaTeX macros) and table files from the frozen result CSVs.
Every number quoted in the manuscript comes from here, so the text cannot drift from the results.
Usage: python -I make_numbers.py <results_root> <paper_dir>
"""
import sys, re, json, pathlib
import numpy as np, pandas as pd


def rd(path):
    """Read a CSV that may be stored gzip-compressed (name.csv.gz)."""
    path = pathlib.Path(path)
    return pd.read_csv(path if path.exists() else path.with_name(path.name + ".gz"))

R = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
S2, S3, S4 = R / "stage2", R / "stage3", R / "stage4"
M = {}


def ci(s):
    v = re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", str(s).replace("np.float64", ""))
    return float(v[0]), float(v[1])


def f(x, d=3):
    return f"{x:.{d}f}"


def sgn(x, d=3):
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{d}f}"


def cis(lo, hi, d=3):
    a = lambda v: ("$-$" if v < 0 else "") + f"{abs(v):.{d}f}"
    return f"[{a(lo)}, {a(hi)}]"


# ---------------- data ----------------
runs3 = rd(S3 / "recenter_runs.csv")
st = runs3[runs3.method == "STATIC"]
M["nWindows"] = str(len(st)); M["nDevices"] = str(st.device.nunique())
for ds in ["QUANT", "CAPTOR", "QM"]:
    sub = st[st.dataset == ds]
    M[f"nWin{ds}"] = str(len(sub)); M[f"nDev{ds}"] = str(sub.device.nunique())
meta = json.load(open(R / "series_meta.json"))
M["nSeriesKept"] = str(sum(m["kept"] for m in meta))

# ---------------- stage 3 summary ----------------
s3 = rd(S3 / "recenter_summary.csv")


def g(m, s, b, col):
    return float(s3[(s3.method == m) & (s3.schedule == s) & (s3.budget == b)][col].iloc[0])


def gci(m, s, b):
    return ci(s3[(s3.method == m) & (s3.schedule == s) & (s3.budget == b)].cov_ci.iloc[0])


tags = {("STATIC", "none", 0): "Static", ("ORACLE", "none", 0): "Oracle",
        ("RW", "CONSEC", 3): "RWconsec", ("RW", "DIST", 3): "RWspread", ("RW", "CONSEC", 7): "RWconsecSeven",
        ("RW", "CONSEC", 30): "RWcont", ("RC-RW", "CONSEC", 3): "RCconsec", ("RC-RW", "DIST", 3): "RCspread",
        ("RC-RW", "CONSEC", 7): "RCconsecSeven", ("RC-RW", "CONSEC", 1): "RCconsecOne",
        ("RC-RW", "CONSEC", 30): "RCcont", ("RC-STATIC", "CONSEC", 3): "RCstatic",
        ("RWT", "CONSEC", 3): "RWTconsec", ("RC-RWT", "CONSEC", 3): "RCTconsec", ("RC-RWT", "DIST", 3): "RCTspread"}
for k, t in tags.items():
    M[f"cov{t}"] = f(g(*k, "coverage")); M[f"wm{t}"] = f(g(*k, "worst_month"))
    M[f"wid{t}"] = f(g(*k, "width"), 1); M[f"wink{t}"] = f(g(*k, "winkler"), 1)
lo, hi = gci("STATIC", "none", 0); M["covStaticCI"] = cis(lo, hi)
lo, hi = gci("RC-RW", "DIST", 3); M["covRCspreadCI"] = cis(lo, hi)

# per dataset x target range for RC-RW spread
runs_u = runs3.groupby(["series", "device", "dataset", "target", "origin", "method", "schedule", "budget"])["coverage"].mean().reset_index()
dt = runs_u[(runs_u.method == "RC-RW") & (runs_u.schedule == "DIST") & (runs_u.budget == 3)].groupby(["dataset", "target"]).coverage.mean()
M["covRCspreadMin"] = f(dt.min()); M["covRCspreadMax"] = f(dt.max())
dts = runs_u[(runs_u.method == "STATIC")].groupby(["dataset", "target"]).coverage.mean()
M["covStaticMin"] = f(dts.min()); M["covStaticMax"] = f(dts.max())

# paired comparisons
P = pd.concat([rd(S3 / "recenter_paired.csv"), rd(S3 / "recenter_paired_extra.csv")])
P = P[P.scope == "ALL"]


def pr(label, metric, name, d=3):
    row = P[(P.comparison == label) & (P.metric == metric)]
    if row.empty:
        raise SystemExit(f"missing {label} {metric}")
    row = row.iloc[0]; lo, hi = ci(row.ci)
    M[name] = sgn(row.mean_diff, d); M[name + "CI"] = cis(lo, hi, d)
    M[name + "Pos"] = f"{100 * row.frac_positive:.0f}"


pr("P1: RC-RW(3) - RW(3)", "coverage", "dPoneCov", 3)
pr("P1: RC-RW(3) - RW(3)", "winkler", "dPoneWink", 2)
pr("P1: RC-RW(3) - RW(3)", "width", "dPoneWid", 2)
pr("P1: RC-RW(3) - RW(3)", "worst_month", "dPoneWM", 3)
pr("P2: RC-RW(3) - RW(7)", "coverage", "dPtwoCov", 3)
pr("RC-RW: DIST(3) - CONSEC(3)", "coverage", "dSpreadCov", 3)
pr("RC-RW: DIST(3) - CONSEC(3)", "winkler", "dSpreadWink", 2)
pr("RC-RW: DIST(3) - CONSEC(3)", "worst_month", "dSpreadWM", 3)
pr("RW: DIST(3) - CONSEC(3)", "coverage", "dSpreadRWCov", 3)
pr("RC-RW DIST3 - RW CONSEC3", "coverage", "dBestVsRWCov", 3)
pr("RC-RW DIST3 - RW CONSEC3", "winkler", "dBestVsRWWink", 2)
pr("RC-RW DIST3 - RW CONSEC30", "coverage", "dBestVsContCov", 3)
pr("RC-RW DIST3 - RW CONSEC30", "winkler", "dBestVsContWink", 2)
pr("RC-RW DIST3 - RW CONSEC30", "worst_month", "dBestVsContWM", 3)
pr("RC-RW DIST3 - RC-RW CONSEC30", "winkler", "dBestVsRCContWink", 2)
print(sorted(set(P.comparison)))

# ---------------- stage 2: decomposition and alarms ----------------
E = rd(S2 / "decomposition.csv")
E["shortfall"] = 0.9 - E.cov_static
E["closed"] = np.where(E.shortfall > 0.02, (E.cov_oracle_bias - E.cov_static) / E.shortfall, np.nan)
cl = E.dropna(subset=["closed"])
rng = np.random.default_rng(0); v = cl.groupby("device").closed.apply(list); bs = []
for _ in range(2000):
    pick = rng.choice(len(v), len(v), replace=True); bs.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
M["fracClosed"] = f"{100 * cl.closed.mean():.0f}"; lo, hi = np.percentile(bs, [2.5, 97.5])
M["fracClosedCI"] = f"[{100 * lo:.0f}, {100 * hi:.0f}]"; M["fracClosedMed"] = f"{100 * cl.closed.median():.0f}"
M["nShortfall"] = str(len(cl))
M["pctImproved"] = f"{100 * (E.cov_oracle_bias > E.cov_static).mean():.0f}"
M["covOracleDecomp"] = f(E.cov_oracle_bias.mean()); M["covStaticDecomp"] = f(E.cov_static.mean())

A = rd(S2 / "alarm_summary.csv"); A = A[(A.setting == "B_rw3") & (A.scope == "ALL")]
for sig, nm in [("sib", "Sib"), ("ks", "KS"), ("zdrift", "Z")]:
    row = A[A.signal == sig].iloc[0]; lo, hi = ci(row.auc_signal_ci)
    M[f"auc{nm}"] = f(row.auc_signal); M[f"auc{nm}CI"] = cis(lo, hi)
    M[f"aucClock{nm}"] = f(row.auc_clock); lo, hi = ci(row.partial_rho_ci)
    M[f"rho{nm}"] = sgn(row.partial_rho); M[f"rho{nm}CI"] = cis(lo, hi)
    M[f"nUnits{nm}"] = str(row.n_units); M[f"nDev{nm}"] = str(row.n_devices)

# ---------------- stage 4: Kalman ----------------
s4 = rd(S4 / "kalman_summary.csv")
for m, s, b, t in [("KC", "CONSEC", 3, "KCconsec"), ("KC", "DIST", 3, "KCspread"),
                   ("KC-noGrowth", "DIST", 3, "KCngSpread"), ("KC-noGrowth", "CONSEC", 3, "KCngConsec")]:
    row = s4[(s4.method == m) & (s4.schedule == s) & (s4.budget == b)].iloc[0]
    M[f"cov{t}"] = f(row.coverage); M[f"wm{t}"] = f(row.worst_month); M[f"wid{t}"] = f(row.width, 1); M[f"wink{t}"] = f(row.winkler, 1)
K = rd(S4 / "kalman_paired.csv"); K = K[K.scope == "ALL"]
for lab, nm in [("K1", "Kone"), ("K2", "Ktwo"), ("Ablation: KC-noGrowth - RC-RW", "Kng")]:
    for met, mn, d in [("coverage", "Cov", 3), ("worst_month", "WM", 3), ("width", "Wid", 2), ("winkler", "Wink", 2)]:
        row = K[(K.comparison.str.startswith(lab)) & (K.metric == met)].iloc[0]; lo, hi = ci(row.ci)
        M[f"d{nm}{mn}"] = sgn(row.mean_diff, d); M[f"d{nm}{mn}CI"] = cis(lo, hi, d)
kr = rd(S4 / "kalman_runs.csv").drop_duplicates(["series", "target", "origin"])
M["parRh"] = f((kr.s_day / kr.s_h).median(), 2); M["parQh"] = f((kr.s_w_day / kr.s_h).median(), 2)
M["parSh"] = f(kr.s_h.median(), 1); M["parQmin"] = f(kr.s_w_day.min(), 3); M["parQmax"] = f(kr.s_w_day.max(), 1)


# ---------------- additional summary statistics ----------------
M["fracClosedPooled"] = f"{100 * (cl.cov_oracle_bias - cl.cov_static).sum() / cl.shortfall.sum():.0f}"
M["nRatioAboveOne"] = str(int((cl.closed > 1).sum()))
r2 = E.groupby(["dataset", "target"]).r2.median()
M["rtwoMin"] = f(r2.min(), 2); M["rtwoMax"] = f(r2.max(), 2); M["rtwoQMNO"] = f(r2.loc[("QM", "NO2")], 2)
M["rtwoMedian"] = f(E.r2.median(), 2)
for sig, nm in [("sib", "Sib"), ("ks", "KS"), ("zdrift", "Z")]:
    row = A[A.signal == sig].iloc[0]; lo, hi = ci(row.auc_diff_ci)
    M[f"aucDiff{nm}"] = sgn(row.auc_diff); M[f"aucDiff{nm}CI"] = cis(lo, hi)
# coverage CIs (device-cluster bootstrap) for every row of Table 3
def boot_cov(sub):
    v = sub.groupby("device").coverage.apply(list); out = []
    rg = np.random.default_rng(0)
    for _ in range(2000):
        pick = rg.choice(len(v), len(v), replace=True); out.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return np.percentile(out, [2.5, 97.5])
runs4 = rd(S4 / "kalman_runs.csv")
u4 = runs4.groupby(["series", "device", "dataset", "target", "origin", "method", "schedule", "budget"])["coverage"].mean().reset_index()
for k, t in list(tags.items()):
    sub = runs_u[(runs_u.method == k[0]) & (runs_u.schedule == k[1]) & (runs_u.budget == k[2])]
    lo, hi = boot_cov(sub); M[f"cov{t}CIb"] = cis(lo, hi)
for m, sch, t in [("KC", "CONSEC", "KCconsec"), ("KC", "DIST", "KCspread")]:
    sub = u4[(u4.method == m) & (u4.schedule == sch) & (u4.budget == 3)]
    lo, hi = boot_cov(sub); M[f"cov{t}CIb"] = cis(lo, hi)
# spread gain by site (RC-RW)
a = runs_u[(runs_u.method == "RC-RW") & (runs_u.budget == 3)].pivot_table(index=["series", "target", "origin"], columns="schedule", values="coverage")
site = runs3.drop_duplicates("series").set_index("series").site
a["site"] = [site[i[0]] for i in a.index]; gs = (a.DIST - a.CONSEC).groupby(a.site).mean()
M["spreadSiteMin"] = sgn(gs.min()); M["spreadSiteMax"] = sgn(gs.max()); M["nSites"] = str(len(gs))
M["nDevManchester"] = str(runs3[runs3.site == "Manchester"].device.nunique())

# ---------------- site-level sensitivity analysis ----------------
site_of = runs3.drop_duplicates("series").set_index("series").site
def paired_units(df, a, b):
    A = df[(df.method == a[0]) & (df.schedule == a[1]) & (df.budget == a[2])].set_index(["series", "target", "origin"]).coverage
    Bv = df[(df.method == b[0]) & (df.schedule == b[1]) & (df.budget == b[2])].set_index(["series", "target", "origin"]).coverage
    d = (Bv - A).dropna().reset_index(name="d"); d["site"] = d.series.map(site_of); return d
def site_boot(d):
    v = d.groupby("site").d.apply(list); rg = np.random.default_rng(0); out = []
    for _ in range(2000):
        pick = rg.choice(len(v), len(v), replace=True); out.append(np.mean(np.concatenate([v.iloc[i] for i in pick])))
    return np.percentile(out, [2.5, 97.5])
def loso(d):
    return [d[d.site != st].d.mean() for st in d.site.unique()]
u4s = u4.copy()
for nm, src, a, b in [("Spread", runs_u, ("RC-RW", "CONSEC", 3), ("RC-RW", "DIST", 3)),
                      ("Pone", runs_u, ("RW", "CONSEC", 3), ("RC-RW", "CONSEC", 3)),
                      ("BestVsRW", runs_u, ("RW", "CONSEC", 3), ("RC-RW", "DIST", 3))]:
    d = paired_units(src, a, b); lo, hi = site_boot(d); M[f"site{nm}CI"] = cis(lo, hi)
    l = loso(d); M[f"loso{nm}Min"] = sgn(min(l)); M[f"loso{nm}Max"] = sgn(max(l))
# K1 (KC vs RC-RW spread) from the stage-4 runs
dk = paired_units(u4s, ("RC-RW", "DIST", 3), ("KC", "DIST", 3)); lo, hi = site_boot(dk); M["siteKoneCI"] = cis(lo, hi)
l = loso(dk); M["losoKoneMin"] = sgn(min(l)); M["losoKoneMax"] = sgn(max(l))

with open(OUT / "numbers.tex", "w") as fh:
    fh.write("% Auto-generated by scripts/make_numbers.py -- do not edit by hand\n")
    for k in sorted(M):
        assert re.fullmatch(r"[A-Za-z]+", k), k
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print(len(M), "macros")

# ---------------- Table 1: datasets ----------------
t1 = r"""\begin{tabular}{@{}llllll@{}}
\br
Dataset & City (sites) & Period used & Devices$^{\rm a}$ & Pollutants & Windows \\
\mr
QUANT \cite{diez2024quant,quantdata} & Manchester, London, York & Dec 2019--Oct 2022 & %s & NO$_2$, O$_3$, PM$_{2.5}$ & %s \\
CAPTOR \cite{captordata} & Barcelona (3) & Jun 2022--Apr 2024 & %s & NO$_2$, O$_3$ & %s \\
QM \cite{mercadovelazquez2026accuracy,qmdata} & London (1) & Apr 2025--Jun 2026 & %s & NO$_2$, PM$_{2.5}$ & %s \\
\mr
Total & & & %s & & %s \\
\br
\end{tabular}""" % (M["nDevQUANT"], M["nWinQUANT"], M["nDevCAPTOR"], M["nWinCAPTOR"], M["nDevQM"], M["nWinQM"],
                    M["nDevices"], M["nWindows"])
(OUT / "table1.tex").write_text(t1)
