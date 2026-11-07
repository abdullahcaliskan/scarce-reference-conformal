"""Figures 1-5 and the graphical abstract, drawn from the frozen result CSVs.
Usage: python -I make_figures.py <results_root> <paper_dir>
Fonts: PDF core Helvetica (IOP accepts Times/Helvetica/Courier/Symbol in vector figures).
"""
import sys, pathlib
import numpy as np, pandas as pd, matplotlib


def rd(path):
    """Read a CSV that may be stored gzip-compressed (name.csv.gz)."""
    path = pathlib.Path(path)
    return pd.read_csv(path if path.exists() else path.with_name(path.name + ".gz"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from scipy.stats import norm
from scipy.optimize import brentq

R = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
sys.argv = [sys.argv[0], str(R), str(R / "stage4")]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import stage4_kalman as s4  # noqa: E402  (frozen Kalman-Conformal implementation)

from matplotlib import font_manager
for fp in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf",
           "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf"]:
    if pathlib.Path(fp).is_file():
        font_manager.fontManager.addfont(fp)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "pdf.use14corefonts": True, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "TeX Gyre Heros"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9993",
    "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.labelcolor": INK, "text.color": INK, "savefig.dpi": 300,
    "axes.titleweight": "normal", "axes.titlelocation": "left", "lines.linewidth": 1.4})
CM = 1 / 2.54


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", color=INK)


def ygrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


s3 = rd(R / "stage3" / "recenter_summary.csv")
s4sum = rd(R / "stage4" / "kalman_summary.csv")


def g(df, m, s, b, col):
    return float(df[(df.method == m) & (df.schedule == s) & (df.budget == b)][col].iloc[0])


# ======================= Figure 1: study design =======================
def month_strip(ax, y, days, label, col, delay=True):
    for d in range(30):
        filled = d in days
        ax.add_patch(Rectangle((d, y), 0.86, 0.62, facecolor=col if filled else "#f1f0ed",
                               edgecolor="none"))
        if delay and filled:
            ax.add_patch(Rectangle((d + 1, y), 0.86, 0.62, facecolor="none", edgecolor=col, lw=0.8, ls=(0, (2, 1.2))))
    ax.text(-0.8, y + 0.31, label, ha="right", va="center", color=INK)


fig = plt.figure(figsize=(15.5 * CM, 6.2 * CM))
ax = fig.add_axes([0.0, 0.70, 1.0, 0.24]); ax.set_xlim(-60, 545); ax.set_ylim(0, 1); ax.axis("off")
for x0, w, lab, col in [(0, 150, "Train the calibration model\n150 days", "#d9d8d3"),
                        (150, 30, "Calib.\n30 d", "#bcbbb5"),
                        (180, 365, "Test: intervals issued every hour, up to 365 days", "#eeeeea")]:
    ax.add_patch(Rectangle((x0, 0.25), w - 2, 0.5, facecolor=col, edgecolor="none"))
    ax.text(x0 + w / 2, 0.5, lab, ha="center", va="center", fontsize=7.5, color=INK)
ax.text(-6, 0.5, "(a)", ha="right", va="center", fontsize=8.5)
ax.text(0, 0.98, "One test window (each device, each pollutant); test periods of successive windows do not overlap", fontsize=7.5, color=INK2, va="top")
ax2 = fig.add_axes([0.20, 0.0, 0.78, 0.62]); ax2.set_xlim(-0.2, 30.2); ax2.set_ylim(-1.2, 2.2); ax2.axis("off")
month_strip(ax2, 1.3, [0, 1, 2], "Consecutive: 3 days in a row", GREY)
month_strip(ax2, 0.3, [0, 10, 20], "Spread: 1 day every 10 days", BLUE)
for d in [0, 10, 20, 29]:
    ax2.text(d + 0.43, 0.1, f"{d + 1}", ha="center", va="top", fontsize=6.5, color=INK2)
ax2.text(15, 0.0 - 0.18, "Day of the 30-day cycle", ha="center", va="top", fontsize=7, color=INK2)
ax2.text(-14.5, 2.05, "(b)", ha="left", va="center", fontsize=8.5)
ax2.text(-12.5, 2.05, "Reference (co-location) days within each 30-day cycle; same budget", va="center", fontsize=7.5, color=INK2)
ax2.add_patch(Rectangle((-12.5, -0.62), 0.86, 0.38, facecolor=GREY, edgecolor="none", clip_on=False))
ax2.text(-11.4, -0.43, "reference day", va="center", fontsize=6.5, color=INK2)
ax2.add_patch(Rectangle((-12.5, -1.12), 0.86, 0.38, facecolor="none", edgecolor=GREY, lw=0.8, ls=(0, (2, 1.2)), clip_on=False))
ax2.text(-11.4, -0.93, "reference values arrive (24 h later)", va="center", fontsize=6.5, color=INK2)
save(fig, "figure1")

# ======================= Figure 2: static intervals fail, mostly through bias =======================
D = rd(R / "stage2" / "alarm_days.csv")
D["month"] = (D.day // 30).astype(int)
m = D[D.month < 12].groupby(["series", "target", "origin", "month"]).cov7_static.mean().reset_index()
agg = m.groupby("month").cov7_static.agg(["mean", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "size"])
agg.columns = ["mean", "q25", "q75", "n"]
E = rd(R / "stage2" / "decomposition_summary.csv")
order = [("QUANT", "PM25"), ("QUANT", "NO2"), ("QUANT", "O3"), ("CAPTOR", "NO2"), ("CAPTOR", "O3"), ("QM", "NO2"), ("QM", "PM25")]
fig, axes = plt.subplots(1, 2, figsize=(15.5 * CM, 6.0 * CM), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.62))
ax = axes[0]
x = agg.index.values + 1
ax.fill_between(x, agg.q25, agg.q75, color=GREY, alpha=0.18, lw=0)
ax.plot(x, agg["mean"], color=INK2, marker="o", ms=3.5)
ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.text(0.7, 0.905, "target 0.90", fontsize=7, color=INK2, ha="left", va="bottom")
ax.set_xlabel("Month after calibration"); ax.set_ylabel("Coverage of the static interval\n(7-day windows)")
ax.set_xticks(range(1, 13)); ax.set_ylim(0.45, 1.0); ygrid(ax)
panel(ax, "a", "Static intervals lose coverage")
ax.text(12.3, 0.47, "line: mean; band: middle 50% of test windows", fontsize=6.5, color=INK2, ha="right")
ax = axes[1]
labels = []
for i, (ds, t) in enumerate(order):
    r = E[(E.dataset == ds) & (E.target == t)].iloc[0]; y = len(order) - 1 - i
    ax.plot([r.cov_static, r.cov_oracle_bias], [y, y], color="#cfcec9", lw=1.2, zorder=1)
    ax.scatter(r.cov_static, y, s=22, color=GREY, zorder=2, marker="o")
    ax.scatter(r.cov_oracle_bias, y, s=24, color=BLUE, zorder=3, marker="s")
    tl = {"PM25": "PM2.5", "NO2": "NO2", "O3": "O3"}[t]
    labels.append((y, f"{ds} {tl} ({int(r.n)})"))
ax.set_yticks([l[0] for l in labels]); ax.set_yticklabels([l[1] for l in labels])
ax.axvline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.set_xlim(0.5, 1.0); ax.set_xlabel("Coverage over the test window"); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
panel(ax, "b", "Removing the monthly bias closes most of the gap")
ax.scatter([], [], s=22, color=GREY, label="static"); ax.scatter([], [], s=24, color=BLUE, marker="s", label="static, true monthly bias removed")
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, -0.42), frameon=False, ncol=2, handletextpad=0.3, columnspacing=1.0)
save(fig, "figure2")

# ======================= Figure 3: budget curves =======================
B = [1, 3, 7, 30]; xs = np.arange(len(B))
fig, axes = plt.subplots(1, 3, figsize=(15.5 * CM, 5.6 * CM), gridspec_kw=dict(wspace=0.42))
for ax, col, title, lab, lim in [(axes[0], "coverage", "Average coverage", "Coverage", (0.84, 0.91)),
                                 (axes[1], "worst_month", "Worst 30-day coverage", "Coverage", (0.6, 0.85)),
                                 (axes[2], "winkler", "Interval score (lower is better)", "Winkler score", (18, 28))]:
    rw = [g(s3, "RW", "CONSEC", b, col) for b in B]; rc = [g(s3, "RC-RW", "CONSEC", b, col) for b in B]
    ax.plot(xs, rw, color=GREY, ls=(0, (4, 2)), marker="o", ms=4, mfc="white", label="Widen only (RW), consecutive")
    ax.plot(xs, rc, color=INK2, marker="s", ms=3.8, label="Correct bias (RC-RW), consecutive")
    ax.scatter([1.0], [g(s3, "RW", "DIST", 3, col)], marker="D", s=26, facecolor="white", edgecolor=GREY, lw=1.0, zorder=4,
               label="Widen only (RW), spread")
    ax.scatter([1.0], [g(s3, "RC-RW", "DIST", 3, col)], marker="D", s=34, color=BLUE, zorder=5, label="Correct bias (RC-RW), spread")
    if col == "coverage":
        ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in B]); ax.set_xlabel("Reference days per 30 days")
    ax.set_ylim(*lim); ax.set_ylabel(lab); ygrid(ax)
    panel(ax, "abc"[list(axes).index(ax)], title)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.1), handletextpad=0.3, columnspacing=2.0)
save(fig, "figure3")

# ======================= Figure 4: Kalman-Conformal vs RC-RW =======================
fig, axes = plt.subplots(1, 3, figsize=(15.5 * CM, 5.4 * CM), gridspec_kw=dict(wspace=0.42))
meths = [("RC-RW", "Correct bias (RC-RW)", BLUE, "D", s3), ("KC", "Kalman-Conformal (KC)", ORANGE, "o", s4sum),
         ("KC-noGrowth", "KC without uncertainty growth", GREY, "^", s4sum)]
for ax, col, title, lab, lim in [(axes[0], "coverage", "Average coverage", "Coverage", (0.85, 0.91)),
                                 (axes[1], "worst_month", "Worst 30-day coverage", "Coverage", (0.62, 0.76)),
                                 (axes[2], "winkler", "Interval score (lower is better)", "Winkler score", (20, 25))]:
    for j, (mm, ml, c, mk, src) in enumerate(meths):
        ys = [g(src, mm, s, 3, col) for s in ["CONSEC", "DIST"]]
        ax.scatter(np.array([0, 1]) + (j - 1) * 0.16, ys, color=c, marker=mk, s=28, zorder=3, label=ml,
                   edgecolor="white", linewidth=0.5)
    if col == "coverage":
        ax.axhline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Consecutive", "Spread"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(*lim); ax.set_ylabel(lab); ygrid(ax); panel(ax, "abc"[list(axes).index(ax)], title)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.5)
save(fig, "figure5")

# ======================= Figure 5: theory =======================
kr = rd(R / "stage4" / "kalman_runs.csv").drop_duplicates(["series", "target", "origin"])
rR, rQ = float((kr.s_day / kr.s_h).median()), float((kr.s_w_day / kr.s_h).median())
Rv, Qv = rR ** 2, rQ ** 2
n = 24 * 360; ts = np.arange(n, dtype=np.int64); dd = ts // 24
paths, theory = {}, []
for nm, mask, gaps in [("3 days, consecutive", (dd % 30) < 3, [1, 1, 28]), ("7 days, consecutive", (dd % 30) < 7, [1] * 6 + [24]),
                       ("3 days, spread", (dd % 10) == 0, [10, 10, 10])]:
    _, _, si = s4.kalman_conformal(ts, np.zeros(n), np.zeros(n), mask, np.ones(50), 1.0, Rv, Qv / 24, Rv + Qv, True)
    v = si[24 * 60:] ** 2; P = v - 1.0 - Rv; sd = np.sqrt(v)
    rho = brentq(lambda r: np.mean(2 * norm.cdf(r / sd) - 1) - 0.9, 0.1, 10)
    cov = 2 * norm.cdf(rho / sd) - 1
    abar = sum(x * x for x in gaps) / (2 * 30); amax = max(gaps)
    theory.append(dict(name=nm, abar=abar, amax=amax, Pm=P.mean(), Px=P.max(), vm=v.mean(), vx=v.max(), rho=rho, covmin=cov.min()))
    paths[nm] = (P[:24 * 60], cov[:24 * 60])
fig, axes = plt.subplots(1, 2, figsize=(15.5 * CM, 5.6 * CM), gridspec_kw=dict(wspace=0.32))
tt = np.arange(24 * 60) / 24
sty = {"3 days, consecutive": (GREY, (0, (4, 2))), "7 days, consecutive": (INK2, (0, (1, 1.2))), "3 days, spread": (BLUE, "-")}
for nm, (P, cov) in paths.items():
    c, ls = sty[nm]
    axes[0].plot(tt, P, color=c, ls=ls, lw=1.3, label=nm); axes[1].plot(tt, cov, color=c, ls=ls, lw=1.3, label=nm)
axes[0].set_ylabel("Bias variance $P_t$ / $\\sigma_h^2$".replace("$P_t$", "P(t)").replace("$\\sigma_h^2$", "hourly noise variance"))
axes[0].set_ylabel("Bias variance P(t)\n(units of hourly noise variance)")
axes[1].axhline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
axes[1].set_ylabel("Hourly coverage (model)")
for ax in axes:
    ax.set_xlabel("Days"); ax.set_xlim(0, 60); ax.set_xticks(range(0, 61, 10)); ygrid(ax)
panel(axes[0], "a", "Bias uncertainty between visits")
panel(axes[1], "b", "Coverage of a fixed-width interval")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), columnspacing=1.5)
save(fig, "figure4")
T = pd.DataFrame(theory); T.to_csv(OUT / "figures" / "theory_table.csv", index=False)
with open(OUT / "theory_numbers.tex", "w") as fh:
    fh.write("% Auto-generated by scripts/make_figures.py\n")
    key = {"3 days, consecutive": "ConsThree", "7 days, consecutive": "ConsSeven", "3 days, spread": "Spread"}
    for r in theory:
        k = key[r["name"]]
        for fld, d in [("abar", 1), ("amax", 0), ("Pm", 2), ("Px", 2), ("vm", 2), ("vx", 2), ("rho", 2), ("covmin", 3)]:
            fh.write(f"\\newcommand{{\\th{fld.capitalize()}{k}}}{{{r[fld]:.{d}f}}}\n")
    a = {r["name"]: r for r in theory}
    fh.write(f"\\newcommand{{\\thAgeRatio}}{{{a['3 days, consecutive']['abar'] / a['3 days, spread']['abar']:.1f}}}\n")
    fh.write(f"\\newcommand{{\\thVarRed}}{{{100 * (1 - a['3 days, spread']['vm'] / a['3 days, consecutive']['vm']):.0f}}}\n")
    fh.write(f"\\newcommand{{\\thPeakRed}}{{{100 * (1 - a['3 days, spread']['vx'] / a['3 days, consecutive']['vx']):.0f}}}\n")
    fh.write(f"\\newcommand{{\\thWidthInc}}{{{100 * (a['3 days, consecutive']['rho'] / a['3 days, spread']['rho'] - 1):.0f}}}\n")
print(T.round(3).to_string())

# ======================= Graphical abstract =======================
fig = plt.figure(figsize=(16 * CM, 6.4 * CM))
ax = fig.add_axes([0.02, 0.08, 0.52, 0.80]); ax.set_xlim(-11, 30.5); ax.set_ylim(-0.6, 3.4); ax.axis("off")
ax.text(-11, 3.15, "Same reference budget: 3 co-location days per month", fontsize=8.5, color=INK, va="center")
for y, days, lab, col in [(1.9, [0, 1, 2], "Consecutive", GREY), (0.6, [0, 10, 20], "Spread", BLUE)]:
    for d in range(30):
        ax.add_patch(Rectangle((d, y), 0.82, 0.7, facecolor=col if d in days else "#f1f0ed", edgecolor="none"))
    ax.text(-0.8, y + 0.35, lab, ha="right", va="center", fontsize=8, color=INK)
ax.text(15, 0.25, "one month", ha="center", va="top", fontsize=7, color=INK2)
ax.annotate("", xy=(0, 0.33), xytext=(30, 0.33), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.6))
ax.text(-11, -0.45, "Low-cost NO2, O3 and PM2.5 sensors; 3 datasets, 38 devices", fontsize=6.8, color=INK2)
ax = fig.add_axes([0.60, 0.20, 0.38, 0.66])
vals = [("No update (static)", g(s3, "STATIC", "none", 0, "coverage"), "#cfcec9"),
        ("Correct bias, consecutive days", g(s3, "RC-RW", "CONSEC", 3, "coverage"), GREY),
        ("Correct bias, spread days", g(s3, "RC-RW", "DIST", 3, "coverage"), BLUE)]
for i, (lab, v, c) in enumerate(vals):
    y = (len(vals) - 1 - i) * 1.0
    ax.barh(y, v - 0.6, left=0.6, height=0.42, color=c)
    ax.text(0.6, y + 0.27, lab, va="bottom", fontsize=7.5, color=INK)
    ax.text(v - 0.006, y, f"{v:.3f}", va="center", ha="right", fontsize=7.5,
            color="white" if c != "#cfcec9" else INK)
ax.set_yticks([]); ax.set_ylim(-0.4, 2.75)
ax.set_xlim(0.6, 0.95); ax.axvline(0.9, color=INK, lw=0.8, ls=(0, (4, 3)))
ax.text(0.903, -0.33, "target", fontsize=6.8, color=INK2, ha="left")
ax.set_xlabel("Coverage of 90% prediction intervals"); ax.spines["left"].set_visible(False)
save(fig, "graphical_abstract")
