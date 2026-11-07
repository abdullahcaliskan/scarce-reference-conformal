"""Stage 2 - reference-free reliability signals and error decomposition.

PRE-SPECIFIED TEST (written on 7 November 2026, before the results were seen)
  Event   : interval coverage below 0.85 in the 7 days starting on a test day (at least 72 reference hours).
  Setting B (primary): RW (last 168 scores) with 3 consecutive reference days per 30 days, phase 0.
            CLOCK = days since the last reference day.
  Setting A (secondary): static split conformal; CLOCK = days since calibration.
  Signals (computed at the start of day d from sensor data of days d-7..d-1; no reference data used):
      SIB    : |x - median(sibling x)| / training median, 7-day median (primary pollutant channel)
      KS     : two-sample KS statistic between the 7-day window and training, maximum over sensor channels
      ZDRIFT : mean |z| of the sensor channels standardised by training, 7-day mean
  Criterion S1: AUROC(SIB) - AUROC(CLOCK) > 0 with a device-cluster bootstrap 95% CI excluding zero,
                and mean AUROC(SIB) >= 0.60.
  Criterion S2: partial Spearman correlation between SIB and 7-day coverage, given CLOCK, < 0 (CI excludes zero).
  The same criteria are reported for KS and ZDRIFT (baselines). Test windows start 365 days apart.
ERROR DECOMPOSITION: static interval versus the same interval re-centred on the true 30-day mean residual (diagnostic only).
Usage: python -I stage2_signals.py <series_dir> <out_dir> [n_processes]
"""
import sys, pathlib, json, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ALPHA, DELAY, BUF = 0.10, 24, 168
TRAIN_D, CAL_D, TEST_MAX_D, TEST_MIN_D, STEP_D = 150, 30, 365, 180, 365
MIN_TRAIN, MIN_CAL, MIN_TEST, WIN, MINLAB = 1500, 300, 2000, 7, 72
PRIMARY = {"QM": {"NO2": "f_no2_low_mgsv2_raw", "PM25": "f_pm2_5_low_sps30_ug_m3"},
           "CAPTOR": {"NO2": "f_we_no2", "O3": "f_we_o3"},
           "QUANT": {"NO2": "f_NO2", "O3": "f_O3", "PM25": "f_PM25"}}
SIBS = [["AQM1", "AQM2", "AQM3"], ["AP1", "AP2", "AP3"], ["NS1", "NS2", "NS3"], ["Mod1", "Mod2", "Mod3"],
        ["SA1", "SA2", "SA3"], ["Poll1", "Poll2"], ["Prax1", "Prax2"], ["PA1", "PA3", "PA4", "PA6"],
        ["AQM388", "AQM390"], ["AQY872", "AQY873"], ["Ari063", "Ari078"], ["Zep188", "Zep344"]]


@njit
def rw_stream(ts, yh, y, mask, cal_s, B, alpha, delay):
    n = len(ts); q = np.empty(n); buf = np.empty(B); m = min(B, len(cal_s))
    for i in range(m):
        buf[i] = cal_s[len(cal_s) - m + i]
    cnt = m; head = m % B
    k = int(np.ceil((cnt + 1) * (1 - alpha))); qc = np.sort(buf[:cnt])[k - 1] if k <= cnt else np.inf
    p = 0
    for t in range(n):
        ch = False
        while p < t and ts[p] + delay <= ts[t]:
            if mask[p] and not np.isnan(y[p]):
                buf[head] = abs(y[p] - yh[p]); head = (head + 1) % B; cnt = min(cnt + 1, B); ch = True
            p += 1
        if ch:
            k = int(np.ceil((cnt + 1) * (1 - alpha))); qc = np.sort(buf[:cnt])[k - 1] if k <= cnt else np.inf
        q[t] = qc
    return q


def add_derived(df):
    if "f_T" in df and "f_RH" in df:
        T, RH = df.f_T, df.f_RH.clip(0, 100)
        df["d_abs_hum"] = 216.7 * (RH / 100 * 6.112 * np.exp(17.62 * T / (243.12 + T))) / (273.15 + T)
    h = df.index.hour
    df["d_hod_sin"], df["d_hod_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    df["d_weekend"] = (df.index.dayofweek >= 5).astype(float)
    return df


def device_of(name):
    p = name.split("__")
    return p[0] + "__" + p[1] + ("__" + p[2] if len(p) > 2 else "")


def sibling_frame(name, prim):
    p = name.split("__")
    if p[0] != "QUANT" or len(p) < 3 or p[2] != "Manchester":
        return None
    grp = next((g for g in SIBS if p[1] in g), None)
    if grp is None:
        return None
    cols = []
    for s in grp:
        if s == p[1]:
            continue
        files = sorted(P.glob(f"series/QUANT__{s}__Manchester*.parquet"))
        parts = [pd.read_parquet(f, columns=[prim])[prim] for f in files if prim in pd.read_parquet(f).columns]
        if parts:
            cols.append(pd.concat(parts).sort_index().rename(s))
    if not cols:
        return None
    return pd.concat(cols, axis=1).median(axis=1)


def ks_grid(train_vals, win_vals, grid):
    if len(win_vals) < 24 or len(train_vals) < 100:
        return np.nan
    f_tr = np.searchsorted(np.sort(train_vals), grid, side="right") / len(train_vals)
    f_w = np.searchsorted(np.sort(win_vals), grid, side="right") / len(win_vals)
    return float(np.max(np.abs(f_tr - f_w)))


def run(args):
    path, dataset, site, target = args
    name = path.stem; df = add_derived(pd.read_parquet(path)); ycol = f"y_{target}"; prim = PRIMARY[dataset][target]
    if prim not in df:
        return [], []
    sib_med = sibling_frame(name, prim)
    fall = [c for c in df if c.startswith("f_") or c.startswith("d_")]
    start, end = df.index[df[prim].notna()].min(), df.index[df[prim].notna()].max()
    days_rows, dec_rows, o, k = [], [], start, 0
    while o + pd.Timedelta(days=TRAIN_D + CAL_D + TEST_MIN_D) <= end:
        tr_end, ca_end = o + pd.Timedelta(days=TRAIN_D), o + pd.Timedelta(days=TRAIN_D + CAL_D)
        te_end = min(ca_end + pd.Timedelta(days=TEST_MAX_D), end + pd.Timedelta(hours=1))
        W = df[(df.index >= o) & (df.index < tr_end)]
        if W[prim].notna().mean() < 0.6:
            o += pd.Timedelta(days=STEP_D); continue
        feats = [c for c in fall if W[c].notna().mean() >= 0.6]
        d = df.dropna(subset=feats)
        tr = d[(d.index >= o) & (d.index < tr_end)].dropna(subset=[ycol])
        ca = d[(d.index >= tr_end) & (d.index < ca_end)].dropna(subset=[ycol])
        te = d[(d.index >= ca_end) & (d.index < te_end)]
        if len(tr) < MIN_TRAIN or len(ca) < MIN_CAL or te[ycol].notna().sum() < MIN_TEST:
            o += pd.Timedelta(days=STEP_D); continue
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                            l2_regularization=1.0, random_state=0).fit(tr[feats].values, tr[ycol].values)
        yh = mdl.predict(te[feats].values); cal_s = np.abs(ca[ycol].values - mdl.predict(ca[feats].values))
        kq = int(np.ceil((len(cal_s) + 1) * (1 - ALPHA))); q0 = np.partition(cal_s, kq - 1)[kq - 1]
        ts = te.index.values.astype("datetime64[h]").astype(np.int64); y = te[ycol].values.astype(float)
        dd = ((te.index - ca_end).total_seconds().values // 86400).astype(int)
        mask = (dd % 30) < 3
        q_rw = rw_stream(ts, yh, y, mask, cal_s, BUF, ALPHA, DELAY)
        ok = ~np.isnan(y); e = np.abs(y - yh)
        cov_st = np.where(ok, e <= q0, np.nan); cov_rw = np.where(ok, e <= q_rw, np.nan)
        # error decomposition
        blk = (ts - ts[0]) // 720; center = yh.copy()
        for b in np.unique(blk):
            sel = (blk == b) & ok
            if sel.sum():
                center[blk == b] = yh[blk == b] + np.mean(y[sel] - yh[sel])
        dec_rows.append(dict(series=name, device=device_of(name), dataset=dataset, site=site, target=target, origin=k,
                             cov_static=float(np.nanmean(cov_st)), cov_oracle_bias=float(np.mean(np.abs(y[ok] - center[ok]) <= q0)),
                             cov_rw3=float(np.nanmean(cov_rw)), bias=float(np.mean(yh[ok] - y[ok])),
                             r2=float(1 - np.sum((y[ok] - yh[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))))
        # daily signals (relative to the training distribution)
        fc = [c for c in feats if c.startswith("f_")]
        mu, sd = tr[fc].mean(), tr[fc].std().replace(0, 1)
        grids = {c: np.quantile(tr[c].values, np.linspace(0.005, 0.995, 200)) for c in fc}
        trv = {c: tr[c].values for c in fc}
        sib_ratio = None
        if sib_med is not None:
            diff_tr = (tr[prim] - sib_med.reindex(tr.index)).abs().dropna()
            if len(diff_tr) > 500:
                s0 = diff_tr.median() + 1e-9
                sib_ratio = ((te[prim] - sib_med.reindex(te.index)).abs() / s0)
        day_idx = pd.Series(dd, index=te.index)
        n_days = int(dd.max()) + 1
        cs = pd.DataFrame({"dd": dd, "st": cov_st, "rw": cov_rw})
        for day in range(WIN, n_days - WIN):
            fut = cs[(cs.dd >= day) & (cs.dd < day + WIN)]
            nl = fut.st.notna().sum()
            if nl < MINLAB:
                continue
            past = te[(dd >= day - WIN) & (dd < day)]
            if len(past) < 48:
                continue
            ks = max(ks_grid(trv[c], past[c].values, grids[c]) for c in fc)
            zd = float(((past[fc] - mu) / sd).abs().mean(axis=1).mean())
            sb = float(sib_ratio[(dd >= day - WIN) & (dd < day)].median()) if sib_ratio is not None else np.nan
            cyc = day % 30
            days_rows.append(dict(series=name, device=device_of(name), dataset=dataset, site=site, target=target, origin=k,
                                  day=day, clock_static=day, clock_rw=max(0, cyc - 2) if cyc >= 3 else 0,
                                  in_coloc=int(cyc < 3), ks=ks, zdrift=zd, sib=sb,
                                  cov7_static=float(fut.st.mean()), cov7_rw=float(fut.rw.mean()), n_lab=int(nl)))
        k += 1
        o += pd.Timedelta(days=STEP_D)
    return days_rows, dec_rows


if __name__ == "__main__":
    meta = [m for m in json.load(open(P / "series_meta.json")) if m["kept"]]
    tasks = [(P / "series" / f"{m['series']}.parquet", m["dataset"], m["site"], t) for m in meta for t in PRIMARY[m["dataset"]]]
    t0 = time.time(); D, E = [], []
    with Pool(NPROC) as pool:
        for i, (a, b) in enumerate(pool.imap_unordered(run, tasks)):
            D += a; E += b
            if i % 10 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(D).to_csv(OUT / "alarm_days.csv", index=False)
    pd.DataFrame(E).to_csv(OUT / "decomposition.csv", index=False)
    print("done", len(D), "day rows,", len(E), "windows,", round(time.time() - t0), "s", flush=True)
