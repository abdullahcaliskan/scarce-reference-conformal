"""Stage 3 - with the same reference budget, widen the interval or correct the bias of the prediction?

PRE-SPECIFIED TEST (written on 7 November 2026, before the results were seen)
  Data: output of prep_series.py; gradient boosting model; test windows start 365 days apart;
  150 days training, 30 days calibration, <= 365 days test.
  Reference values are revealed only on reference days and 24 hours late. Budget b in {1,3,7,30} days per 30 days.
  Schedules: b consecutive days (5 phases: 0,6,12,18,24); in addition, for b=3, 'spread' (1 day in 10, 5 phases: 0,2,4,6,8).
  Methods:
    STATIC     : split conformal from the calibration month, never updated
    RW         : widen only; quantile of the last 168 scores
    RWT        : widen only; scores of the last 90 days (time window, independent of the budget)
    RC-STATIC  : re-centre on the mean residual of the last 72 reference values; calibration radius kept
    RC-RW      : re-centre + RW on the re-centred scores (last 168)
    RC-RWT     : re-centre + last 90 days
    ORACLE     : diagnostic; STATIC re-centred on the true mean residual of each 30-day block
  Primary criterion P1 (b=3, consecutive): RC-RW - RW coverage difference > 0 and Winkler difference < 0, both with
    device-cluster bootstrap 95% CIs excluding zero.
  Secondary P2: coverage of RC-RW (b=3) not lower than RW (b=7) (lower CI of the difference > -0.01).
Usage: python -I stage3_recenter.py <series_dir> <out_dir> [n_processes]
"""
import sys, pathlib, json, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ALPHA, DELAY, B, K_RC, WIN_H = 0.10, 24, 168, 72, 90 * 24
TRAIN_D, CAL_D, TEST_MAX_D, TEST_MIN_D, STEP_D = 150, 30, 365, 180, 365
MIN_TRAIN, MIN_CAL, MIN_TEST = 1500, 300, 2000
PRIMARY = {"QM": {"NO2": "f_no2_low_mgsv2_raw", "PM25": "f_pm2_5_low_sps30_ug_m3"},
           "CAPTOR": {"NO2": "f_we_no2", "O3": "f_we_o3"},
           "QUANT": {"NO2": "f_NO2", "O3": "f_O3", "PM25": "f_PM25"}}


@njit
def _q(s, a):
    n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        return np.inf
    return np.sort(s)[k - 1]


@njit
def stream(ts, yh, y, mask, cal_s, cal_t, mode, recenter, q0):
    """mode 0: fixed radius (q0); 1: last B scores; 2: last 90 days. recenter: mean residual of the last K_RC reference values."""
    n = len(ts); m = len(cal_s)
    q = np.empty(n); bias_i = np.zeros(n)
    S = np.empty(m + n); T = np.empty(m + n, dtype=np.int64)
    S[:m] = cal_s; T[:m] = cal_t; cnt = m
    Rr = np.empty(n); rc = 0; bias = 0.0
    if mode == 0:
        qc = q0
    else:
        qc = _q(S[max(0, cnt - B):cnt], ALPHA)
    p = 0; start = 0; last_day = -1
    for t in range(n):
        changed = False
        while p < t and ts[p] + DELAY <= ts[t]:
            if mask[p] and not np.isnan(y[p]):
                S[cnt] = abs(y[p] - (yh[p] + bias_i[p])); T[cnt] = ts[p]; cnt += 1
                Rr[rc] = y[p] - yh[p]; rc += 1
                if recenter:
                    bias = np.mean(Rr[max(0, rc - K_RC):rc])
                changed = True
            p += 1
        day = ts[t] // 24
        if mode == 1 and changed:
            qc = _q(S[max(0, cnt - B):cnt], ALPHA)
        elif mode == 2 and (changed or day != last_day):
            while start < cnt and T[start] < ts[t] - WIN_H:
                start += 1
            lo = start if cnt - start >= 24 else max(0, cnt - 24)
            qc = _q(S[lo:cnt], ALPHA)
        last_day = day
        q[t] = qc; bias_i[t] = bias
    return q, bias_i


def metrics(ts, y, center, q):
    ok = ~np.isnan(y); e = np.abs(y[ok] - center[ok]); qq = q[ok]
    cov = e <= qq
    blk = ((ts[ok] - ts[0]) // 720).astype(np.int64)
    c = np.bincount(blk, weights=cov.astype(float)); n = np.bincount(blk)
    mon = c[n >= 100] / n[n >= 100]
    fin = np.isfinite(qq)
    return dict(coverage=cov.mean(), worst_month=mon.min() if len(mon) else np.nan, frac_inf=1 - fin.mean(),
                width=np.mean(2 * qq[fin]) if fin.any() else np.nan,
                winkler=np.mean(2 * qq[fin] + (2 / ALPHA) * np.maximum(e[fin] - qq[fin], 0)) if fin.any() else np.nan)


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
    return "__".join(p[:3]) if len(p) >= 3 else name


METHODS = [("RW", 1, False), ("RWT", 2, False), ("RC-STATIC", 0, True), ("RC-RW", 1, True), ("RC-RWT", 2, True)]


def run(args):
    path, dataset, site, target = args
    name = path.stem; df = add_derived(pd.read_parquet(path)); ycol = f"y_{target}"; prim = PRIMARY[dataset][target]
    if prim not in df:
        return []
    fall = [c for c in df if c.startswith("f_") or c.startswith("d_")]
    start, end = df.index[df[prim].notna()].min(), df.index[df[prim].notna()].max()
    rows, o, k = [], start, 0
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
        cal_t = ca.index.values.astype("datetime64[h]").astype(np.int64)
        kq = int(np.ceil((len(cal_s) + 1) * (1 - ALPHA))); q0 = np.partition(cal_s, kq - 1)[kq - 1]
        ts = te.index.values.astype("datetime64[h]").astype(np.int64); y = te[ycol].values.astype(float)
        dd = ((te.index - ca_end).total_seconds().values // 86400).astype(int)
        base = dict(series=name, device=device_of(name), dataset=dataset, site=site, target=target, origin=k)
        rows.append(dict(base, method="STATIC", schedule="none", budget=0, phase=-1, n_labels=0,
                         **metrics(ts, y, yh, np.full(len(ts), q0))))
        ok = ~np.isnan(y); blk = (ts - ts[0]) // 720; center = yh.copy()
        for b_ in np.unique(blk):
            sel = (blk == b_) & ok
            if sel.sum():
                center[blk == b_] = yh[blk == b_] + np.mean(y[sel] - yh[sel])
        rows.append(dict(base, method="ORACLE", schedule="none", budget=0, phase=-1, n_labels=0,
                         **metrics(ts, y, center, np.full(len(ts), q0))))
        scheds = []
        for b in [1, 3, 7, 30]:
            for ph in ([0, 6, 12, 18, 24] if b < 30 else [0]):
                scheds.append(("CONSEC", b, ph, (((dd - ph) % 30) < b) if b < 30 else np.ones(len(ts), bool)))
        for ph in [0, 2, 4, 6, 8]:
            scheds.append(("DIST", 3, ph, ((dd - ph) % 10) == 0))
        for sname, b, ph, mask in scheds:
            for mname, mode, rcen in METHODS:
                q, bi = stream(ts, yh, y, mask, cal_s, cal_t, mode, rcen, q0)
                rows.append(dict(base, method=mname, schedule=sname, budget=b, phase=ph,
                                 n_labels=int((mask & ok).sum()), **metrics(ts, y, yh + bi, q)))
        k += 1
        o += pd.Timedelta(days=STEP_D)
    return rows


if __name__ == "__main__":
    meta = [m for m in json.load(open(P / "series_meta.json")) if m["kept"]]
    tasks = [(P / "series" / f"{m['series']}.parquet", m["dataset"], m["site"], t) for m in meta for t in PRIMARY[m["dataset"]]]
    t0 = time.time(); rows = []
    with Pool(NPROC) as pool:
        for i, r in enumerate(pool.imap_unordered(run, tasks)):
            rows += r
            if i % 10 == 0:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "recenter_runs.csv", index=False)
    print("done", len(rows), round(time.time() - t0), "s", flush=True)
