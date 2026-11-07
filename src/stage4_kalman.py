"""Stage 4 - Kalman-Conformal: bias tracking with a Kalman filter + normalised conformal interval
whose width grows with the time since the last reference value.

MODEL (day-level local-level model)
  Daily mean residuals z_d = b_d + u_d (day effect, variance R), b_d = b_{d-1} + w_d (variance Q per day),
  within-day noise variance s_h2. The reference values of one calendar day form one observation
  (their mean, observation variance R + s_h2/n). R and Q: maximum likelihood on the 30 daily means of the
  calibration month; s_h2: mean within-day variance. Interval scale s_t = sqrt(s_h2 + R + P_t), where P_t grows
  by Q/24 per hour without reference data.
  Interval: [ yhat_t + bhat_t +/- qhat * s_t ]; qhat is the finite-sample 90% conformal quantile of the last 168
  normalised scores |y - yhat - bhat| / s (values at issue time); initially the calibration-month scores.
  Start: bhat_0 = 0 (as RC-RW), P_0 = variance of the daily mean calibration residuals.
  Ablation KC-noGrowth: P does not grow between reference values.
  VERSION NOTE (changed BEFORE the full run): the first version treated every hourly reference value as an
  independent observation; in a check on two series it made the interval far too wide. The criteria were not changed.

PRE-SPECIFIED CRITERIA (7 November 2026, before the results were seen)
  K1 (primary, b=3 spread schedule): KC - RC-RW
      lower 95% CI of the mean coverage difference > -0.01 (non-inferiority)
      AND worst-month coverage difference > 0 with CI excluding zero.
  K2 (secondary, b=3 consecutive schedule): same criteria.
  Data and splits as in stage 3.
Usage: python -I stage4_kalman.py <series_dir> <out_dir> [n_processes]
"""
import sys, pathlib, json, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
P = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
NPROC = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ALPHA, DELAY, B, K_RC = 0.10, 24, 168, 72
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
def rc_rw(ts, yh, y, mask, cal_s):
    """RC-RW of stage 3: re-centre on the mean residual of the last 72 reference values + RW with the last 168 scores."""
    n = len(ts); m = len(cal_s); q = np.empty(n); bi = np.zeros(n)
    S = np.empty(m + n); S[:m] = cal_s; cnt = m; R = np.empty(n); rc = 0; bias = 0.0
    qc = _q(S[max(0, cnt - B):cnt], ALPHA); p = 0
    for t in range(n):
        ch = False
        while p < t and ts[p] + DELAY <= ts[t]:
            if mask[p] and not np.isnan(y[p]):
                S[cnt] = abs(y[p] - (yh[p] + bi[p])); cnt += 1
                R[rc] = y[p] - yh[p]; rc += 1; bias = np.mean(R[max(0, rc - K_RC):rc]); ch = True
            p += 1
        if ch:
            qc = _q(S[max(0, cnt - B):cnt], ALPHA)
        q[t] = qc; bi[t] = bias
    return q, bi


@njit
def kalman_conformal(ts, yh, y, mask, cal_norm, s_h2, R, Qh, P0, growth):
    """Day-level Kalman filter: the reference values of one calendar day are one observation (mean, variance R + s_h2/n).
    As values arrive hour by hour (24 h late), that day's update is recomputed from the state before the day."""
    n = len(ts); m = len(cal_norm)
    q = np.empty(n); bi = np.zeros(n); si = np.empty(n)
    S = np.empty(m + n); S[:m] = cal_norm; cnt = m
    bhat = 0.0; Pk = P0; t_last = ts[0]
    b_pre = 0.0; P_pre = P0; cur_day = -1; zsum = 0.0; zn = 0
    qn = _q(S[max(0, cnt - B):cnt], ALPHA); p = 0
    for t in range(n):
        ch = False
        while p < t and ts[p] + DELAY <= ts[t]:
            if mask[p] and not np.isnan(y[p]):
                S[cnt] = abs(y[p] - (yh[p] + bi[p])) / si[p]; cnt += 1
                d = ts[p] // 24
                if d != cur_day:
                    # new day: prior state = previous posterior + growth during the hours without reference data
                    b_pre = bhat
                    P_pre = Pk + (Qh * max(0, ts[p] - t_last) if growth else 0.0)
                    cur_day = d; zsum = 0.0; zn = 0
                zsum += y[p] - yh[p]; zn += 1
                Rn = R + s_h2 / zn
                K = P_pre / (P_pre + Rn)
                bhat = b_pre + K * (zsum / zn - b_pre); Pk = (1.0 - K) * P_pre
                t_last = ts[p]; ch = True
            p += 1
        if ch:
            qn = _q(S[max(0, cnt - B):cnt], ALPHA)
        Pt = Pk + (Qh * max(0, ts[t] - t_last) if growth else 0.0)
        s = np.sqrt(s_h2 + R + Pt)
        bi[t] = bhat; si[t] = s; q[t] = qn * s
    return q, bi, si


def metrics(ts, y, center, q):
    ok = ~np.isnan(y); e = np.abs(y[ok] - center[ok]); qq = q[ok]
    cov = e <= qq
    blk = ((ts[ok] - ts[0]) // 720).astype(np.int64)
    c = np.bincount(blk, weights=cov.astype(float)); n = np.bincount(blk)
    mon = c[n >= 100] / n[n >= 100]
    fin = np.isfinite(qq)
    return dict(coverage=cov.mean(), worst_month=mon.min() if len(mon) else np.nan,
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


def _ll_local_level(theta, z):
    R, Q = np.exp(theta); b, Pk, ll = z[0], R + 1e3 * np.var(z), 0.0
    for t in range(1, len(z)):
        Pp = Pk + Q; F = Pp + R; v = z[t] - b
        ll += np.log(F) + v * v / F
        K = Pp / F; b += K * v; Pk = (1 - K) * Pp
    return ll


def kalman_params(ca_index, resid):
    """Daily mean residuals z_d = b_d + u_d (u: day effect, variance R), b_d a random walk (variance Q per day).
    R, Q: maximum likelihood on the daily means of the calibration month; within-day noise s_h2 = mean within-day variance."""
    r = pd.Series(resid, index=ca_index); g = r.groupby(ca_index.floor("D"))
    s_h2 = float(np.nanmean(g.var().values)); z = g.mean().values; v = max(np.var(z, ddof=1), 1e-6)
    best = None
    for r0 in [0.1, 0.5, 0.9]:
        x0 = np.log([r0 * v, max((1 - r0) * v * 0.2, 1e-6)])
        res = minimize(_ll_local_level, x0, args=(z,), method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    R, Q = np.exp(best.x); Q = max(Q, 1e-4 * v)
    return s_h2, float(R), float(Q), float(v)


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
        yh = mdl.predict(te[feats].values); cres = ca[ycol].values - mdl.predict(ca[feats].values); cal_s = np.abs(cres)
        s_h2, R, Q, P0 = kalman_params(ca.index, cres)
        cal_norm = cal_s / np.sqrt(s_h2 + P0)
        ts = te.index.values.astype("datetime64[h]").astype(np.int64); y = te[ycol].values.astype(float)
        dd = ((te.index - ca_end).total_seconds().values // 86400).astype(int); ok = ~np.isnan(y)
        base = dict(series=name, device=device_of(name), dataset=dataset, site=site, target=target, origin=k,
                    s_h=float(np.sqrt(s_h2)), s_day=float(np.sqrt(R)), s_w_day=float(np.sqrt(Q)), P0=P0)
        scheds = [("CONSEC", b, ph, (((dd - ph) % 30) < b) if b < 30 else np.ones(len(ts), bool))
                  for b in [1, 3, 7, 30] for ph in ([0, 6, 12, 18, 24] if b < 30 else [0])]
        scheds += [("DIST", 3, ph, ((dd - ph) % 10) == 0) for ph in [0, 2, 4, 6, 8]]
        for sname, b, ph, mask in scheds:
            q, bi = rc_rw(ts, yh, y, mask, cal_s)
            rows.append(dict(base, method="RC-RW", schedule=sname, budget=b, phase=ph, mean_s=np.nan, **metrics(ts, y, yh + bi, q)))
            for mname, gr in [("KC", True), ("KC-noGrowth", False)]:
                q, bi, si = kalman_conformal(ts, yh, y, mask, cal_norm, s_h2, R, Q / 24.0, P0, gr)
                rows.append(dict(base, method=mname, schedule=sname, budget=b, phase=ph, mean_s=float(np.mean(si[ok])),
                                 **metrics(ts, y, yh + bi, q)))
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
    pd.DataFrame(rows).to_csv(OUT / "kalman_runs.csv", index=False)
    print("done", len(rows), round(time.time() - t0), "s", flush=True)
