"""Build one hourly table per sensor series from the three datasets (QUANT, CAPTOR, QM).

QUANT rules: only manufacturer data versions that were available in real time are used (overlapping,
possibly retrospective re-calibrated versions are replaced by 'out-of-box'); a series is split when a
sensor head is replaced; values outside physical bounds and values flagged 'Error' are set to missing.

Each output table (one sensor device at one site) contains:
  index : hourly time stamp (QUANT, QM: UTC; CAPTOR: local time, matched to the reference without shift)
  f_*   : the sensor's own signals (model inputs)
  y_NO2, y_O3, y_PM25 : reference measurements (ug/m3)
  y_NO2_alt, y_O3_alt : alternative reference (QM: earlier reference download; QUANT: second Manchester instrument)

Usage: python -I prep_series.py <data_dir> <out_dir>
  The following files are searched for anywhere under <data_dir> (unzip archives first):
    merged_15min_node_3.csv, merged_15min_node_5.csv              (from prep_qm.py)
    P2103007.63_raw.csv, P2103016.63_raw.csv, P2103019.63_raw.csv (CAPTOR, Zenodo 10.5281/zenodo.17751285)
    CAPTOR_reference_Barcelona_2022_2024.csv                      (Generalitat de Catalunya export, see README)
    QUANT_SensorSystems_hourly.csv (or sensors.parquet), QUANT_Reference_hourly.csv, QUANT_DuplicateRef_hourly.csv
"""
import sys, pathlib, json
import numpy as np
import pandas as pd

V = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); (OUT / "series").mkdir(parents=True, exist_ok=True)


def find(name):
    """Find a file anywhere below the data folder."""
    hits = sorted(V.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found: unzip the data files inside {V}")
    return hits[0]
MIN_SPAN_DAYS, MIN_HOURS = 360, 4000
meta = []


def save(name, df, info):
    df = df.sort_index()
    df = df[~df.index.duplicated()]
    full = pd.date_range(df.index.min().floor("h"), df.index.max().floor("h"), freq="h")
    df = df.reindex(full)
    fcols = [c for c in df if c.startswith("f_")]
    valid = df[fcols].notna().any(axis=1)
    span = (df.index[valid].max() - df.index[valid].min()).days if valid.any() else 0
    info.update(series=name, span_days=int(span), hours_with_sensor=int(valid.sum()),
                features=fcols, targets={c: int(df[c].notna().sum()) for c in df if c.startswith("y_")})
    density = valid.sum() / max(1, span * 24)
    info["density"] = round(float(density), 3)
    if span < MIN_SPAN_DAYS or valid.sum() < MIN_HOURS or density < 0.5:   # series with long gaps (devices moved away and back) are excluded
        info["kept"] = False; meta.append(info); return
    info["kept"] = True; meta.append(info)
    df.index.name = "time"
    df.to_parquet(OUT / "series" / f"{name}.parquet")


# ---------------- QM London (15 min -> hourly) ----------------
QM_FEATS = ["no2_low_mgsv2_raw", "co_low_mgsv2_raw", "o3_low_mq131_voltage", "temperature_low_sht45_celsius",
            "humidity_low_sht45_percentage", "voc_low_sgp40_comp_raw", "pm2_5_low_sps30_ug_m3", "pm10_low_sps30_ug_m3",
            "no2_mid_alphasense_op1_voltage", "no2_mid_alphasense_op2_voltage",
            "o3_mid_alphasense_op1_voltage", "o3_mid_alphasense_op2_voltage"]
for node in ["node_3", "node_5"]:
    m = pd.read_csv(find(f"merged_15min_{node}.csv"), index_col=0, parse_dates=True)
    m.index = m.index.tz_convert(None)
    m.loc[m.n_rows_in_bin < 5, QM_FEATS] = np.nan
    h = m.resample("h").mean()
    df = pd.DataFrame(index=h.index)
    for c in QM_FEATS:
        name = {"temperature_low_sht45_celsius": "f_T", "humidity_low_sht45_percentage": "f_RH"}.get(c, "f_" + c)
        df[name] = h[c]
    df["y_NO2"], df["y_PM25"], df["y_NO2_alt"] = h.ref_no2_new, h.ref_pm25_new, h.ref_no2_old
    save(f"QM__{node}", df, dict(dataset="QM", site={"node_3": "London TH2", "node_5": "London TH7"}[node]))

# ---------------- CAPTOR Barcelona ----------------
r = pd.read_csv(find("CAPTOR_reference_Barcelona_2022_2024.csv"))
L = r.melt(id_vars=["codi_eoi", "data", "contaminant"], value_vars=[f"h{i:02d}" for i in range(1, 25)],
           var_name="h", value_name="v")
L["time"] = pd.to_datetime(L.data) + pd.to_timedelta(L.h.str[1:].astype(int) - 1, unit="h")  # h01 = 00:00-01:00
STATIONS = {"P2103007.63": (8019044, "Gracia-Sant Gervasi"), "P2103016.63": (8019057, "Palau Reial"),
            "P2103019.63": (8019043, "Eixample")}
for sid, (code, site) in STATIONS.items():
    d = pd.read_csv(find(f"{sid}_raw.csv"), parse_dates=["timestamp"]).set_index("timestamp")
    swap = sid == "P2103016.63"   # dataset readme: in this node A0/A1 = O3 and A2/A3 = NO2
    df = pd.DataFrame(index=d.index)
    df["f_we_no2"], df["f_ae_no2"] = (d.A2, d.A3) if swap else (d.A0, d.A1)
    df["f_we_o3"], df["f_ae_o3"] = (d.A0, d.A1) if swap else (d.A2, d.A3)
    df["f_T"], df["f_RH"] = d.temperature, d.humidity
    for pol, col in [("NO2", "y_NO2"), ("O3", "y_O3")]:
        ref = L[(L.codi_eoi == code) & (L.contaminant == pol)].set_index("time").v.astype(float)
        df[col] = ref.reindex(df.index)
    save(f"CAPTOR__{sid.split('.')[0]}", df, dict(dataset="CAPTOR", site=site))

# ---------------- QUANT (real-time versions, sensor-head changes, physical bounds) ----------------
pq = sorted(V.rglob("sensors.parquet"))
if pq:
    s = pd.read_parquet(pq[0])
else:
    s = pd.read_csv(find("QUANT_SensorSystems_hourly.csv"), parse_dates=["time"],
                    dtype={c: "category" for c in ["location", "instrument", "version", "measurand", "flag", "flagreason"]})
s.loc[s.flag == "Error", "measurement"] = np.nan
BOUNDS = {"NO2": (-50, 2000), "O3": (-50, 2000), "NO": (-50, 2000), "PM2.5": (-10, 2000), "PM10": (-10, 2000),
          "Temperature": (-40, 60), "RelHumidity": (0, 100), "Pressure": (800, 1100)}
for m, (lo, hi) in BOUNDS.items():
    bad = (s.measurand == m) & ((s.measurement < lo) | (s.measurement > hi))
    s.loc[bad, "measurement"] = np.nan
ALGOS = {"atmospheric", "indoor"}                     # the two real-time PurpleAir algorithms
ref = pd.read_csv(find("QUANT_Reference_hourly.csv"), parse_dates=["time"])
ref = ref.pivot_table(index=["location", "time"], columns="measurand", values="measurement")
dup = pd.read_csv(find("QUANT_DuplicateRef_hourly.csv"), parse_dates=["time"])
dup = dup[dup.qc_flag.fillna(0) == 0]
dup_no2 = dup[(dup.measurand == "NO2")].groupby("time").measurement.mean()
dup_o3 = dup[(dup.measurand == "O3")].groupby("time").measurement.mean()
FEATMAP = {"NO2": "f_NO2", "O3": "f_O3", "NO": "f_NO", "PM2.5": "f_PM25", "PM10": "f_PM10",
           "Temperature": "f_T", "RelHumidity": "f_RH", "Pressure": "f_P"}
for (inst, loc), g in s.groupby(["instrument", "location"], observed=True):
    cols, vers, cuts = {}, {}, set()
    for m, f in FEATMAP.items():
        gm = g[g.measurand == m]
        if gm.empty:
            continue
        rng = gm.groupby("version", observed=True).time.agg(["min", "max", "size"]).sort_values("min")
        vset = set(str(v) for v in rng.index)
        sequential = all(rng["min"].iloc[i] >= rng["max"].iloc[i - 1] - pd.Timedelta(days=1) for i in range(1, len(rng)))
        if len(rng) == 1 and vset <= ({"out-of-box"} | ALGOS):
            gv = gm; vers[m] = str(rng.index[0])
        elif sequential:
            gv = gm; vers[m] = "+".join(str(v) for v in rng.index)
        elif vset <= ALGOS:
            ver = rng["size"].idxmax(); gv = gm[gm.version == ver]; vers[m] = str(ver)
        else:
            # overlapping (possibly retrospective) calibration versions: use the out-of-box version only
            gv = gm[gm.version == "out-of-box"]; vers[m] = "out-of-box(only)"
            if gv.empty:
                continue
        # sensor heads: concurrent channels are averaged, successive heads define cut points
        hr = gv.groupby("sensornumber", observed=True).time.agg(["min", "max"]).sort_values("min")
        if len(hr) > 1:
            seqh = all(hr["min"].iloc[i] >= hr["max"].iloc[i - 1] - pd.Timedelta(hours=2) for i in range(1, len(hr)))
            if seqh:
                for i in range(1, len(hr)):
                    cuts.add(hr["min"].iloc[i].floor("D"))
        cols[f] = gv.groupby("time").measurement.mean()
    df = pd.DataFrame(cols)
    if df.empty:
        continue
    rl = ref.loc[loc] if loc in ref.index.get_level_values(0) else None
    for m, col in [("NO2", "y_NO2"), ("O3", "y_O3"), ("PM2.5", "y_PM25")]:
        df[col] = rl[m].reindex(df.index) if (rl is not None and m in rl) else np.nan
    if loc == "Manchester":
        df["y_NO2_alt"], df["y_O3_alt"] = dup_no2.reindex(df.index), dup_o3.reindex(df.index)
    cuts = sorted(cuts)
    edges = [df.index.min()] + cuts + [df.index.max() + pd.Timedelta(hours=1)]
    for k in range(len(edges) - 1):
        seg = df[(df.index >= edges[k]) & (df.index < edges[k + 1])]
        if seg.empty:
            continue
        name = f"QUANT__{inst}__{loc}" + (f"__h{k + 1}" if len(edges) > 2 else "")
        save(name, seg, dict(dataset="QUANT", site=loc, instrument=str(inst), version=vers, head_segment=k + 1,
                             head_changes=[str(c.date()) for c in cuts]))

json.dump(meta, open(OUT / "series_meta.json", "w"), indent=1, default=str)
k = [m for m in meta if m["kept"]]
print(f"{len(k)} / {len(meta)} series kept")
for m in meta:
    print(m["kept"], m["series"], m.get("version", ""), m["span_days"], m["hours_with_sensor"], m["targets"])
