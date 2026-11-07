"""Build 15-minute QM tables (sensor means + reference) from the raw QM Air Quality Network files.

Inputs (QM Air Quality Network Data v1.0.1, Zenodo 10.5281/zenodo.23482843):
  <data_dir>/data_qm/node_3.csv, node_5.csv
  <data_dir>/data_qm/reference/node_TH2_MER.csv, node_TH7_KEMP.csv
  <data_dir>/data_old/node_TH2_MER.csv, node_TH7_KEMP.csv   (optional: earlier reference download, used only for
                                                             the alternative reference column y_NO2_alt)
Output: <data_dir>/out/merged_15min_<node>.csv (one row per 15-min bin; sensor means, number of raw rows,
        reference values)
Usage: python -I prep_qm.py <data_dir>
"""
import sys, pathlib
import numpy as np
import pandas as pd

S = pathlib.Path(sys.argv[1])
OUT = S / "out"; OUT.mkdir(exist_ok=True)

SENS = ["no2_low_mgsv2_raw", "co_low_mgsv2_raw", "o3_low_mq131_voltage",
        "temperature_low_sht45_celsius", "humidity_low_sht45_percentage",
        "voc_low_sgp40_comp_raw", "pm2_5_low_sps30_ug_m3", "pm10_low_sps30_ug_m3",
        "no2_mid_alphasense_op1_voltage", "no2_mid_alphasense_op2_voltage",
        "o3_mid_alphasense_op1_voltage", "o3_mid_alphasense_op2_voltage",
        "co_mid_alphasense_op1_voltage", "co_mid_alphasense_op2_voltage"]
REF = ["no2_high_met_one_ug_m3", "pm2_5_high_met_one_ug_m3"]
PAIRS = {"node_3": "node_TH2_MER", "node_5": "node_TH7_KEMP"}


def load(path, cols):
    df = pd.read_csv(path, usecols=["timestamp_utc"] + cols)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.set_index("timestamp_utc").sort_index()


for node, ref in PAIRS.items():
    raw = load(S / "data_qm" / f"{node}.csv", SENS)
    n_raw = len(raw)
    agg = raw.resample("15min").agg(["mean", "count"])
    sens = pd.DataFrame({c: agg[(c, "mean")] for c in SENS})
    sens["n_rows_in_bin"] = agg[(SENS[0], "count")]
    new = load(S / "data_qm" / "reference" / f"{ref}.csv", REF).resample("15min").mean()
    new.columns = ["ref_no2_new", "ref_pm25_new"]
    m = sens.join(new, how="outer")
    old_path = S / "data_old" / f"{ref}.csv"
    if old_path.exists():
        old = load(old_path, REF).resample("15min").mean()
        old.columns = ["ref_no2_old", "ref_pm25_old"]
        m = m.join(old, how="left")
    else:
        m["ref_no2_old"], m["ref_pm25_old"] = float("nan"), float("nan")
    m.index.name = "timestamp_utc"
    m.to_csv(OUT / f"merged_15min_{node}.csv", float_format="%.5g")
    print(node, "raw rows", n_raw, "-> 15-min bins", len(m),
          "| bins with sensor data", int(m[SENS[0]].notna().sum()),
          "| missing per col (%):", (m[SENS + ["ref_no2_new", "ref_pm25_new"]].isna().mean() * 100).round(1).to_dict())
