# How to spend scarce reference days: drift correction and scheduling for conformal prediction intervals of low-cost air quality sensors

Code and result files for the article of the same title (Caliskan and Parlak; manuscript prepared for submission).

The study asks how a small budget of reference (co-location) days should be used to keep conformal prediction intervals of low-cost NO2, O3 and PM2.5 sensors valid. It replays three open co-location datasets under scarce, delayed reference data and compares:
- static intervals;
- widening the interval (RW);
- correcting the bias and then widening (RC-RW);
- a Kalman-filter variant (KC);

each under consecutive and spread reference schedules.

## Data

The data are not included. Download them from the original sources:

| Dataset | Source | Files used |
|---|---|---|
| QUANT | https://doi.org/10.5281/zenodo.10775692 | `QUANT_SensorSystems_hourly.csv`, `QUANT_Reference_hourly.csv`, `QUANT_DuplicateRef_hourly.csv` |
| CAPTOR | https://doi.org/10.5281/zenodo.17751285 | `P2103007.63_raw.csv`, `P2103016.63_raw.csv`, `P2103019.63_raw.csv` |
| CAPTOR reference | Generalitat de Catalunya open data portal, https://analisi.transparenciacatalunya.cat/d/tasf-thgu | see below |
| QM | https://doi.org/10.5281/zenodo.23482843 | `node_3.csv`, `node_5.csv`, `reference/node_TH2_MER.csv`, `reference/node_TH7_KEMP.csv` |

**CAPTOR reference file.** Export the hourly data of stations 08019044 (Gràcia-Sant Gervasi), 08019057 (Palau Reial) and 08019043 (Eixample) for NO2 and O3, 2022–2024, from the portal. Keep the portal's columns (`codi_eoi`, `data`, `contaminant`, `h01` … `h24`) and save the file as `CAPTOR_reference_Barcelona_2022_2024.csv`.

## Pipeline

Run every script with `python -I`. Steps 2–4 take an optional number of worker processes as the last argument; with two processes each step runs in about 5–10 minutes.

```bash
pip install -r requirements.txt

# 1. Hourly series
python -I src/prep_qm.py data/qm                         # writes data/qm/out/merged_15min_node_*.csv
python -I src/prep_series.py data out/series_root        # series/*.parquet + series_meta.json

# 2. Reference-free signals and error decomposition (criteria S1, S2)
python -I src/stage2_signals.py out/series_root out/stage2 2
python -I src/analyze_stage2.py out/stage2

# 3. Bias correction and reference schedules (criteria P1, P2)
python -I src/stage3_recenter.py out/series_root out/stage3 2
python -I src/analyze_stage3.py out/stage3
python -I src/extra_stage3.py out/stage3

# 4. Kalman-Conformal (criteria K1, K2)
python -I src/stage4_kalman.py out/series_root out/stage4 2
python -I src/analyze_stage4.py out/stage4

# 5. Numbers, tables and figures of the article
python -I paper/make_numbers.py results out/paper
python -I paper/make_figures.py results out/paper
```

Step 1 expects all data files somewhere below `data/`. Step 5 reads either the shipped `results/` folder or your own outputs; to use your own, copy `series_meta.json` and the `stage2`, `stage3` and `stage4` folders into one folder.

The pre-specified decision criteria of each stage are stated in the header of its script.

## Results

`results/` holds the result files behind every number in the article. Per-run files are gzip-compressed. `paper/make_numbers.py` turns them into the LaTeX macros used in the manuscript, so the text and the results cannot diverge.

## Requirements

Python 3.10 or later and the packages listed in `requirements.txt`. The article used scikit-learn 1.9, numpy 2.5 and pandas 3.0.

## License

The code is released under the MIT License. The data remain under the licences of their original providers.
