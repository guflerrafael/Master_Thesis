# Master Thesis — Context Document

> Purpose: single source of truth for future work on this project. It consolidates the thesis proposal (`infos.txt`), the supervisor's messages, the two datasets in `data/`, the initial EDA (`eda.py`, `eda_report.html`) and an independent data audit done when this file was written (2026-10-01). Numbers marked **[audit]** were computed directly from the raw files, not copied from the proposal.

---

## 1. Thesis in one paragraph

Use deep learning on **satellite-derived Fire Radiative Power (FRP) time series** (MSG/SEVIRI, one image every 15 min) to characterise wildfire evolution and, as the headline task, **estimate the ignition location** (coordinates and an ignition-probability map). Ground truth comes from Portugal's official fire database (ICNF). The work starts in Portugal and, if successful, is extended to Europe. Final output: an *interpretable* framework that reconstructs fire origins from remote-sensing observations, usable for post-fire reconstruction and, later, operational monitoring.

## 2. Scope and objectives (from the proposal)

Main objective: investigate deep learning approaches for analysing the spatiotemporal evolution of satellite FRP and apply them to wildfire characterisation, with emphasis on learning spatial/temporal propagation patterns and relating them to wildfire attributes, notably the **ignition location**.

Specific objectives:
1. Build a spatiotemporal dataset integrating FRP observations with complementary environmental information.
2. Design and evaluate deep-learning architectures for modelling wildfire evolution from sequential satellite observations.
3. Assess different modelling strategies for extracting information from FRP time series.
4. Quantify the contribution of environmental variables (topography, land cover, fuel, meteorology) to model performance.
5. Evaluate robustness, accuracy and interpretability across wildfire scenarios.

Reference cited for motivation: Xu et al., 2025 (DL for wildfire monitoring/management).

## 3. Planned methodology

| Stage | Plan |
|---|---|
| Data compilation | Historical Portuguese fires with reliable ignition records (ICNF) + SEVIRI FRP, organised into **spatiotemporal sequences** from the earliest detectable stage onward. |
| Covariates | DEM, land cover / vegetation, fuel characteristics, meteorology (wind, temperature, humidity, precipitation) from national and European services, **harmonised onto a common spatial and temporal grid**. Named sources: NASA FIRMS; ERA5-Land hourly (1950–present, Copernicus CDS). |
| Model 1 | **ConvLSTM** (Shi et al., 2015): FRP as a temporal image sequence to capture fire-front evolution. |
| Model 2 | **CNN spatial encoder + Transformer encoder** (Vaswani et al., 2017) for long-range temporal dependencies. |
| Targets | Ignition **coordinates** and **ignition probability maps** (both models). |
| Evaluation | Independent (held-out) fire events; localisation accuracy, computational efficiency, generalisation across fire size, vegetation type and environmental setting. |
| Explainability | XAI to identify the spatiotemporal FRP patterns driving predictions. |
| Extension | If successful, scale to Europe with harmonised datasets (wider fire regimes). |

## 4. Supervisor guidance (verbatim intent of the messages in `infos.txt`)

- Two files were recovered from previous research; **not up to date**, but good enough to start.
- SEVIRI/MSG is geostationary: scene observed every **15 minutes** (the message first said 5 min, then corrected it — **15 min is right**; confirmed by the data **[audit]**).
- The satellite only detects "large" fires; the ICNF table only gives start/end dates, so small fires will not be seen. Low-confidence hotspots are probably artefacts (cloud borders, water bodies, highly reflective surfaces).
- **Suggested first step:** look at events with **very large burned areas** and study how well they match the "clouds" of hotspots in **lat–lon–time** space. "The first step is to feel the data."

## 5. Repository layout

```
Master_Thesis/
├── data/
│   ├── FRP_SEVIRI_PT_2004_2019.mat   (3.5 MB)  SEVIRI FRP hotspots
│   └── ignic_ICNF.xlsx               (51 MB)   official ICNF fire occurrences 1980–2023
├── eda.py             ydata-profiling script for the Excel file
├── eda_report.html    its output (2.6 MB, profiling of ignic_ICNF.xlsx only)
├── infos.txt          proposal outline + supervisor messages
├── thesis.md          this file
├── .venv/             Python 3.11 venv
└── .vscode/settings.json
```
Not a git repository. No modelling code exists yet.

### Environment
Python 3.11 venv with: pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, matplotlib 3.10, seaborn 0.13.2, statsmodels, openpyxl, ydata-profiling 4.18 (+ fg_data_profiling), numba, networkx, phik, PyWavelets. **Not installed:** PyTorch/TensorFlow, scikit-learn, xarray, geopandas, rasterio, netCDF/h5py — needed for the planned work (and `cdsapi` for ERA5-Land).

### `eda.py` (note)
Only 6 lines: reads `ignic_ICNF.xlsx` and writes a ydata-profiling HTML. The path is relative (`"ignic_ICNF.xlsx"`) and works only when run from inside `data/` (or after editing it to `data/ignic_ICNF.xlsx`); the report then sits wherever it was written. It does **not** touch the `.mat` file and does no cleaning, so the report inherits the data problems listed in §7.

## 6. Datasets

### 6.1 SEVIRI FRP — `data/FRP_SEVIRI_PT_2004_2019.mat`
MATLAB 5.0 file (created 2026-07-22, readable with `scipy.io.loadmat`). Five 1-D arrays of length **569,019** — each row is one *hotspot detection* (pixel × timestamp), **not** a gridded image:

| Variable | Type | Notes |
|---|---|---|
| `TIME` | string `YYYY-MM-DDTHH:MM:SS.000000000` | 15-min slots, presumed UTC (not documented — verify). |
| `LATITUDE`, `LONGITUDE` | float | pixel centre, 2-decimal precision |
| `FRP` | float (MW) | 8.8 – 1048; median 80.8, mean 143, p99 807 |
| `FIRE_CONFIDENCE` | int 1–100 | see below |

Facts **[audit]**:
- Period 2004-01-19 15:30 → 2019-12-31 23:45; **72,620 distinct timestamps** (≈ 13 % of the ~560k possible 15-min slots in 2004–2019 have any detection), 1–189 detections per timestamp (median 4; max 189 on 2005-08-20 18:30). Gaps up to 93,060 min (~65 days) between detections.
- Extent: lat 36.35–43.58 N, lon −9.39 – −6.00 W → covers Portugal **plus parts of Spain (Galicia, Extremadura, Andalucía)**; ICNF only covers Portugal.
- Minutes are exactly {0, 15, 30, 45}. **File is not sorted by time.** 896 duplicate (time, lat, lon) rows.
- Grid: 600 unique lats × 321 unique lons; spacing is irregular (~0.01–0.08°) because the native SEVIRI grid is projected (≈3 km at nadir, ≈4–5 km over Iberia). A regular lat/lon raster will need resampling or nearest-pixel snapping.
- **Only fire pixels are stored** — absence of a row means "no detection" (or cloud / no valid retrieval; cannot be distinguished). Building image sequences requires creating the zero-filled grid, and the missing-vs-zero ambiguity must be handled.
- No NaNs; all FRP > 0. FRP seems to saturate around 1,040–1,050 MW (4 values ≥ 1040).
- Detections per year (very uneven, driven by fire years): 2004 39.7k, **2005 104.0k**, 2006 44.1k, 2007 10.8k, 2008 6.4k, 2009 32.2k, 2010 37.2k, 2011 31.7k, 2012 35.6k, 2013 51.4k, 2014 6.3k, 2015 24.5k, 2016 50.1k, **2017 70.9k**, 2018 11.1k, 2019 13.2k.
- Seasonality: Aug 255.5k, Jul 95.6k, Sep 76.4k, Oct 57.7k (≈ 90 % Jul–Oct); diurnal peak 14–18 h UTC (min ≈ 07–08 h).
- **Confidence** is strongly coupled with FRP: ≥30: 99.8 %, ≥50: 97.4 %, ≥70: 81.5 %, ≥80: 61.9 %, ≥90: 32.0 %. Low-confidence rows have FRP ≈ 20–35 MW; confidence 100 has mean FRP ≈ 371 MW. Thresholding on confidence will preferentially remove the weak, early-stage detections that matter most for ignition localisation — a key design trade-off.

### 6.2 ICNF fire occurrences — `data/ignic_ICNF.xlsx`
One sheet (`Sheet1`), **813,384 rows × 9 columns** (the supervisor's "line 330415" = first coordinate row, consistent with the audit):

| Column | Meaning |
|---|---|
| `Unnamed: 0` | leftover index (not unique — restarts at several points; file is a concatenation) |
| `Data` | fire start datetime (minute resolution) |
| `DataFim` | fire end datetime |
| `Dicofre` | administrative parish code (278 unique values) |
| `Distrito` | district name (inconsistent casing/accents, 18 real districts) |
| `Concelho` | municipality name |
| `AA` | burned area, hectares |
| `lat`, `lon` | estimated ignition point — **only from 2001-01-05 onward** |

Facts **[audit]**:
- Period 1980-01-30 → 2023-12-30 (ordering is not chronological globally).
- Coordinates exist for **482,971** events (all events from 2001 on, none before). In the 2004–2019 overlap with SEVIRI there are **361,258** events, all with coordinates.
- Burned area `AA`: median 0.1 ha, p90 3 ha, p99 70 ha, p99.9 903 ha, max 53,619 ha. Heavy-tailed; 87 events ≥ 5,000 ha.
- Events with coordinates **and** inside 2004–2019: ≥1 ha: 65,730; ≥10 ha: 10,912; **≥100 ha: 2,311**; ≥500 ha: 630; ≥1,000 ha: 292; ≥5,000 ha: 40. These are the realistic candidates for a labelled training set; this is small for deep learning (see §8).
- Fire durations are short: median 1.2 h, p90 3.9 h, p99 17.7 h; 125 events last > 10 days. Annual burned area peaks: 2003 (≈472 kha), 2005 (≈347 kha), **2017 (≈540 kha)**, 2016, 2022.
- Largest events: 2017-10-15 Lousã/Coimbra (53.6 kha), 2017-10-15 Arganil (38.0), 2017-07-23 Sertã (33.6), 2017-06-17 Pedrógão Grande (30.4 kha), 2018-08-03 Monchique (26.8), 2012-07-18 Tavira (24.8), 2022-08-06 Covilhã (24.3), 2003-08-02 Chamusca (22.2).
- Largest event counts are in the north (Porto, Braga, Viseu, Aveiro) — mostly small fires.

## 7. Data-quality issues to fix before modelling [audit]

ICNF:
- **Duplicated/missing end times:** `DataFim` null for 1,651 rows; one `DataFim` = 1900-01-01 sentinel; **20,932 rows have `DataFim < Data`** (19,376 of them in 1994 alone, the rest scattered over 1980–2000, **none after 2000** — so irrelevant for the 2004–2019 overlap; `DataFim` is apparently misparsed there, and mean duration is −5,143 h unless filtered).
- `Data` null for 6 rows (all 1993); `AA` null for 1 row; **21,339 rows with AA ≤ 0**.
- **Invalid coordinates:** 32 rows outside the Portugal bounding box, including `lon=0`, `lat≈48`, `lon≈−17.2`, and a repeated (36.944, −10.378) point (Atlantic) that looks like a default/geocoding-fail value; 2 rows with zero lat or lon.
- **Inconsistent categorical text:** `Distrito` appears in upper and title case and with/without accents (37 raw spellings for 18 districts); `Concelho` likewise. Normalise before grouping.
- 6,635 duplicated rows (same Data, DataFim, Dicofre, AA).
- Only 3.4 % of `Data` timestamps are exactly midnight, so times are real; but **the time zone is undocumented** (likely Portuguese local time, UTC / UTC+1 in summer) — must be resolved before matching to SEVIRI (UTC). Time resolution is minutes, the ignition time is the *reported* alert/start time, not necessarily the physical ignition.
- Location = *estimated* ignition point; positional accuracy unknown and probably heterogeneous (reported by observers/GNR, not measured).

SEVIRI:
- Unsorted, 896 duplicates, no zero/no-data distinction, cloud/obscuration not recorded, saturation near 1,050 MW, probable false alarms at low confidence, extent includes Spain.

## 8. Preliminary matching result (quick sanity check, not the final method)

For the **292 events ≥1,000 ha with coordinates in 2004–2019** I searched SEVIRI detections within **15 km** of the ICNF ignition point and within **[Data − 6 h, DataFim + 6 h]**:
- **98.3 %** have at least one matching hotspot (mean ≈ 633 detections per event, median 344) → large fires are indeed visible, as the supervisor expected.
- Nearest-in-time first detection: median distance to the ICNF point ≈ **4.5 km** (mean 5.7 km, ≤ 15 km by construction); median first-detection time ≈ **1 h before** the reported start (IQR −3.3 h to −0.5 h). Negative lags mean the first hotspot in the window occurs *before* the official start: likely due to time-zone offset, unreliable ICNF start times, or contamination by neighbouring fires / the 6 h pre-window. **This must be disentangled** (check UTC vs local shift: a +1 h or +2 h offset would explain the median).
- These are crude numbers (no cluster definition, no confidence filter). They give a baseline: even a trivial "location of first hotspot" predictor is roughly **4–5 km** off for large fires. Any DL model must beat this (and a time-zone-correct version of it) to be meaningful.

## 9. Key challenges and open questions

1. **Label vs. observation mismatch.** ICNF gives point + start/end; SEVIRI gives hotspot clouds. Need a rigorous event ↔ hotspot-cluster matching rule (space buffer, time window, confidence filter, handling of overlapping fires; e.g. 2017-10-15 had many simultaneous large fires).
2. **Time zone** of ICNF `Data` and of SEVIRI `TIME` (assumed UTC).
3. **Sample size.** ~300 events ≥1,000 ha and ~2,300 ≥100 ha with coordinates in the overlap period; DL (ConvLSTM/Transformer) will need augmentation (spatial/rotation/flip, crops), a lower size threshold with noisier detections, pre-training/synthetic data, or extension to Europe (via FIRMS) — cf. the proposal's European extension. Decide the minimum burned area / detection count.
4. **Resolution gap.** SEVIRI pixel ≈ 3–5 km: ignition localisation to better than a pixel requires sub-pixel regression or probability-map-weighted centroid; define the metric (km error, hit within k pixels, calibration of the probability map).
5. **Temporal framing.** "From the earliest detectable stage onwards": sequence length, how many 15-min steps after first detection, handling of sparse/irregular detections (gaps due to cloud or low FRP).
6. **Cross-validation.** Split by *event* and also by year/region to avoid leakage between adjacent pixels/time steps of the same fire and between simultaneous fires.
7. **Environmental covariates** (not yet obtained): DEM, CORINE/land cover, fuel maps, ERA5-Land hourly meteorology (and possibly FIRMS MODIS/VIIRS active fires for finer spatial detail). Common grid choice needed (SEVIRI native, or fixed ~0.04–0.05° / 5 km).
8. **Data currency.** FRP data end in 2019, ICNF in 2023; the supervisor notes the files are outdated — extend SEVIRI FRP (e.g. LSA SAF FRP-PIXEL product / FIRMS) to 2020–2023 if possible.
9. **Spain overlap.** Hotspots in the SEVIRI domain outside Portugal have no labels — decide whether to mask or use as negatives / context.
10. **Baselines.** First-hotspot centroid, FRP-weighted centroid of the first *n* detections, earliest-cluster + backward fire-spread heuristic; compare against ConvLSTM and CNN–Transformer.
11. **Interpretability.** Planned XAI (saliency / attention / integrated gradients) — to define after the first model works.

## 10. Suggested roadmap

1. **Feel the data (current stage):** fix cleaning issues in §7; parse both files into tidy parquet; plot large events in lat–lon–time (the supervisor's recommended first step); compare local vs UTC time for `Data`.
2. **Event ↔ hotspot matching:** build the labelled event catalogue (event id, ignition point, first detection, cluster time series, FRP stats) and the baseline predictors above; report error distribution by fire size.
3. **Dataset construction:** rasterise to a fixed grid (per-timestep FRP, confidence, valid-pixel masks), crop patches around events, add covariates (DEM, land cover, ERA5-Land), define train/val/test by event.
4. **Model 1 — ConvLSTM**, targets: coordinate regression and probability map; ablation on sequence length and covariates.
5. **Model 2 — CNN + Transformer**, same targets and splits.
6. **Evaluation & XAI:** localisation error, hit rate, generalisation by size/vegetation/region, compute cost; saliency/attention analysis.
7. **Extension (optional):** European scale with FIRMS/harmonised datasets.
8. **Write-up.**

## 11. Reference list (from the proposal)

- Xu et al., 2025 — DL for wildfire monitoring/management (cited in the outline; full reference to be filled in).
- Shi et al., 2015 — *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting* (NeurIPS).
- Vaswani et al., 2017 — *Attention Is All You Need* (NeurIPS).
- Data sources: NASA FIRMS (earthdata.nasa.gov … /firms); ERA5-Land hourly (Copernicus CDS, `reanalysis-era5-land`); ICNF (Instituto da Conservação da Natureza e das Florestas) national fire database; EUMETSAT MSG/SEVIRI FRP (supervisor-provided extract).

## 12. Conventions for future work

- Raw files in `data/` are read-only; write derived data to a new folder (e.g. `data/processed/`) and code to `src/` or `notebooks/`.
- Keep a single timezone (UTC) internally; document any conversion of ICNF times.
- Always load the `.mat` with `scipy.io.loadmat` and flatten with `.ravel()` (arrays are shape `(1, N)`, `TIME` is `(N,)`).
- Install modelling dependencies (PyTorch, scikit-learn, xarray, geopandas, rasterio, cdsapi) into `.venv` when needed.
