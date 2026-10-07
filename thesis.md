# Master Thesis — Context Document

> Purpose: single source of truth for future work on this project. It consolidates the thesis proposal (`infos.txt`), the supervisor's messages, the two datasets in `data/raw/` and an independent data audit done when this file was first written (2026-10-01). Numbers marked **[audit]** were computed directly from the raw files, not copied from the proposal.
>
> Last updated 2026-10-07. Status: data cleaned, time zone resolved, first exploration done (see §13). Step-by-step methods log, with every decision and row count: `notes.md`.

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
│   ├── raw/                           as received, never modified
│   │   ├── FRP_SEVIRI_PT_2004_2019.mat   (3.5 MB)  SEVIRI FRP hotspots
│   │   └── ignic_ICNF.xlsx               (51 MB)   official ICNF fire occurrences 1980–2023
│   └── processed/                     written by src/clean_data.py
│       ├── icnf_fires.parquet
│       └── seviri_frp.parquet
├── src/
│   ├── clean_data.py          raw → processed parquet (cleaning + UTC conversion)
│   ├── eda.py                 profiling reports, raw and processed
│   ├── timezone_check.py      ICNF time zone test → figures/timezone_lag.png
│   └── plot_largest_fires.py  largest fires in lat–lon–time → figures/largest_fires_lat_lon_time.png
├── outputs/profiling/{raw,processed}/{icnf,seviri}_eda_report.html
├── figures/
├── notes.md           methods log (basis for the write-up)
├── thesis.md          this file
├── infos.txt          proposal outline + supervisor messages
├── requirements.txt
├── .gitignore         excludes .venv/, __pycache__/, .DS_Store
└── .vscode/settings.json
```
Git repository (branch `main`, no remote yet), first commit 2026-10-07. No modelling code exists yet. Reproduce with the four scripts in the order given in `notes.md` §0, run from the repository root.

### Environment
Python 3.11 venv (`.venv/`, not in git). `requirements.txt` lists only what `src/` imports: pandas 2.3.3, numpy 2.3.5, scipy 1.16.3, openpyxl 3.1.5, pyarrow 25.0.1, matplotlib 3.10.0, fg-data-profiling 4.19.1 (imported as `data_profiling`). The venv also contains seaborn, statsmodels, ydata-profiling and others not used by the code. **Not installed:** PyTorch/TensorFlow, scikit-learn, xarray, geopandas, rasterio, netCDF/h5py — needed for the planned work (and `cdsapi` for ERA5-Land).

## 6. Datasets

### 6.1 SEVIRI FRP — `data/raw/FRP_SEVIRI_PT_2004_2019.mat`
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

### 6.2 ICNF fire occurrences — `data/raw/ignic_ICNF.xlsx`
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

> **Status 2026-10-07:** handled in `src/clean_data.py` (details in `notes.md` §1.3): duplicates dropped, invalid end times and coordinates set to missing and flagged, text normalised, times converted to UTC. Additions found during cleaning: the 1994 `DataFim` < `Data` rows have end date 1970-01-01 (only the time of day survived); the 23 repeated Atlantic points are 2014–2015 fires in Porto, Lisboa and Setúbal; `Dicofre` is a municipality code (1-to-1 with `Concelho`), not a parish code. Still open: coordinates inside the bounding box but in the wrong place, no SEVIRI confidence filter, Spain not masked.

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

> **Update 2026-10-07:** the "first detection ~1 h before start" result below was the time zone. ICNF times are Portuguese local time (§13); after converting to UTC, SEVIRI first sees a fire ~10–20 min *after* the reported start.

For the **292 events ≥1,000 ha with coordinates in 2004–2019** I searched SEVIRI detections within **15 km** of the ICNF ignition point and within **[Data − 6 h, DataFim + 6 h]**:
- **98.3 %** have at least one matching hotspot (mean ≈ 633 detections per event, median 344) → large fires are indeed visible, as the supervisor expected.
- Nearest-in-time first detection: median distance to the ICNF point ≈ **4.5 km** (mean 5.7 km, ≤ 15 km by construction); median first-detection time ≈ **1 h before** the reported start (IQR −3.3 h to −0.5 h). Negative lags mean the first hotspot in the window occurs *before* the official start: likely due to time-zone offset, unreliable ICNF start times, or contamination by neighbouring fires / the 6 h pre-window. **This must be disentangled** (check UTC vs local shift: a +1 h or +2 h offset would explain the median).
- These are crude numbers (no cluster definition, no confidence filter). They give a baseline: even a trivial "location of first hotspot" predictor is roughly **4–5 km** off for large fires. Any DL model must beat this (and a time-zone-correct version of it) to be meaningful.

## 9. Key challenges and open questions

1. **Label vs. observation mismatch.** ICNF gives point + start/end; SEVIRI gives hotspot clouds. Need a rigorous event ↔ hotspot-cluster matching rule (space buffer, time window, confidence filter, handling of overlapping fires; e.g. 2017-10-15 had many simultaneous large fires).
2. ~~**Time zone** of ICNF `Data` and of SEVIRI `TIME`.~~ Resolved: ICNF = Europe/Lisbon local time (§13); SEVIRI still assumed UTC (consistent with the result).
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

1. **Feel the data (done 2026-10-07):** fix cleaning issues in §7; parse both files into tidy parquet; plot large events in lat–lon–time (the supervisor's recommended first step); compare local vs UTC time for `Data`.
2. **Event ↔ hotspot matching (next):** build the labelled event catalogue (event id, ignition point, first detection, cluster time series, FRP stats) and the baseline predictors above; report error distribution by fire size.
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

- Raw files in `data/raw/` are read-only; derived data goes to `data/processed/`, code to `src/`, plots to `figures/`, reports to `outputs/`.
- Work from the processed parquet files, not the raw files.
- Keep a single timezone (UTC) internally: use `start_utc`/`end_utc` for ICNF and `time_utc` for SEVIRI.
- Always load the `.mat` with `scipy.io.loadmat` and flatten with `.ravel()` (arrays are shape `(1, N)`, `TIME` is `(N,)`).
- Log every step and decision in `notes.md`.
- Install modelling dependencies (PyTorch, scikit-learn, xarray, geopandas, rasterio, cdsapi) into `.venv` when needed, and add them to `requirements.txt`.

## 13. Results so far (2026-10-07)

Details and numbers in `notes.md`.

- **Cleaning** (`notes.md` §1.3): ICNF 813,384 → 806,926 rows (477,674 with valid coordinates; 361,091 in 2004–2019, of which 2,311 ≥ 100 ha and 292 ≥ 1,000 ha). SEVIRI 569,019 → 568,123 rows. The ICNF ".csv" copy was not lossless (timestamps shifted by −1 s in 1989–1997) and was deleted; the ".xlsx" is the source.
- **Time zone** (`notes.md` §1.4): ICNF times are Portuguese local time. The lag between ICNF start and first SEVIRI detection jumps by ~0.9 h at the daylight saving switch (also when comparing only the same months). UTC columns added (§1.5); 42 start times in DST-switch hours could not be converted.
- **Largest fires in lat–lon–time** (`notes.md` §2.1, 12 largest fires 2004–2019):
  - 10 of 12 are detected within ~1 h of the ICNF start, first hotspot 2–6 km (1–2 SEVIRI pixels) from the ICNF ignition point; hotspots then spread away from it over time.
  - Pedrógão Grande and Góis (2017-06-17) are not detected for 16–20 h: very few detections in the whole domain that afternoon (smoke/pyrocumulus or rejected retrievals). Missing hotspots ≠ no fire.
  - Simultaneous neighbouring fires (2017-10-15, Arouca 2016) mix their hotspots, so hotspots must be assigned to individual fires (spatio-temporal clustering) before events can be used as samples.
  - ICNF end times do not match the hotspot activity; the hotspots should define the event end.
