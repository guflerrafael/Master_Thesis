# Data pipeline — technical reference

> Purpose: one place that describes, step by step, how the raw SEVIRI and ICNF files become model inputs, labels, loss weights, predictions and scores — with the exact rules, constants, file names and row counts as implemented in `src/`. `notes.md` is the chronological methods log (why and when each decision was made); this file is the current state of the pipeline (what the code does now). Section references like "notes §6.12" point to `notes.md`.
>
> State: 2026-11-13, build 2004–2025 (`build_info.json`: `georef_fix = true`, change date 2017-12-01, shift 0.5 px; label period 2004-01-20 → 2025-12-30; persistent pixels from 2004–2019). Sources: SEVIRI extract 2004–2019 + LSA SAF download 2020–2025 (notes §7), ICNF SGIF export 2001–2025 (notes §8). The split is a default to be revisited (notes §8.8). Reference training configuration = `train.DEFAULTS` (v1, hourly); the best model `cnn3d_final` uses the same data settings with `--arch cnn3d --neg-ratio 2`. Figures: `figures/pipeline/` (`src/plot_pipeline_figures.py`). A non-technical walkthrough with diagrams exists as a Claude doc ("Data pipeline: from raw SEVIRI/ICNF to model inputs, labels and scores"); it describes the 2004–2019 build.
>
> All trained models so far (`models/ablations_georef/`, `models/final/`) were trained on the 2004–2019 build. For 2004–2019 the current build is identical to it (§11), so they load and score unchanged.

## 0. Overview

```
data/raw/FRP_SEVIRI_PT_2004_2019.mat ──┐
data/processed/seviri_frp_2020_2025.parquet (download_msg_frp.py --merge) ──clean_data.py──► seviri_frp_2004_2025.parquet ──cluster_hotspots.py──► seviri_frp_events.parquet
data/raw/icnf/Registos_Incendios_SGIF_*.xlsx (2001–2025) ──┐                                                                                (+ seviri_events, seviri_nuclei)
data/raw/ignic_ICNF.xlsx (1980–2000 only) ─────────────────┴──clean_data.py──► icnf_fires.parquet ───────────────────────────────────┐
                                                                                                                                     ▼
                                        build_dataset.py (+ grid.py, tiling_design.py) ──► data/processed/model/
                                            detections · fires · labels · periods · rekindles .parquet · static.npz · build_info.json
                                                                                                                                     │
            ┌──────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────┘
            ▼                                              ▼
 dataset.py FireData.sample(period, crop) → x, y, w     baselines.py (nucleus births → points)
            ▼                                              │
 train.py features() → model → heatmap → peaks → points    │
            └──────────────────────► evaluate.py ◄─────────┘   (match to ICNF within 5 / 10 km → AP, recall, F1, error)
```

`clean_data.py` also writes `seviri_frp.parquet` (the 2004–2019 extract alone), which the exploratory scripts of notes §1–2 read.

Invariants:
- ICNF never enters the model input (no crops, sequence starts or masks derived from ICNF). It only selects/labels samples and scores predictions (thesis.md §12). The SGIF fields (cause, fire weather, land cover) are for reporting only.
- All times are UTC internally (`time_utc`, `start_utc`, `end_utc`).
- Spatial unit = one native SEVIRI pixel ("cell"); no resampling anywhere.
- The test split (2024–2025) is never evaluated during development; `evaluate_test.py` and `plot_test_examples.py` need `--confirm-test` for it.

## 1. Raw data

| | SEVIRI FRP extract | SEVIRI LSA SAF download | ICNF SGIF export | ICNF old file |
|---|---|---|---|---|
| File | `data/raw/FRP_SEVIRI_PT_2004_2019.mat` (3.5 MB, `scipy.io.loadmat`, arrays `(1, N)` → `.ravel()`) | day files `data/raw/msg_frp_pt/<year>/<yyyymmdd>.parquet` (not versioned), merged by `src/download_msg_frp.py --merge` into `data/processed/seviri_frp_2020_2025.parquet` | `data/raw/icnf/Registos_Incendios_SGIF_{2001_2010,2011_2020,2021_2025}.xlsx` (157 MB, not versioned), sheets `SGIF_<years>`, `Legenda`, `SQL` | `data/raw/ignic_ICNF.xlsx` (51 MB, one sheet) |
| Row | one fire pixel at one 15-min scan | idem | one reported fire | one reported fire |
| Rows | 569,019 | 210,372 | 497,478 | 813,384 |
| Period | 2004-01-19 15:30 → 2019-12-31 23:45 | files 2020-01-01 → 2025-12-31; detections 2020-01-01 04:00 → 2025-11-28 15:45 | 2001-01-05 → 2025-12-31 18:10 | 1980–2023; lat/lon from 2001-01-05 |
| Used for | 2004–2019 | 2020–2025 | 2001–2025 | 1980–2000 only |
| Columns | `TIME`, `LATITUDE`, `LONGITUDE` (2 decimals), `FRP` [MW], `FIRE_CONFIDENCE` (1–100) | the same five fields, plus 20 more product fields in the day files | 41 fields, see notes §8.1 | `Data`, `DataFim`, `Dicofre`, `Distrito`, `Concelho`, `AA` [ha], `lat`, `lon` |
| Time zone | UTC (slot of the LSA SAF file name, notes §7.2) | UTC | Europe/Lisbon local time (§2.2) | Europe/Lisbon local time |

Properties that shape the pipeline:
- Only fire pixels are stored: absence = no detection, cloud, smoke or missing scan (indistinguishable, no cloud mask).
- FRP 8.3–1,147 MW (8.8–1,048 MW in 2004–2019), median 86 MW. Confidence rises with FRP (low-confidence rows ≈ 20–35 MW).
- Extent of all detections: lat 36.35–43.58, lon −9.39 to −6.00 (Portugal + parts of Spain).
- The two SEVIRI sources share the same pixel lat/lon, grid, time label and corrected geo-referencing frame (notes §7.2–7.3). FRP-PIXEL version 2.4.0 on every downloaded day; satellite MSG4 until 2023-03-21, MSG3 after.
- ICNF start = reported alert time (uncertain by ~1 h); position = estimated ignition point (72 % of all points 2001–2023 shared with other fires; for fires ≥ 10 ha 39.5 % in 2004–2010, 2.2 % in 2024–2025); end times unreliable.

## 2. Cleaning (`src/clean_data.py`)

Outputs: `data/processed/icnf_fires.parquet` (821,433 rows), `data/processed/seviri_frp.parquet` (568,123 rows, 2004–2019), `data/processed/seviri_frp_2004_2025.parquet` (778,495 rows). Rows with unusable fields are flagged and kept; downstream code filters on the flags. Log: `outputs/clean_data_log.txt`.

### 2.1 Rules

| Table | Step | Rule | Rows |
|---|---|---|---|
| ICNF | sources | SGIF export for 2001–2025; `ignic_ICNF.xlsx` for rows before 2001 (and the 6 without a start); the `.csv` copy of the old file had timestamps shifted by −1 s in 1989–1997 (deleted) | 330,413 + 497,478 |
| ICNF | columns | old file: drop `Unnamed: 0`, rename → `start_local`, `end_local`, `municipality_code`, `district`, `municipality`, `burned_area_ha`. SGIF: `DataHoraAlerta`, `DataHora_Extincao` (floored to the minute), `DTCCFR // 100`, `Distrito`, `Concelho`, `AreaTotal_ha`, `Latitude`, `Longitude` | – |
| ICNF | text | NFKD → ASCII, upper case, whitespace collapsed (`normalise_text`) | districts 37 → 18 spellings |
| ICNF | missing start | drop | 6 |
| ICNF | duplicates | exact duplicates over the eight columns above (SGIF: double entries with different SGIF codes, notes §8.2); first row in file order kept | 6,452 (5,265 from SGIF) |
| ICNF | end time | NaT if missing, year < 1980, or `end < start`; flag `end_invalid` | 22,410 |
| ICNF | coordinates | NaN if outside lat 36.8–42.2 / lon −9.6 to −6.1; flag `coord_invalid` | 32 |
| ICNF | UTC | `tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")` → `start_utc`, `end_utc`, `first_intervention_utc` | 42 / 46 / 17 NaT (DST switch hours) |
| ICNF | order, id | sort by `start_local`, `municipality_code`, `burned_area_ha`; `fire_id` = 0…N−1; `duration_h`. `fire_id` 0–806,925 have the same content as in the old table, except the end time of fire 806684 (notes §8.2) | – |
| ICNF | municipality | `municipality_code` nullable (`Int16`): one 2024 row has no district, municipality or parish | 1 |
| SEVIRI | columns | `time_utc` (labelled UTC), `lat`/`lon` rounded to 2 decimals, `frp_mw` float32, `confidence` int8 | – |
| SEVIRI | duplicates | drop exact duplicates; assert no conflicting (time, lat, lon) duplicates | 896 (extract), 0 (download) |
| SEVIRI | merge | extract + `seviri_frp_2020_2025.parquet`; assert equal dtypes, no overlap in time, no conflicting duplicates | 568,123 + 210,372 |
| SEVIRI | order | sort by `time_utc`, `lat`, `lon` | – |

Columns of `icnf_fires.parquet` taken from SGIF (NaN before 2001): `source`, `sgif_code`, `anepc_code`, `parish_code`, `area_forest_ha`, `area_shrub_ha`, `area_agric_ha`, `first_intervention_local`/`_utc`, `cause_code`, `cause_type`, `cause_group`, `rekindle` (nullable: NA where the cause is unknown), `alert_source`, `fwi`, `isi`, `ffmc`, `dmc`, `dc`, `bui`, `dsr` (−80 → NaN), `coord_check_km` (lat/lon vs. the PT-TM06/ETRS89 coordinates of the same row). Cause coverage: below 13 % of all fires in 2001–2006, 49–64 % in 2008–2010, 70–92 % in 2011–2020, at least 96 % from 2021; for fires ≥ 50 ha at least 95 % from 2008. 29,374 rekindle records in total.

Not done on purpose: no confidence threshold, no FRP cap, Spain kept, zero-area ICNF fires kept.

### 2.2 Time zone (`src/timezone_check.py`, notes §1.4; `src/check_new_years.py`)
- Lag first SEVIRI detection (≤ 5 km, [start − 6 h, start + 24 h]) − recorded start, for fires ≥ 10 ha without earlier activity nearby: 2004–2019 summer (DST) −0.70 h, winter +0.17 h → jump ≈ 0.9 h at the DST switch, also within the same months. 2020–2023: +0.65 h, 2024–2025: +0.61 h. Conclusion: ICNF = local time in all years. After conversion the median lag is +10–20 min.

### 2.3 Fire time window
All downstream steps use ICNF fires with `start_utc` in `tiling_design.LABEL_PERIOD` = [2004-01-20, 2025-12-30], i.e. fully inside the SEVIRI record: 411,259 fires with valid coordinates and UTC start. The exploratory scripts and the clustering diagnostics keep `eda_figures.PERIOD` = [2004-01-20, 2019-12-30].

## 3. Spatial frame (`src/grid.py`)

### 3.1 Native grid
- Projection: `+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y +units=m`; sampling distance at nadir `SSD = 3000.403165817 m`.
- `native_rc(lat, lon)`: `row = y / SSD`, `col = x / SSD` (continuous). Only `lon_0 = 0` fits (93–100 % of pixels within ±0.25 px of a node in every year vs. ~25 % by chance). For 2020–2025 the product's own indices confirm it on every row: `ABS_LINE = 1857 − row`, `ABS_PIXEL = 1857 + col`.
- All 9,017 distinct pixels → 9,017 distinct native pixels; max residual 0.42 px (2-decimal rounding).
- Domain: rows 1206–1382, cols −260 to −155 → `H × W = 177 × 106`. Image index `i = 1382 − row` (north at top), `j = col + 260` (west at left). One cell over Portugal ≈ 3.2 km E–W × 4.3–4.8 km N–S.
- Detections: `(i, j)` rounded to int16 (exact, they are pixel centres). ICNF ignitions: continuous `(i_f, j_f)`. Predictions: continuous, converted back with `ij_to_latlon`.

### 3.2 Static maps (`static.npz`, built in `build_dataset.static()`)

| Key | Definition | Cells |
|---|---|---|
| `pt` | cell centre inside the mainland Portugal polygon (Natural Earth 1:10m, largest Portugal part) | 6,127 |
| `pt_loss` | `binary_dilation(pt, 3 × 3)` (coastal / border ignitions inside) | 6,760 |
| `persistent` | cell of a detection flagged persistent (§4.3) | 142 |
| `lat`, `lon` | cell-centre coordinates | 18,762 |

### 3.3 Geo-referencing correction (notes §6.11–6.12)
- SEVIRI Level 1.5 positions before December 2017 are shifted 1.5 km N and W at the sub-satellite point = 0.5 native px (satpy reader documentation). Measured from fires: label − first detection changes by +0.46 px E, −0.52 px N across the change.
- Constants: `GEOREF_CHANGE = "2017-12-01"` (first corrected day; exact day undocumented, only 2 targets between mid-Oct 2017 and Feb 2018), `GEOREF_SHIFT_PX = 0.5`.
- `georef_shift(t)` = 0.5 before the change, else 0. `true_to_sensor_ij`: `i − s, j − s` (north-west); `sensor_to_true_ij`: `i + s, j + s`.
- 2020–2025 are in the corrected frame: ICNF − first detection +0.22 / +0.02 px (S / E) in 2020–2023 and +0.19 / +0.07 px in 2024–2025, against +0.17 / +0.02 px in 2018–2019 (notes §7.3, §8.3).
- Frames used:

| Object | Frame | Where |
|---|---|---|
| detections, model input | sensor (as delivered) | `detections.parquet` |
| labels (`i_f`, `j_f`), rekindle records, "seen" test | sensor, shift evaluated at the label's **period** start (seen test: at the fire start) | `labels.parquet`, `rekindles.parquet`, `build_dataset.fires/labels/rekindles` |
| ICNF positions for scoring | true | `fires.parquet`, `rekindles.parquet` (`lat`, `lon`) |
| predictions, nucleus births | sensor → true before scoring | `train.maps_to_points`, `baselines.predictions` |

- Effect (2004–2019 build): targets 2,705 → 2,821 (+116 fires whose first detection had been pushed outside the 5 km seen radius); median label − nucleus birth before the change −0.01 px E, −0.22 px N (after: 0.00 / −0.16) → offset removed. Remaining ~0.2 px N–S offset in all periods, larger for night detections (notes §8.3; possible terrain parallax, `model_review.md` §6.1).
- Figure: `figures/pipeline/georef_frames.png`.

## 4. Hotspot clustering (`src/cluster_hotspots.py`, notes §3)

SEVIRI only; ICNF used for diagnostics only (fires of `eda_figures.PERIOD`, 2004–2019). Since notes §4.0 the clustering is **not** in front of the model; it provides the baseline, flags and the optional `births` channel. Input: `seviri_frp_2004_2025.parquet`, clustered in one pass (one event crosses 2020-01-01). Event and nucleus ids of 2004–2019 are the same as in the 2004–2019 build.

### 4.1 Events
- Link rule: pixel centres ≤ `D_KM = 6` km (local metric coordinates `xy_km`) and |Δt| ≤ `GAP_H = 6` h, same time step included. Events = connected components (union–find over time-sorted detections; `cKDTree` neighbour lists per pixel; per pixel only the last detection is kept as link partner).
- 6 km = 8-neighbourhood (nearest neighbour ~3.2 km, 8th ~6.3 km). 6 h from the sensitivity grid (notes §3.1): merged fires ≥ 100 ha 30.8 %, isolated fires ≥ 90 % intact 88.7 %.
- 31,873 events; 8,265 with ≥ 10 detections hold 92.0 % of detections (2004–2019 build: 24,690; 6,620; 91.4 %).

### 4.2 Nuclei
- Per time step: a detection inherits the nucleus of the **nearest** linked earlier detection; else the nucleus of the nearest already-assigned neighbour in the same step (iterated); else it is a **birth**, and connected new detections of the step form one new nucleus.
- Per nucleus (`seviri_nuclei.parquet`): `t_birth`, birth position = FRP-weighted centroid of the birth step, `n_det`, `fre_tj` (Σ FRP × 900 s), `order_in_event`, `dist_active_km` (nearest detection of another nucleus in the previous 6 h, ≤ 100 km), `reappeared` (detections ≤ 6 km in [birth − 48 h, birth − 6 h)).
- 33,655 nuclei; 34.6 % of those with ≥ 3 detections are reappearances (2004–2019 build: 25,954; 32 %).

### 4.3 Flags carried into `detections.parquet`

| Column | Definition | Used by |
|---|---|---|
| `persistent` | pixel with ≥ 20 active days and ≥ 50 % of them outside Jun–Oct (`eda_figures.pixel_stats`), computed on detections before `tiling_design.PERSISTENT_END` = 2020-01-01 only; pixels first active later are False | `static["persistent"]` → input channel; baseline variants |
| `in_pt` | detection inside the mainland polygon | `periods.n_det_pt`, hard-negative periods |
| `event_id`, `nucleus_id` | §4.1–4.2 | `births` channel (ablation), baseline |

Caveat: `persistent` uses 2004–2019, i.e. the training and validation years and the block `train2` (SEVIRI-only, no labels, but not strictly causal for the earlier years). Computed on all years it would have 157 cells (notes §7.4).

### 4.4 Baseline (`src/baselines.py`)
Every nucleus born inside a target period = one predicted ignition at its birth position (moved to the true frame before scoring). Score = FRE of the nucleus inside the period's input window up to the end of the 6 h tail. Variants: without reappeared; without reappeared/persistent; first-hour centroid. Validation 2015–2016: AP 0.448 @ 5 km, 0.577 @ 10 km (unchanged by the extension).

## 5. Sample design (`src/tiling_design.py`, `build_dataset.periods()`)

### 5.1 Constants

| Constant | Value | Meaning |
|---|---|---|
| `PERIOD_H` | 12 | target period length [h] |
| `PERIOD_START_UTC` | 6 | periods [06, 18) and [18, 06) UTC |
| `HISTORY_H`, `TAIL_H` | 6, 6 | input before / after the period [h] |
| `T` (`dataset.T`) | 96 | 15-min slots per window (24 h) |
| `TARGET_HA`, `IGNORE_HA` | 50, 10 | target / candidate size thresholds [ha] |
| `SEEN_KM`, `SEEN_PRE_H`, `SEEN_POST_H` | 5, 1, 6 | "seen" test |
| `LABEL_PERIOD` | 2004-01-20 → 2025-12-30 | fires and periods with labels |
| `PERSISTENT_END` | 2020-01-01 | persistent pixels from earlier detections only |
| `SEVIRI_FILE` | `seviri_frp_2004_2025.parquet` | input of clustering and design numbers |
| `SPLIT` | train 2004–2014, val 2015–2016, train2 2017–2019, train3 2020–2023, test 2024–2025 | by calendar year of the period; default to be revisited (notes §8.8). `build_dataset.split_of` fails on a year without a split |

### 5.2 Periods
- `period_of(t)` = `floor(t − 6 h, 12 h) + 6 h`: every fire belongs to the period containing its UTC start.
- 16,032 periods (2004-01-19 06 UTC → 2025-12-29 18 UTC). Per period: `n_det` (detections in [period − 6 h, period + 18 h)), `n_det_pt` (same, mainland only), `n_targets`, `n_ignore`.
- Windows of consecutive periods overlap by 12 h; every slot appears in two samples.

| Split | Periods | with detections | with ≥ 1 target | hard-negative (detections in PT, no target) | Targets |
|---|---|---|---|---|---|
| train | 8,000 | 4,469 | 805 | 2,784 | 1,979 |
| val | 1,462 | 844 | 163 | 540 | 403 |
| train2 (2017–2019) | 2,190 | 1,269 | 215 | 896 | 439 |
| train3 (2020–2023) | 2,922 | 1,562 | 213 | 1,062 (1,058 usable, §8.1) | 321 |
| test (2024–2025) | 1,458 | 747 | 113 | 508 | 232 |

Targets per positive period: 1 → 857, 2–5 → 547, 6–20 → 99, > 20 → 6 (max 32, 2017-10-15 06 UTC). 76 % of targets start in a 06–18 UTC period; median 7.1 h after the period start. The test of notes §6.17 (2017–2019, 2,186 periods then) is now `train2`; it has 4 more periods because the label period no longer ends on 2019-12-30.

## 6. Labels and targets (`build_dataset.fires()`, `labels()`, `rekindles()`, `tiling_design.seen()`)

### 6.1 Candidate fires → `fires.parquet` (13,047 rows)
Filter: valid `lat`/`lon`, valid `start_utc` in the fire time window (§2.3), `burned_area_ha ≥ IGNORE_HA = 10`. The 398,212 smaller fires are not in any label table (= background).

Columns: `fire_id`, `start_utc`, `end_utc`, true `lat`/`lon`, true `i_f`/`j_f`, `burned_area_ha`, `size_class` (10–50, 50–100, 100–500, 500–1k, ≥ 1k ha; `right=False`), `municipality`, `district`, `seen`, `is_target`, `isolated`, `period`, `year`, `split`; for reporting only `sgif_code`, `rekindle`, `cause_type`, `fwi`, `area_forest_ha`, `area_shrub_ha`, `area_agric_ha`.

- **`seen`**: ∃ SEVIRI detection with haversine distance ≤ 5 km from the ICNF point (moved into the sensor frame at the fire start) and `time_utc` ∈ [start − 1 h, start + 6 h). Any detection counts (Spain, persistent pixels, other fires).
- **`is_target`** = `burned_area_ha ≥ 50` and `seen`.
- **`isolated`**: no other ICNF fire ≥ 10 ha within 20 km and ±72 h (`proximity_analysis.count_neighbours`). Reporting subgroup only, never an input.

| Size class | Fires | Seen | Seen share | Role |
|---|---|---|---|---|
| 10–50 ha | 8,455 | 3,600 | 43 % | ignore (seen or not) |
| 50–100 ha | 1,862 | 1,228 | 66 % | seen → target, unseen → ignore |
| 100–500 ha | 1,963 | 1,483 | 76 % | idem |
| 500–1,000 ha | 384 | 326 | 85 % | idem |
| ≥ 1,000 ha | 383 | 337 | 88 % | idem |

### 6.2 Roles → `labels.parquet` (13,322 rows: 3,374 target, 9,948 ignore)
Columns: `period`, `fire_id`, `role`, `i_f`, `j_f` (sensor frame of that period).

| Role | Rule | Entries |
|---|---|---|
| target | `is_target`, own period | 3,374 |
| ignore | 10–50 ha (seen or not), own period | 8,455 |
| ignore | ≥ 50 ha not seen, own period | 1,218 |
| ignore (boundary, `BOUNDARY_H = 1`) | target with `start − period < 1 h` → also ignore in the previous period; `period + 12 h − start ≤ 1 h` → also ignore in the next period | 275 |
| background (implicit) | fires < 10 ha, fires started in an earlier period, unrecorded fires | – |

Targets by size: 1,228 / 1,483 / 326 / 337 (50–100 / 100–500 / 500–1k / ≥ 1k ha); 910 isolated. 3,018 targets have a known cause, 218 of them rekindles.

Known property of `seen` (2004–2019 build, not recomputed): for 528 targets (19 %), every detection that satisfies the test belongs to a nucleus born more than 1 h before the ICNF start, i.e. the fire is "seen" through an already burning neighbour (or shares its nucleus, cf. notes §3.2).

### 6.3 Rekindle records → `rekindles.parquet` (25,494 rows)
Every ICNF record with cause type "Reacendimento" (code 711) in the fire time window, of any size, with valid coordinates: `fire_id`, `start_utc`, true `lat`/`lon`, `burned_area_ha`, `period`, sensor-frame `i_f`/`j_f`. Not a label table: used only by the second evaluation protocol (§10.1) and the `rekindle_ignore` ablation (§6.6). Before 2008 most small fires have no cause, so the table is incomplete in those years.

### 6.4 Target heatmap `y` (`FireData.sample`)
- `y(i, j) = max_k exp(−((i − i_k)² + (j − j_k)²) / (2σ²))` over the targets of the period, σ = `SIGMA = 1.0` px (`--sigma`); sub-pixel centres; maximum, not sum.
- σ in pixels → anisotropic in km (~3.2 km E–W, ~4.5 km N–S).
- The cell nearest an ignition is ≤ 0.71 px away → y ≥ exp(−0.25) = 0.78.

### 6.5 Loss weight `w` (`FireData.sample`)
1. `w = pt_loss` (crop of it).
2. For every ignore entry of the period: `w = 0` where `d² ≤ IGNORE_R² = 4` px² (`--ignore-r`).
3. Only with `rekindle_ignore=True`: the same for every rekindle record of the period (§6.3).
4. For every target: `w = pt_loss` where `d² ≤ TARGET_CORE² = 1` px² (targets win; before this rule 295 / 2,705 targets had zero weight at their own pixel).

Figure: `figures/pipeline/sample_labels.png` (2016-08-06 06 UTC: 3 targets, 10 ignores, incl. an unseen 157 ha fire and a 2,577 ha boundary ignore).

### 6.6 Options of `sample()` used in ablations
- `min_target_ha`: targets below it become ignore (`target100`).
- `births=True`: 4th input channel (first step of each nucleus).
- `rekindle_ignore=True` (`--rekindle-ignore`, experiment `cnn3d_final_rk`): ignore regions at all rekindle records.
- `sigma`, `ignore_r` as above.

### 6.7 Space-time targets (option, not yet used by a trained model; notes §9)
`sample(..., time_bins=K, sigma_t=1.0)` returns `(y3, y2)` and `(w3, w2)`: y3 `(K, h, w)` = spatial Gaussian × Gaussian in time (σ_t hours) at the ICNF start, value at the bin centres of [0, 12) h; w3 = ignore discs in all bins, target core restored only in bins within 2 σ_t of the target's start; y2, w2 = §6.4–6.5. Positive voxels: `pos_threshold(σ, σ_t, bin_h)` = 0.667 for 1 px / 1 h / 1 h. Flips mirror only the spatial axes (`train.flip`).

## 7. Input tensors

### 7.1 `FireData.sample(period, crop=None)` → `x, y, w, lab`
- `s0 = first_slot(period)` = slot number (minutes / 15 since epoch) of period − 6 h; detections in `[s0, s0 + 96)` by `searchsorted` on the slot-sorted table; crop filter; write `x[slot − s0, c, i − i0, j − j0]`.
- Crop = `(i0, j0, size)`; `None` = full grid.

| Channel | Value | Range |
|---|---|---|
| 0 `frp` | `log1p(FRP) / log1p(1100)` (`FRP_NORM`) | 0.32 (8.3 MW) – 1.006 (1,147 MW; 21 rows above 1,100 MW, all 2022 and 2025) |
| 1 `confidence` | `confidence / 100` | 0.01–1 |
| 2 `detected` | 1 | {0, 1} |
| 3 `births` (optional) | 1 at a nucleus' first step | {0, 1} |

Empty cell/slot = 0 in all channels. Full domain: 96 × 3 × 177 × 106 float32 = 21.6 MB. `FRP_NORM` is kept so that the trained models see the same scaling.
Checks (`dataset.checks`, `outputs/dataset_check_log.txt`): 0 detection-count mismatches in 300 random periods; 0 targets without their peak pixel; 0 of 3,374 targets with zero weight; 200/200 jittered 32 × 32 crops contain their target.

### 7.2 `train.features(x, static, cfg)` → `(T', C', h, w)`
1. Window cut: slots `[(6 − history_h)·4, (18 + tail_h)·4)`.
2. Time aggregation: reshape to `(T/step_agg, step_agg, C, h, w)` and `.max(1)` per channel independently. Default `step_agg = 4` → 24 hourly steps (`detected` = any slot of the hour; FRP and confidence maxima may come from different slots). `step_agg = 1` → 96 steps. Assert: `step_agg` divides window, history and period.
3. Context channels: `period` = 1 for steps `[history/step, (history + 48)/step)` (hourly: steps 6–17), `time` = `t / (T' − 1)`, `pt`, `persistent` (broadcast over time).
4. Stack in canonical order `ALL_CHANNELS = (frp, confidence, detected, births, period, time, pt, persistent)` restricted to `cfg["channels"]`. Default `DEFAULT_CHANNELS` = 7 channels (no `births`).

Shapes: training `(B, 24, 7, 32, 32)`, inference `(B, 24, 7, 177, 106)`.
Figure: `figures/pipeline/time_aggregation.png` (18 detected slots → 6 hourly steps; a single-slot gap disappears).

## 8. Training (`train.CropSet`, `train.main`, `model.py`)

### 8.1 Crops
| Type | Count per epoch | Centre | Resampled |
|---|---|---|---|
| positive | one per target label of the training splits (`--train-splits`, default `train`: 1,979; `train,train2,train3`: 2,739; `min_target_ha` filters) | sensor-frame `(i_f, j_f)` + uniform integer jitter ±8 px per axis, clipped to the grid (`crop_origin`) | jitter each draw |
| hard negative | `round(neg_ratio × n_pos)`: 1,979 (default) / 3,958 (`neg2`, `cnn3d_final`) | random detection on a `pt` cell within the 96-slot window of a period with `n_targets == 0` and `n_det_pt > 0` and at least one detection on a `pt` cell (train: 2,784 periods; 4 periods of 2021–2022 fail the last condition, notes §7.4); sampled with replacement only if `n_neg >` periods; same ±8 px jitter | periods each epoch (`resample()`), detection each draw |

- Crop size `crop = 32` (≈ 100 × 145 km). Positive crops contain all labels of the period inside the crop (other targets, ignore discs).
- Augmentation (`flip=True`): horizontal flip p = 0.5 and, independently, vertical flip p = 0.5, applied to features, y, w. No rotations.
- `DataLoader(shuffle=True, batch_size=16)`; workers get independent RNG streams (`_worker_init`).
- Guards: training on `test` is impossible; training on `val` requires `--no-val`.

### 8.2 Loss (`--loss focal`, default)
- `p = sigmoid(logit)` clamped to [1e-4, 1 − 1e-4]; positives: `y ≥ pos_threshold(σ) = min(0.75, 0.97·exp(−0.5 / (2σ²)))` (= 0.75 for σ = 1).
- `L = [Σ w·pos·(−(1 − p)² log p) + Σ w·neg·(−(1 − y)⁴ p² log(1 − p))] / max(Σ w·pos, 1)` (CornerNet/CenterNet, α = 2, β = 4).
- Alternatives: `bce` (weighted BCE vs. soft targets, same normalisation), `mse` (weighted MSE on probabilities, normalised by Σw).
- Output bias initialised to prior 0.01 (`_init_prior`).

### 8.3 Optimisation and selection
AdamW (lr 1e-3, weight decay 1e-4), `CosineAnnealingLR(T_max = epochs)`, 20 epochs, gradient clipping 5, seeds 0–2. Validation every 2 epochs on the full grid (§9–10); checkpoint = best `(AP@5 + AP@10) / 2` → `models/<name>.pt` (`state_dict`, `config`, `epoch`, `val`); `<name>_history.csv` after every epoch; `<name>_done.json` at the end.

## 9. Inference and post-processing (`train.predict_maps`, `heatmap_to_points`, `maps_to_points`)

1. Periods: split periods with `n_det > 0` (val: 844). Periods without detections → no predictions.
2. Full-grid features, `val_batch = 16` periods per call (`--val-amp`: fp16 on CUDA), sigmoid → probability map (sensor frame).
3. Peaks: `p == max_pool2d(p, 3, stride 1, pad 1)` and `p ≥ 0.05`; top 50 per period by probability.
4. Sub-pixel: probability-weighted centroid of the 3 × 3 window (zero-padded at the edges); `subpixel=False` → cell centre.
5. Frame: `sensor_to_true_ij(fi, fj, period)` (+0.5 px S and E before 2017-12-01), then `ij_to_latlon`.
6. Output: `DataFrame(period, lat, lon, score = peak probability)`.

Space-time maps (notes §9.4, §9.9): `train.decode` with `decode="3d"` (peaks over 3 bins × 3 × 3 cells, centroid in the peak's bin, time by a log-parabola over three bins), `"max"` (maximum over bins, then the 2D decoder) or `"2d"` (auxiliary 2D map); points get `t_h` (hours since the period start).

Peak ablation (notes §6.9): threshold 0.02–0.1 no effect; sub-pixel +0.027 AP@5; 5 × 5 window worse. Log-parabola decoding and seed ensembles: `model_review.md` §3.1 (not yet in `src/`).

## 10. Evaluation (`src/evaluate.py`)

### 10.1 Matching (`match`, per period, k ∈ {5, 10} km)
1. Drop predictions whose rounded cell is outside `pt_loss`.
2. Sort by score (descending).
3. Per prediction: nearest unmatched target (true `lat`/`lon`) with haversine ≤ k → `tp` (store `err_km`); else any ignore fire of the period within k → `ignored`; else `fp` (incl. duplicates on a matched target).
4. Second protocol only (`neutral_rekindles=True`): a prediction that would be `fp` but lies within k of a rekindle record of its period (§6.3) → `ignored`. The default protocol is unchanged.

### 10.2 Metrics (`evaluate`)

| Metric | Definition |
|---|---|
| AP | over all non-ignored predictions of the split sorted by score: recall = cumulative TP / all targets of the split, precision = TP / (TP + FP), interpolated (running max from the right), AP = Σ Δrecall × precision |
| recall / precision (all preds) | last point of the curve |
| best F1 | max F1 over thresholds; P, R at that point; false alarms / day = FP with score ≥ threshold / (periods / 2) |
| median error TP | median `err_km` of TPs |
| nearest pred ≤ 5 km, median | distance target → nearest prediction of its period, no matching |
| recall by subgroup | `isolated`, `size_class`; with `report_attrs` also `rekindle` (True / False / unknown), `cause_type`, `fwi_quartile` (fixed bins 23.7 / 32.8 / 40.5, quartiles of the fires ≥ 50 ha 2004–2023) and `cover` (≥ 50 % forest / shrub / agricultural, else mixed) |

What the metrics condition on: only seen targets are in the denominator (unseen ≥ 50 ha are ignore); predictions at fires < 10 ha, ongoing fires and unrecorded fires are FP (in the second protocol not those next to a rekindle record); predictions at 10–50 ha / unseen / boundary fires are neutral.

Validation 2015–2016, `ablations_georef/cnn3d_final_s0-2` on this build: AP 0.635 / 0.704 @ 5 / 10 km (as stored in the checkpoints); second protocol 0.650 / 0.737.

Space-time protocol (notes §9.9): predictions with `t_h` get the time error of their hits (`dt_h`, median |Δt|, ≤ 1 h, ≤ 2 h, per subgroup); `evaluate(..., tau_h=τ)` also requires |Δt| ≤ τ for a hit. Extra subgroups `in_pair` (another target ≤ 2 cells, same period) and `already_burning` (detection ≤ 1.5 cells in [start − 6 h, start − 30 min]) from `evaluate.target_context`. Time baselines (`baselines.py`): nucleus birth time + `ACQ_OFFSET_H` (10 min; the SEVIRI time label is the slot start, Portugal is scanned ~10 min later), and `first_detection_times` for the points of a 2D model.

`src/evaluate_test.py`: several checkpoint folders in one run (`--folders`), output folder `--out` (never `final`, which holds notes §6.17), AP per year and with rekindles neutral, recall by all subgroups; the test split only with `--confirm-test`.

## 11. Reproduction

```
.venv/bin/python src/download_msg_frp.py --merge                                # seviri_frp_2020_2025.parquet from the day files
.venv/bin/python src/clean_data.py > outputs/clean_data_log.txt
.venv/bin/python src/cluster_hotspots.py > outputs/cluster_hotspots_log.txt
.venv/bin/python src/build_dataset.py > outputs/build_dataset_log.txt          # --no-georef-fix = uncorrected build
.venv/bin/python src/dataset.py > outputs/dataset_check_log.txt
.venv/bin/python src/check_new_years.py > outputs/new_years_check.txt           # data checks + regression vs commit 5a0f39d
.venv/bin/python src/baselines.py > outputs/baselines_log.txt
.venv/bin/python src/train.py --name ablations_georef/cnn3d_final_s0 --seed 0 --arch cnn3d --neg-ratio 2
.venv/bin/python src/plot_pipeline_figures.py
```

Regression (`outputs/new_years_check.txt`): for 2004–2019 the detections (12 columns), `static.npz`, fires, labels and period counts are identical to the build of commit `5a0f39d`; positions differ by at most 3 × 10⁻¹⁴ px.

## 12. Open issues (data side)

- Close ignitions: one 2D map per period; 13 % of val targets have another target ≤ 2 px (recall 0.44 @ 10 km; `model_review.md` §3.3). Pairs are median 4.2 km / 1.7 h apart.
- `seen` accepts any detection (19 % of targets seen only via older nuclei in the 2004–2019 build, §6.2).
- Size threshold: fires < 10 ha are background; ~30 % of FPs at best F1 lie next to one. 3,600 seen 10–50 ha fires are only ignored.
- Rekindles vs. ongoing fires: the SGIF cause gives a rekindle flag (§6.3), incomplete before 2008; memory limited to the 6 h history.
- Observability: no cloud mask; whole-domain empty slots between busy slots (105 in 2004–2017, 5 in 2018–2025) are hidden by the hourly max.
- Residual ~0.2 px N–S offset after the geo-referencing fix, larger at night (terrain parallax hypothesis, needs a DEM).
- ICNF position noise unquantified (shared coordinates, less frequent in later years).
- `persistent` computed on 2004–2019 (training, validation and `train2` years).
- The split is a default (notes §8.8).
- Row-level comparison of the SEVIRI extract with the LSA SAF archive on the overlap days (notes §7.6 step 0) not done: needs the LSA SAF login.
