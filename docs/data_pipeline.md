# Data pipeline — technical reference

> Purpose: one place that describes, step by step, how the raw SEVIRI and ICNF files become model inputs, labels, loss weights, predictions and scores — with the exact rules, constants, file names and row counts as implemented in `src/`. `notes.md` is the chronological methods log (why and when each decision was made); this file is the current state of the pipeline (what the code does now). Section references like "notes §6.12" point to `notes.md`.
>
> State: 2026-10-27, corrected build (`build_info.json`: `georef_fix = true`, change date 2017-12-01, shift 0.5 px). Reference training configuration = `train.DEFAULTS` (v1, hourly); the best model `cnn3d_final` uses the same data settings with `--arch cnn3d --neg-ratio 2`. Figures: `figures/pipeline/` (`src/plot_pipeline_figures.py`). A non-technical walkthrough with diagrams exists as a Claude doc ("Data pipeline: from raw SEVIRI/ICNF to model inputs, labels and scores").

## 0. Overview

```
data/raw/FRP_SEVIRI_PT_2004_2019.mat ──clean_data.py──► seviri_frp.parquet ──cluster_hotspots.py──► seviri_frp_events.parquet
data/raw/ignic_ICNF.xlsx             ──clean_data.py──► icnf_fires.parquet ─────────────────────────────┐   (+ seviri_events, seviri_nuclei)
                                                                                                         ▼
                                        build_dataset.py (+ grid.py, tiling_design.py) ──► data/processed/model/
                                            detections.parquet · fires.parquet · labels.parquet · periods.parquet · static.npz · build_info.json
                                                                                                         │
            ┌──────────────────────────────────────────────┬──────────────────────────────────────────────┘
            ▼                                              ▼
 dataset.py FireData.sample(period, crop) → x, y, w     baselines.py (nucleus births → points)
            ▼                                              │
 train.py features() → model → heatmap → peaks → points    │
            └──────────────────────► evaluate.py ◄─────────┘   (match to ICNF within 5 / 10 km → AP, recall, F1, error)
```

Invariants:
- ICNF never enters the model input (no crops, sequence starts or masks derived from ICNF). It only selects/labels samples and scores predictions (thesis.md §12).
- All times are UTC internally (`time_utc`, `start_utc`, `end_utc`).
- Spatial unit = one native SEVIRI pixel ("cell"); no resampling anywhere.
- Test split 2017–2019 is never evaluated during development.

## 1. Raw data

| | SEVIRI FRP extract | ICNF fire register |
|---|---|---|
| File | `data/raw/FRP_SEVIRI_PT_2004_2019.mat` (3.5 MB, `scipy.io.loadmat`, arrays `(1, N)` → `.ravel()`) | `data/raw/ignic_ICNF.xlsx` (51 MB, one sheet) |
| Row | one fire pixel at one 15-min scan | one reported fire |
| Rows | 569,019 | 813,384 |
| Period | 2004-01-19 15:30 → 2019-12-31 23:45 | 1980–2023; lat/lon only from 2001-01-05 |
| Columns | `TIME`, `LATITUDE`, `LONGITUDE` (2 decimals), `FRP` [MW], `FIRE_CONFIDENCE` (1–100) | `Data`, `DataFim`, `Dicofre`, `Distrito`, `Concelho`, `AA` [ha], `lat`, `lon` |
| Extent | lat 36.35–43.58, lon −9.39 to −6.00 (Portugal + parts of Spain) | mainland Portugal |
| Time zone | assumed UTC (consistent with all checks) | Europe/Lisbon local time (verified, §2.2) |

Properties that shape the pipeline:
- Only fire pixels are stored: absence = no detection, cloud, smoke or missing scan (indistinguishable, no cloud mask).
- FRP 8.8–1,048 MW, median 81 MW, saturation near 1,050 MW. Confidence rises with FRP (low-confidence rows ≈ 20–35 MW).
- ICNF start = reported alert time (uncertain by ~1 h); position = estimated ignition point (72 % of all points shared with other fires; 10 % for fires ≥ 100 ha); end times unreliable.

## 2. Cleaning (`src/clean_data.py`)

Outputs: `data/processed/icnf_fires.parquet` (806,926 rows), `data/processed/seviri_frp.parquet` (568,123 rows). Rows with unusable fields are flagged and kept; downstream code filters on the flags.

### 2.1 Rules

| Table | Step | Rule | Rows |
|---|---|---|---|
| ICNF | source | `.xlsx` (the `.csv` copy had timestamps shifted by −1 s in 1989–1997; deleted) | – |
| ICNF | columns | drop `Unnamed: 0`; rename → `start_local`, `end_local`, `municipality_code`, `district`, `municipality`, `burned_area_ha` | – |
| ICNF | text | NFKD → ASCII, upper case, whitespace collapsed (`normalise_text`) | districts 37 → 18 spellings |
| ICNF | missing start | drop | 6 |
| ICNF | duplicates | `drop_duplicates()` after text normalisation | 6,452 |
| ICNF | end time | NaT if missing, year < 1980, or `end < start`; flag `end_invalid` | 22,372 (none after 2000) |
| ICNF | coordinates | NaN if outside lat 36.8–42.2 / lon −9.6 to −6.1; flag `coord_invalid` | 32 |
| ICNF | UTC | `tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")` → `start_utc`, `end_utc` | 42 / 45 NaT (DST switch hours) |
| ICNF | order, id | sort by `start_local`, `municipality_code`, `burned_area_ha`; `fire_id` = 0…N−1; `duration_h` | – |
| SEVIRI | columns | `time_utc` (labelled UTC), `lat`/`lon` rounded to 2 decimals, `frp_mw` float32, `confidence` int8 | – |
| SEVIRI | duplicates | drop exact duplicates; assert no conflicting (time, lat, lon) duplicates | 896 |
| SEVIRI | order | sort by `time_utc`, `lat`, `lon` | – |

Not done on purpose: no confidence threshold, no FRP cap, Spain kept, zero-area ICNF fires kept.

### 2.2 Time zone (`src/timezone_check.py`, notes §1.4)
- Lag first SEVIRI detection (≤ 5 km, [start − 6 h, start + 24 h]) − recorded start, for 3,602 fires ≥ 10 ha without earlier activity nearby: summer (DST) −0.70 h, winter +0.17 h → jump ≈ 0.9 h at the DST switch, also within the same months. Conclusion: ICNF = local time. After conversion the median lag is +10–20 min.

### 2.3 Fire time window
All downstream steps use ICNF fires with `start_utc` in [2004-01-20, 2019-12-30] (`eda_figures.PERIOD`), i.e. fully inside the SEVIRI record: 361,031 fires with valid coordinates and UTC start.

## 3. Spatial frame (`src/grid.py`)

### 3.1 Native grid
- Projection: `+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y +units=m`; sampling distance at nadir `SSD = 3000.403165817 m`.
- `native_rc(lat, lon)`: `row = y / SSD`, `col = x / SSD` (continuous). Only `lon_0 = 0` fits (93–96 % of pixels within ±0.25 px of a node in every year vs. ~25 % by chance).
- All 8,402 distinct pixels → 8,402 distinct native pixels; max residual 0.42 px (2-decimal rounding).
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
- Frames used:

| Object | Frame | Where |
|---|---|---|
| detections, model input | sensor (as delivered) | `detections.parquet` |
| labels (`i_f`, `j_f`), "seen" test | sensor, shift evaluated at the label's **period** start (seen test: at the fire start) | `labels.parquet`, `build_dataset.fires/labels` |
| ICNF positions for scoring | true | `fires.parquet` (`lat`, `lon`, `i_f`, `j_f`) |
| predictions, nucleus births | sensor → true before scoring | `train.maps_to_points`, `baselines.predictions` |

- Effect: targets 2,705 → 2,821 (+116 fires whose first detection had been pushed outside the 5 km seen radius); median label − nucleus birth before the change −0.01 px E, −0.22 px N (after: 0.00 / −0.16) → offset removed. Remaining ~0.2 px N–S offset in both periods (possible terrain parallax, `model_review.md` §6.1).
- Figure: `figures/pipeline/georef_frames.png`.

## 4. Hotspot clustering (`src/cluster_hotspots.py`, notes §3)

SEVIRI only; ICNF used for diagnostics only. Since notes §4.0 the clustering is **not** in front of the model; it provides the baseline, flags and the optional `births` channel.

### 4.1 Events
- Link rule: pixel centres ≤ `D_KM = 6` km (local metric coordinates `xy_km`) and |Δt| ≤ `GAP_H = 6` h, same time step included. Events = connected components (union–find over time-sorted detections; `cKDTree` neighbour lists per pixel; per pixel only the last detection is kept as link partner).
- 6 km = 8-neighbourhood (nearest neighbour ~3.2 km, 8th ~6.3 km). 6 h from the sensitivity grid (notes §3.1): merged fires ≥ 100 ha 30.8 %, isolated fires ≥ 90 % intact 88.7 %.
- 24,690 events; 6,620 with ≥ 10 detections hold 91.4 % of detections.

### 4.2 Nuclei
- Per time step: a detection inherits the nucleus of the **nearest** linked earlier detection; else the nucleus of the nearest already-assigned neighbour in the same step (iterated); else it is a **birth**, and connected new detections of the step form one new nucleus.
- Per nucleus (`seviri_nuclei.parquet`): `t_birth`, birth position = FRP-weighted centroid of the birth step, `n_det`, `fre_tj` (Σ FRP × 900 s), `order_in_event`, `dist_active_km` (nearest detection of another nucleus in the previous 6 h, ≤ 100 km), `reappeared` (detections ≤ 6 km in [birth − 48 h, birth − 6 h)).
- 25,954 nuclei; 32 % of those with ≥ 3 detections are reappearances.

### 4.3 Flags carried into `detections.parquet`

| Column | Definition | Used by |
|---|---|---|
| `persistent` | pixel with ≥ 20 active days and ≥ 50 % of them outside Jun–Oct (`eda_figures.pixel_stats`, all of 2004–2019) | `static["persistent"]` → input channel; baseline variants |
| `in_pt` | detection inside the mainland polygon | `periods.n_det_pt`, hard-negative periods |
| `event_id`, `nucleus_id` | §4.1–4.2 | `births` channel (ablation), baseline |

Caveat: `persistent` uses all years incl. validation/test (SEVIRI-only, no labels, but not strictly causal).

### 4.4 Baseline (`src/baselines.py`)
Every nucleus born inside a target period = one predicted ignition at its birth position (moved to the true frame before scoring). Score = FRE of the nucleus inside the period's input window up to the end of the 6 h tail. Variants: without reappeared; without reappeared/persistent; first-hour centroid.

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
| `SPLIT` | train 2004–2014, val 2015–2016, test 2017–2019 | by calendar year of the period |

### 5.2 Periods
- `period_of(t)` = `floor(t − 6 h, 12 h) + 6 h`: every fire belongs to the period containing its UTC start.
- 11,648 periods (2004-01-20 06 UTC → 2019-12-30 06 UTC). Per period: `n_det` (detections in [period − 6 h, period + 18 h)), `n_det_pt` (same, mainland only), `n_targets`, `n_ignore`.
- Windows of consecutive periods overlap by 12 h; every slot appears in two samples.

| Split | Periods | with detections | with ≥ 1 target | hard-negative (detections in PT, no target) | Targets |
|---|---|---|---|---|---|
| train | 8,000 | 4,469 | 805 | 2,784 | 1,979 |
| val | 1,462 | 844 | 163 | 540 | 403 |
| test | 2,186 | 1,265 | 215 | 894 | 439 |

Targets per positive period: 1 → 642, 2–5 → 442, 6–20 → 94, > 20 → 5 (max 32, 2017-10-15 06 UTC). 76 % of targets start in a 06–18 UTC period; median 7.1 h after the period start.

## 6. Labels and targets (`build_dataset.fires()`, `labels()`, `tiling_design.seen()`)

### 6.1 Candidate fires → `fires.parquet` (10,911 rows)
Filter: valid `lat`/`lon`, valid `start_utc` in the fire time window (§2.3), `burned_area_ha ≥ IGNORE_HA = 10`. The 350,120 smaller fires are not in any label table (= background).

Columns: `fire_id`, `start_utc`, `end_utc`, true `lat`/`lon`, true `i_f`/`j_f`, `burned_area_ha`, `size_class` (10–50, 50–100, 100–500, 500–1k, ≥ 1k ha; `right=False`), `municipality`, `district`, `seen`, `is_target`, `isolated`, `period`, `year`, `split`.

- **`seen`**: ∃ SEVIRI detection with haversine distance ≤ 5 km from the ICNF point (moved into the sensor frame at the fire start) and `time_utc` ∈ [start − 1 h, start + 6 h). Any detection counts (Spain, persistent pixels, other fires).
- **`is_target`** = `burned_area_ha ≥ 50` and `seen`.
- **`isolated`**: no other ICNF fire ≥ 10 ha within 20 km and ±72 h (`proximity_analysis.count_neighbours`). Reporting subgroup only, never an input.

| Size class | Fires | Seen | Seen share | Role |
|---|---|---|---|---|
| 10–50 ha | 7,042 | 3,031 | 43 % | ignore (seen or not) |
| 50–100 ha | 1,558 | 1,022 | 66 % | seen → target, unseen → ignore |
| 100–500 ha | 1,681 | 1,261 | 75 % | idem |
| 500–1,000 ha | 338 | 282 | 83 % | idem |
| ≥ 1,000 ha | 292 | 256 | 88 % | idem |

### 6.2 Roles → `labels.parquet` (11,145 rows: 2,821 target, 8,324 ignore)
Columns: `period`, `fire_id`, `role`, `i_f`, `j_f` (sensor frame of that period).

| Role | Rule | Entries |
|---|---|---|
| target | `is_target`, own period | 2,821 |
| ignore | 10–50 ha (seen or not), own period | 7,042 |
| ignore | ≥ 50 ha not seen, own period | 1,048 |
| ignore (boundary, `BOUNDARY_H = 1`) | target with `start − period < 1 h` → also ignore in the previous period; `period + 12 h − start ≤ 1 h` → also ignore in the next period | 234 |
| background (implicit) | fires < 10 ha, fires started in an earlier period, unrecorded fires | – |

Targets by size: 1,022 / 1,261 / 282 / 256 (50–100 / 100–500 / 500–1k / ≥ 1k ha); 679 isolated.

Known property of `seen`: for 528 targets (19 %), every detection that satisfies the test belongs to a nucleus born more than 1 h before the ICNF start, i.e. the fire is "seen" through an already burning neighbour (or shares its nucleus, cf. notes §3.2).

### 6.3 Target heatmap `y` (`FireData.sample`)
- `y(i, j) = max_k exp(−((i − i_k)² + (j − j_k)²) / (2σ²))` over the targets of the period, σ = `SIGMA = 1.0` px (`--sigma`); sub-pixel centres; maximum, not sum.
- σ in pixels → anisotropic in km (~3.2 km E–W, ~4.5 km N–S).
- The cell nearest an ignition is ≤ 0.71 px away → y ≥ exp(−0.25) = 0.78.

### 6.4 Loss weight `w` (`FireData.sample`)
1. `w = pt_loss` (crop of it).
2. For every ignore entry of the period: `w = 0` where `d² ≤ IGNORE_R² = 4` px² (`--ignore-r`).
3. For every target: `w = pt_loss` where `d² ≤ TARGET_CORE² = 1` px² (targets win; before this rule 295 / 2,705 targets had zero weight at their own pixel).

Figure: `figures/pipeline/sample_labels.png` (2016-08-06 06 UTC: 3 targets, 10 ignores, incl. an unseen 157 ha fire and a 2,577 ha boundary ignore).

### 6.5 Options of `sample()` used in ablations
- `min_target_ha`: targets below it become ignore (`target100`).
- `births=True`: 4th input channel (first step of each nucleus).
- `sigma`, `ignore_r` as above.

## 7. Input tensors

### 7.1 `FireData.sample(period, crop=None)` → `x, y, w, lab`
- `s0 = first_slot(period)` = slot number (minutes / 15 since epoch) of period − 6 h; detections in `[s0, s0 + 96)` by `searchsorted` on the slot-sorted table; crop filter; write `x[slot − s0, c, i − i0, j − j0]`.
- Crop = `(i0, j0, size)`; `None` = full grid.

| Channel | Value | Range |
|---|---|---|
| 0 `frp` | `log1p(FRP) / log1p(1100)` (`FRP_NORM`) | 0.33 (8.8 MW) – 0.99 (1,048 MW) |
| 1 `confidence` | `confidence / 100` | 0.01–1 |
| 2 `detected` | 1 | {0, 1} |
| 3 `births` (optional) | 1 at a nucleus' first step | {0, 1} |

Empty cell/slot = 0 in all channels. Full domain: 96 × 3 × 177 × 106 float32 = 21.6 MB.
Checks (`dataset.checks`, `outputs/dataset_check_log.txt`): 0 detection-count mismatches in 300 random periods; 0 targets without their peak pixel; 0 targets with zero weight; 200/200 jittered 32 × 32 crops contain their target.

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
| positive | one per target label of the split (train: 1,979; `min_target_ha` filters) | sensor-frame `(i_f, j_f)` + uniform integer jitter ±8 px per axis, clipped to the grid (`crop_origin`) | jitter each draw |
| hard negative | `round(neg_ratio × n_pos)`: 1,979 (default) / 3,958 (`neg2`, `cnn3d_final`) | random detection inside `pt` within the 96-slot window of a period with `n_targets == 0` and `n_det_pt > 0` (train: 2,784 periods; sampled with replacement only if `n_neg >` periods), same ±8 px jitter | periods each epoch (`resample()`), detection each draw |

- Crop size `crop = 32` (≈ 100 × 145 km). Positive crops contain all labels of the period inside the crop (other targets, ignore discs).
- Augmentation (`flip=True`): horizontal flip p = 0.5 and, independently, vertical flip p = 0.5, applied to features, y, w. No rotations.
- `DataLoader(shuffle=True, batch_size=16)`; workers get independent RNG streams (`_worker_init`).

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

Peak ablation (notes §6.9): threshold 0.02–0.1 no effect; sub-pixel +0.027 AP@5; 5 × 5 window worse. Log-parabola decoding and seed ensembles: `model_review.md` §3.1 (not yet in `src/`).

## 10. Evaluation (`src/evaluate.py`)

### 10.1 Matching (`match`, per period, k ∈ {5, 10} km)
1. Drop predictions whose rounded cell is outside `pt_loss`.
2. Sort by score (descending).
3. Per prediction: nearest unmatched target (true `lat`/`lon`) with haversine ≤ k → `tp` (store `err_km`); else any ignore fire of the period within k → `ignored`; else `fp` (incl. duplicates on a matched target).

### 10.2 Metrics (`evaluate`)

| Metric | Definition |
|---|---|
| AP | over all non-ignored predictions of the split sorted by score: recall = cumulative TP / all targets of the split, precision = TP / (TP + FP), interpolated (running max from the right), AP = Σ Δrecall × precision |
| recall / precision (all preds) | last point of the curve |
| best F1 | max F1 over thresholds; P, R at that point; false alarms / day = FP with score ≥ threshold / (periods / 2) |
| median error TP | median `err_km` of TPs |
| nearest pred ≤ 5 km, median | distance target → nearest prediction of its period, no matching |
| recall by subgroup | `isolated`, `size_class` |

What the metrics condition on: only seen targets are in the denominator (unseen ≥ 50 ha are ignore); predictions at fires < 10 ha, ongoing fires and unrecorded fires are FP; predictions at 10–50 ha / unseen / boundary fires are neutral.

## 11. Reproduction (corrected build)

```
.venv/bin/python src/clean_data.py
.venv/bin/python src/cluster_hotspots.py > outputs/cluster_hotspots_log.txt
.venv/bin/python src/build_dataset.py > outputs/build_dataset_log.txt          # --no-georef-fix = old build
.venv/bin/python src/dataset.py > outputs/dataset_check_log.txt
.venv/bin/python src/baselines.py > outputs/baselines_log.txt
.venv/bin/python src/train.py --name ablations_georef/cnn3d_final_s0 --seed 0 --arch cnn3d --neg-ratio 2
.venv/bin/python src/plot_pipeline_figures.py
```

## 12. Open issues (data side)

- Close ignitions: one 2D map per period; 13 % of val targets have another target ≤ 2 px (recall 0.44 @ 10 km; `model_review.md` §3.3). Pairs are median 4.2 km / 1.7 h apart.
- `seen` accepts any detection (19 % of targets seen only via older nuclei, §6.2).
- Size threshold: fires < 10 ha are background; ~30 % of FPs at best F1 lie next to one. 3,031 seen 10–50 ha fires are only ignored.
- Rekindles vs. ongoing fires: no ICNF flag; memory limited to the 6 h history.
- Observability: no cloud mask; whole-domain empty slots between busy slots (105 on busy summer days) are hidden by the hourly max.
- Residual ~0.2 px N–S offset after the geo-referencing fix (terrain parallax hypothesis, needs a DEM).
- ICNF position noise unquantified (shared coordinates).
- `persistent` computed over all years incl. test.
