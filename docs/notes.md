# Master Thesis - Notes/Actions

This document serves as a basis for the write up. All notes (more specifically actions taken) will be noted here. Each step and decision will be documented to make sure the final work is complete and reproducable.

## 0. Repository layout and how to reproduce

```
data/raw/          ignic_ICNF.xlsx, FRP_SEVIRI_PT_2004_2019.mat   (as received, never modified)
data/processed/    icnf_fires.parquet, seviri_frp.parquet          (src/clean_data.py)
                   seviri_frp_events.parquet, seviri_events.parquet, seviri_nuclei.parquet (src/cluster_hotspots.py)
data/processed/model/ sample index for the heatmap model: detections, fires, labels, periods .parquet, static.npz (src/build_dataset.py)
data/external/     ne_10m_admin_0_countries.zip (country borders, see 2.2)
outputs/profiling/ ydata-profiling reports in raw/ and processed/  (src/eda.py)
outputs/           *_log.txt, cluster_hotspots_sensitivity.txt     (numbers printed by the scripts)
figures/           plots                                           (src/timezone_check.py, src/plot_largest_fires.py)
figures/eda/       descriptive figures 01–11                       (src/eda_figures.py, src/proximity_analysis.py)
figures/clustering/ hotspot events and nuclei, case study 2017-10-15 (src/cluster_hotspots.py)
figures/tiling/    example training sample                         (src/tiling_design.py)
figures/dataset/   full-domain example sample                      (src/dataset.py)
figures/baselines/ precision-recall curves of the baselines        (src/baselines.py)
figures/model/    model vs. baseline precision-recall curves      (src/evaluate_model.py)
models/           trained checkpoints (*.pt)                      (src/train.py)
src/               all code
docs/              thesis.md (context), notes.md (this file), literature.md (literature review), data_pipeline.md (current pipeline, technical reference), model_review.md
```

Environment: Python 3.11, `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (only packages imported by `src/`, pinned).

Run from the repository root, in this order:
1. `.venv/bin/python src/clean_data.py`: raw → processed parquet (1.3, 1.5)
2. `.venv/bin/python src/eda.py`: profiling reports (1.6)
3. `.venv/bin/python src/timezone_check.py`: time zone check (1.4)
4. `.venv/bin/python src/plot_largest_fires.py`: largest fires in lat–lon–time (2.1)
5. `.venv/bin/python src/eda_figures.py > outputs/eda_figures_log.txt`: descriptive figures (2.2)
6. `.venv/bin/python src/proximity_analysis.py > outputs/proximity_analysis_log.txt`: neighbouring fires and shared hotspots (2.3)
7. `.venv/bin/python src/cluster_hotspots.py > outputs/cluster_hotspots_log.txt`: hotspot events and nuclei (3.1, 3.2); add `--sensitivity` for the parameter grid (`outputs/cluster_hotspots_sensitivity.txt`)
8. `.venv/bin/python src/tiling_design.py > outputs/tiling_design_log.txt`: numbers for the sample design (4.1, 4.2)
9. `.venv/bin/python src/build_dataset.py > outputs/build_dataset_log.txt`: sample index for the model (5.1)
10. `.venv/bin/python src/dataset.py > outputs/dataset_check_log.txt`: checks of the sample builder + example figure (5.1)
11. `.venv/bin/python src/baselines.py > outputs/baselines_log.txt`: baselines on train/validation (5.2, 5.3)
12. `.venv/bin/python src/train.py --name convlstm_v1 > outputs/convlstm_v1_log.txt`: train the ConvLSTM (6.1; ~45 min on an M1 Pro GPU)
13. `.venv/bin/python src/evaluate_model.py --name convlstm_v1 > outputs/convlstm_v1_eval.txt`: model vs. baseline on validation (6.2)
14. `.venv/bin/python src/plot_model_diagram.py`: architecture diagram (6.1) and ConvLSTM detail diagram (6.1.2), concept diagram of the long-term memory (6.6)
15. `.venv/bin/python src/plot_predictions.py --name convlstm_v1 > outputs/convlstm_v1_predictions_log.txt`: predicted vs. ICNF ignition maps (6.3)
16. `.venv/bin/python src/train.py --name convlstm_v2_15min --step-agg 1 > outputs/convlstm_v2_15min_log.txt`, then `src/evaluate_model.py --name convlstm_v2_15min` and `src/plot_predictions.py --name convlstm_v2_15min`: 15-min model (6.5)
17. `python src/run_ablations.py --tier 1 --seeds 0 1 2` (or `colab/ablations_colab.ipynb` on a Colab GPU), then `python src/summarize_ablations.py`: ablation study (6.7); `python src/peak_ablation.py --name convlstm_v1`: peak-extraction ablation
18. `.venv/bin/python src/plot_pipeline_figures.py`: figures explaining labels, loss weights, time aggregation and the geo-referencing frames → `figures/pipeline/` (data-pipeline explainer doc, 2026-10-26)
19. `.venv/bin/python src/plot_overview_diagrams.py`: overview figures of the data pipeline (`figures/pipeline/pipeline_overview.png/.pdf`) and one per model (`figures/model/model_convlstm.png/.pdf`, `model_cnn3d.png/.pdf`; use the checkpoints `ablations_georef/conv_final_s0` and `cnn3d_final_s0`)

## 1. Data

### 1.1 ICNF

- Portugal's official fire database, supplied by the supervisor as `ignic_ICNF.xlsx`
- One row per fire: start (`Data`) and end (`DataFim`) date/time, municipality code (`Dicofre`), district (`Distrito`), municipality (`Concelho`), burned area in ha (`AA`), and from 2001 on the estimated ignition point (`lat`, `lon`)
- 813,384 fires, 1980–2023
- The first EDA (ydata-profiling of the raw ".xlsx") has been replaced by the raw/processed reports in 1.6.

### 1.2 SEVIRI FRP

- Fire Radiative Power hotspots from SEVIRI on Meteosat Second Generation (geostationary, one scan every 15 min), supplied by the supervisor as `FRP_SEVIRI_PT_2004_2019.mat`
- One row per detected fire pixel and time step: `TIME`, `LATITUDE`, `LONGITUDE`, `FRP` (MW), `FIRE_CONFIDENCE` (1–100)
- 569,019 detections, 2004-01-19 → 2019-12-31; covers Portugal and neighbouring parts of Spain
- Only fire pixels are stored: a missing pixel means "no detection", which cannot be distinguished from cloud cover / no valid retrieval

### 1.3 Cleaning (2026-10-07)

Script: `src/clean_data.py` (run from the repository root with `.venv/bin/python src/clean_data.py`). Raw files in `data/raw/` are not modified; outputs go to `data/processed/`. `pyarrow` was installed into `.venv` to write parquet.

**ICNF → `data/processed/icnf_fires.parquet`** (813,384 → 806,926 rows)

| Step | Rule | Rows affected |
|---|---|---|
| Source | Read from ".xlsx" (see 1.1) | – |
| Index column | Dropped `Unnamed: 0` (leftover index, not unique: file is a concatenation of several tables) | – |
| Renaming | `Data`→`start_local`, `DataFim`→`end_local`, `Dicofre`→`municipality_code`, `Distrito`→`district`, `Concelho`→`municipality`, `AA`→`burned_area_ha` | – |
| Text | `district`/`municipality` upper case, accents removed, whitespace trimmed | district: 37 spellings → 18 districts; municipality: 278 → 278 |
| Missing start | Rows without `start_local` dropped (all 1993) | 6 dropped |
| Duplicates | Exact duplicate rows dropped (after text normalisation), first occurrence kept | 6,452 dropped |
| Invalid end time | `end_local` set to NaT, flagged in `end_invalid` | 22,372 rows (see below) |
| Invalid coordinates | `lat`/`lon` set to NaN when outside mainland Portugal bbox (lat 36.8–42.2, lon −9.6 – −6.1), flagged in `coord_invalid` | 32 rows |
| Order / id | Sorted by start time; `fire_id` (0…N−1) added | – |

Rows are kept even when end time or coordinates are invalid (the burned area and start time remain usable); downstream steps filter on the flags.

Invalid end times (categories overlap):
- 1,519 missing
- 19,376 rows from 1994 with `DataFim` = 1970-01-01 + time of day: the date was lost (Unix-epoch default) and only the hour survived. This causes almost all of the "end before start" cases. The time of day could in principle be recovered by assuming the fire ended on the start day, but that is not needed since 1994 lies outside the SEVIRI period.
- 1 sentinel 1900-01-01, 1 date in 1935
- 20,853 rows in total with end before start (none after 2000)

Invalid coordinates (32 rows, all 2009–2021):
- 23 rows at (36.944, −10.378) in the Atlantic, apparently a default value used in 2014–2015 for fires in Porto, Lisboa and Setúbal
- 1 at (0, 0), 1 with lon = 0, 1 at lat ≈ 48, 1 at lat ≈ 28.9, 1 at lon ≈ −17.2, and points in Spain or offshore (e.g. Celorico de Basto at lat 42.45, Mafra at lon −9.62)
- Limitation: the bounding box does not catch wrong points that still fall inside it (e.g. in Spain near the border or in the wrong municipality). A check of each point against its municipality polygon would need geopandas and administrative boundaries; to do later if needed.

Not changed (decisions to revisit):
- **Time zone:** `start_local`/`end_local` are kept as recorded. Portuguese local time was confirmed in 1.4, and UTC columns were added in 1.5.
- **Burned area:** 21,339 rows with `burned_area_ha` = 0 and 1 missing value are kept (they are irrelevant once filtering by size).
- `municipality_code` (`Dicofre`) has 278 values and maps 1-to-1 to `municipality`, so it is a municipality code, not a parish code.

Result: 477,674 rows with valid coordinates; 361,091 of them in 2004–2019 (≥100 ha: 2,311; ≥1,000 ha: 292). Durations (valid): median 1.17 h, p90 3.93 h, p99 17.75 h.

**SEVIRI → `data/processed/seviri_frp.parquet`** (569,019 → 568,123 rows)

| Step | Rule | Rows affected |
|---|---|---|
| Loading | `scipy.io.loadmat`, arrays flattened | – |
| Columns | `time_utc`, `lat`, `lon`, `frp_mw` (float32), `confidence` (int8) | – |
| Time | Parsed and labelled UTC (standard for SEVIRI products; not stated in the file) | – |
| Coordinates | Rounded to 2 decimals (the native precision; removes float noise such as 43.579998) | – |
| Duplicates | Exact duplicate rows dropped; all duplicates were identical copies (same FRP and confidence), mostly in groups of 3 | 896 dropped |
| Order | Sorted by time, lat, lon | – |

Not changed (decisions to revisit):
- No confidence threshold: low-confidence detections are likely artefacts, but they are also the weak early-stage detections relevant for ignition. To decide during event matching.
- Detections in Spain are kept (no ICNF labels there; mask or use as context later).
- FRP saturation around ~1,050 MW left as is.

### 1.4 Time zone check (2026-10-07)

Script: `src/timezone_check.py`; figure: `figures/timezone_lag.png`.

Question: are ICNF start times in UTC or Portuguese local time (UTC+0 in winter, UTC+1 in summer)? SEVIRI is assumed UTC.

Method:
- Fires from 2004–2019 with valid coordinates and ≥ 10 ha (10,911 fires).
- For each fire: first SEVIRI detection within 5 km of the ignition point in [start − 6 h, start + 24 h], with the recorded start time treated as UTC. Lag = first detection − start.
- To avoid picking up a different, earlier fire, fires with any detection within 10 km in the 48 h before the window were excluded. 3,602 fires remained with a detection.
- Each fire was labelled summer/winter by whether Lisbon was on daylight saving time at the recorded start. If ICNF is local time, the naive lag should be about 1 h smaller in summer; if it is UTC, there should be no difference.

Results (median lag, 95 % bootstrap CI):

| Subset | Summer (DST) | Winter | Winter − summer |
|---|---|---|---|
| ≥ 10 ha | −0.70 h (−0.72 .. −0.67), n = 3,253 | +0.17 h (+0.08 .. +0.33), n = 349 | +0.87 h |
| ≥ 100 ha | −0.67 h, n = 1,072 | +0.22 h, n = 45 | +0.88 h |
| Mar/Apr + Oct/Nov only (same season, either side of the switch) | −0.67 h, n = 507 | +0.23 h, n = 231 | +0.90 h |

Conclusion:
- **ICNF times are Portuguese local time (Europe/Lisbon, with daylight saving).** The lag jumps by about 1 h at the DST switch, also when only comparing fires from the same months, so it is not a seasonal effect.
- Before the correction, summer fires appear to be detected ~40 min *before* they started. This was the cause of the "first detection 1 h before start" result in thesis.md §8.
- After converting to UTC, the median lag is positive in both seasons (summer ≈ +0.3 h, winter ≈ +0.2 h): SEVIRI first sees a fire roughly 10–20 min after the reported start, as expected.
- Decision: ICNF times will be converted with `tz_localize("Europe/Lisbon").tz_convert("UTC")` and all matching done in UTC. Wall-clock times that do not exist or are ambiguous at the DST switches (one hour per year each) become NaT.

### 1.5 UTC columns added to ICNF (2026-10-07)

- `src/clean_data.py` now adds `start_utc` and `end_utc` (tz-aware UTC) to `icnf_fires.parquet`, converted from `start_local`/`end_local` as Europe/Lisbon time. The `_local` columns are kept as recorded.
- The time zone database also covers Portugal's 1992–1996 period on Central European Time (UTC+1/+2), so older records are converted correctly as well.
- Times in a DST switch hour cannot be converted unambiguously and are NaT: 42 `start_utc` (17 of them in 2004–2019 with coordinates) and 45 `end_utc`.
- **From here on all ICNF–SEVIRI matching uses `start_utc`/`end_utc`.**

### 1.6 Profiling reports, raw vs. processed (2026-10-07)

Script: `src/eda.py` (ydata-profiling, `explorative=True`). Reports in `outputs/profiling/raw/` and `outputs/profiling/processed/`, each containing `icnf_eda_report.html` and `seviri_eda_report.html`.

- Raw = the files as received. The only change for SEVIRI is flattening the `.mat` arrays into a table and parsing `TIME` as a date so it is profiled as such.
- Processed = the parquet files from `src/clean_data.py`.

| Report | Variables | Rows | Missing cells | Duplicate rows (as reported) |
|---|---|---|---|---|
| ICNF raw | 9 | 813,384 | 662,484 (9.0 %) | 0 |
| ICNF processed | 14 | 806,926 | 725,708 (6.4 %) | 0 |
| SEVIRI raw | 5 | 569,019 | 0 | 421 |
| SEVIRI processed | 5 | 568,123 | 0 | 0 |

How to read these numbers:
- ICNF raw "0 duplicates" is misleading: the leftover index column `Unnamed: 0` makes every row unique. Without it there are 6,452 duplicates, which cleaning removed.
- SEVIRI raw "421 duplicate rows" counts duplicated *groups*. They contain 896 extra copies (mostly 3 copies of the same row), and all 896 were removed.
- Missing cells go *up* in absolute terms for ICNF processed because new columns were added (`end_utc`, `start_utc`, `duration_h`) and invalid values were set to missing on purpose (end times, coordinates). The missing coordinates before 2001 (~330k rows × 2) make up most of the total in both reports.

## 2. Exploration

### 2.1 Largest fires in lat–lon–time (2026-10-07)

Script: `src/plot_largest_fires.py`; figure: `figures/largest_fires_lat_lon_time.png`.

- The 12 largest ICNF fires 2004–2019 (valid coordinates, UTC start). For each: all SEVIRI detections within 40 km of the ignition point from 6 h before the ICNF start to 6 h after the ICNF end (at least 48 h after the start). Neighbouring fires were not removed.
- Two panels per fire: a map (colour = hours since start, size = FRP, star = ICNF ignition point) and distance to ignition vs. time.

| Fire | Area (ha) | Detections ≤ 40 km | ≤ 10 km | First detection ≤ 10 km |
|---|---|---|---|---|
| 2017-10-15 Lousã | 53,619 | 846 | 83 | +0.8 h, 4.2 km |
| 2017-10-15 Arganil | 37,959 | 634 | 44 | +1.0 h, 4.9 km |
| 2017-07-23 Sertã | 33,640 | 4,986 | 872 | 0.0 h, 5.7 km |
| 2017-10-15 Sertã | 33,193 | 714 | 219 | 0.0 h, 4.8 km |
| 2017-06-17 Pedrógão Grande | 30,359 | 2,079 | 123 | **+16.3 h**, 8.4 km |
| 2018-08-03 Monchique | 26,764 | 4,760 | 2,050 | 0.0 h, 2.0 km |
| 2012-07-18 Tavira | 24,843 | 3,170 | 1,929 | +0.3 h, 3.8 km |
| 2017-08-11 Alvaiázere | 22,824 | 953 | 260 | +0.1 h, 2.6 km |
| 2016-08-08 Arouca | 21,909 | 11,937 | 3,910 | −0.1 h, 5.8 km |
| 2017-10-15 Figueira da Foz | 19,026 | 212 | 21 | +0.7 h, 2.0 km |
| 2017-06-17 Góis | 17,521 | 2,092 | 487 | **+19.6 h**, 5.3 km |
| 2017-10-15 Alcobaça | 17,265 | 71 | 21 | +0.5 h, 2.3 km |

Observations:
- **Most fires are seen within ~1 h of the ICNF start**, with the first nearby hotspot 2–6 km from the ICNF ignition point (about 1–2 SEVIRI pixels). The hotspot cloud then spreads away from the start point over time (rising distance in the right panels), which is the propagation signal the models should learn from.
- **Pedrógão Grande and Góis (2017-06-17) are not detected for 16–20 h**, although Pedrógão was one of the most intense fires on record. Over the whole SEVIRI domain there are only 3–15 detections per hour that afternoon and none between 18:00 and 22:00 UTC. Likely cause: the fire's own smoke / pyrocumulus cloud hid it from the satellite, or the retrieval was rejected. Absent hotspots therefore do not always mean "no fire", and some large fires have no usable early stage.
- **Overlapping fires are a major problem.** On 2017-10-15 several fires >15,000 ha burned at the same time (Lousã, Arganil, Sertã, Figueira da Foz, Alcobaça, …), so their 40 km windows contain each other's hotspots. Arouca 2016 sits in a region with many simultaneous fires. A matching rule must assign each hotspot to one fire (e.g. by spatio-temporal clustering) before any event can be used as a sample.
- ICNF end times are often well before the last hotspots (e.g. Sertã 2017-07-23, Tavira) or long after them (Lousã, Arganil). End times are not reliable for cutting sequences; the hotspots themselves should define the end of an event.
- The map view shows SEVIRI's coarse, irregular pixel grid (~3–5 km), confirming that sub-pixel ignition localisation will be the hard part.

### 2.2 Descriptive figures (2026-10-07)

Script: `src/eda_figures.py`; figures: `figures/eda/01–09`; printed numbers: `outputs/eda_figures_log.txt`.

Setup:
- `geopandas` installed into `.venv` (added to `requirements.txt`).
- Country borders: Natural Earth 1:10m Admin 0 – Countries (public domain), downloaded 2026-10-07 from `https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip` to `data/external/`. The mainland Portugal polygon is the largest part of the Portugal geometry and is also used to decide whether a SEVIRI detection lies in Portugal.
- Colours: one blue for single series; a single-hue blue (or orange) ramp for magnitudes; at most three categorical colours (blue, orange, aqua), checked for colour-blind separation, with different line styles as a second cue.
- ICNF fires used in figs. 02, 07–09: start (UTC) between 2004-01-20 and 2019-12-30 and valid coordinates (361,031 fires; 2,311 of them ≥ 100 ha). When `end_utc` is missing, the start time is used as end.

Exploratory ICNF ↔ SEVIRI matching (figs. 07–09 only; **not** the final event matching):
- Search radius around the ignition point grows with fire size: R = 5 km + 2 · √(A/π), A = burned area in km², capped at 40 km (5.0 km for tiny fires, ~13 km at 1,000 ha, ~30 km at 20,000 ha).
- Fig. 07: a fire is "detected" if any SEVIRI detection lies within R in [start − 1 h, end + 3 h]. The same test with the fire shifted ±7 days (same place and hour) gives the chance rate of matching an unrelated fire.
- Figs. 08/09: generous window [start − 1 h, end + 24 h] (ICNF end times are often too early, see 2.1). A detection matching several fires is assigned to the largest one.

| Figure | Content | Main result |
|---|---|---|
| 01 `icnf_annual` | Fires and burned area per year, 1980–2023; SEVIRI period shaded | The number of records rose from the 1980s to a maximum in 2005 (41.6k) and has fallen since ~2012. Burned area is dominated by single extreme years: 2017 (540 kha), 2003 (469 kha), 2005 (347 kha). |
| 02 `icnf_size_distribution` | Share of fires ≥ x (log-log) and share of burned area from fires ≥ x | Heavy-tailed. In 2004–2019 (with coordinates), fires ≥ 100 ha are **0.64 % of fires but 79.5 % of burned area**; fires ≥ 1,000 ha are 0.08 % of fires and 51 % of the area. |
| 03 `icnf_ignition_map` | Density of all ignition points 2001–2023 (hexbins), and fires ≥ 100 ha sized by area | Ignitions are densest in the north-west (Porto–Braga) and around Lisbon. Large fires concentrate in the centre (Coimbra–Castelo Branco–Leiria), the north-east and the Algarve hills (Monchique, Tavira). Two-thirds of all ignitions and of fires ≥ 100 ha lie north of 40.5° N. |
| 04 `seviri_temporal` | Detections per year, month, hour (UTC); FRP per hour | 85 % of detections in Jul–Oct, plus a small March peak (4.7 %; winter/spring burning, mostly in the north). Detections peak at 16:00 UTC (minimum 07–08 UTC). Median FRP peaks earlier, at 12:00 UTC (121 MW vs. ~57 MW at night). |
| 05 `seviri_frp_confidence` | 2D histogram FRP × confidence; share of detections kept per confidence threshold, by FRP | Confidence rises with FRP. A threshold of 50 keeps 97 % of detections but removes most detections below ~15 MW; 70 removes most below ~30 MW; 90 keeps only 32 % overall. A filter therefore cuts exactly the weak, early detections. The dip of the ≥ 90 curve at ~1,000 MW is at the saturation level. |
| 06 `seviri_pixel_maps` | Detections per SEVIRI pixel; share of each pixel's active days outside Jun–Oct | 8,402 pixels with ≥ 1 detection. **142 "persistent off-season" pixels** (≥ 20 active days, ≥ 50 % of them outside Jun–Oct; 25,366 detections), 93 in Spain: north-west Iberia (Galicia/León, likely winter–spring pasture burning) and a line at ~38.9° N in the Guadiana valley near Badajoz (e.g. one pixel with 112 active days, all off-season; agricultural or industrial sources). 49 in Portugal, mostly in the northern mountains (Minho/Gerês, Trás-os-Montes, Douro) plus a cluster at 40.5° N, −7.5° E; one pixel at 39.44° N, −8.49° E has 49 active days, all off-season. The sources still need checking (e.g. satellite imagery); these pixels are candidates for a mask, but in the north they may be real (pastoral) fires. |
| 07 `detection_probability` | Share of ICNF fires with a SEVIRI detection, by size class, with 95 % Wilson intervals and the ±7-day chance rate | See table below. **Detection rises above chance from ~1–5 ha and passes 50 % between 10–50 ha and 50–100 ha; from 100 ha on, ≥ 80 % are detected.** For the largest fires the chance rate is also high (big radius, long window, busy fire seasons), so this test alone cannot be used as the matching rule. |
| 08 `fre_vs_burned_area` | Fire radiative energy (FRE = Σ FRP × 900 s) vs. burned area, per fire (≥ 100 ha) and per year (all detections in mainland Portugal) | Per year: log-log r = **0.97**, so SEVIRI energy tracks burned area very well. Per fire (1,880 of 2,311 fires ≥ 100 ha received detections): r = 0.58, slope ≈ 1.0 (FRE roughly proportional to area) but large scatter (detections missed under cloud/smoke, wrong assignment between neighbouring fires). |
| 09 `unmatched_hotspots` | SEVIRI detections in mainland Portugal with no ICNF fire within R / the time window: map, share by confidence, share by month | 395,573 detections (70 %) lie in mainland Portugal; **16.9 % of them are unmatched**. Unmatched share falls with confidence (47 % at < 30 → 12 % at 90–100) and is highest in winter (72 % in January vs. 13 % in August). Unmatched detections are spread over the north and centre, plus clusters at the persistent pixels of fig. 06. |

Detection rate by size (fig. 07):

| Burned area [ha] | Fires | Detected [%] | Chance [%] |
|---|---|---|---|
| 0.01–0.1 | 131,293 | 1.9 | 1.3 |
| 0.1–1 | 86,356 | 2.8 | 1.5 |
| 1–5 | 47,447 | 7.9 | 2.3 |
| 5–10 | 7,329 | 24.1 | 3.5 |
| 10–50 | 7,042 | 42.9 | 4.1 |
| 50–100 | 1,558 | 67.7 | 6.9 |
| 100–500 | 1,681 | 80.2 | 9.5 |
| 500–1,000 | 338 | 90.5 | 16.3 |
| 1,000–5,000 | 252 | 96.0 | 28.6 |
| ≥ 5,000 | 40 | 97.5 | 56.2 |

New data-quality finding (not changed in cleaning):
- **Placeholder burned areas:** 17,693 fires have 0 < `burned_area_ha` < 0.001 ha (< 10 m²), with values down to 1e-12 ha (e.g. series of 1e-7 / 1e-8 / 1e-9 in Viana do Castelo 2001 and Leiria 2003). These are codes for "negligible area" rather than measurements. Figs. 02 and 07 start at 0.001 ha / 0.01 ha. Irrelevant for the SEVIRI work (far below the detection limit).

Implications for the next steps:
- **Minimum fire size:** ~100 ha is a sensible lower limit for labelled events (≥ 80 % detected; 2,311 fires in 2004–2019). Going down to ~10–50 ha would add ~8,600 fires, but less than half of them are detected and the detections are weak.
- **Confidence:** do not filter on confidence before matching. Low-confidence detections are more often unmatched (artefacts), but they are also the early-stage signal. Check the persistent off-season pixels and mask those that are clearly not wildfires, and decide on confidence later (possibly as a model input).
- **Spain:** 30 % of all detections lie outside mainland Portugal and have no labels. Keep them as spatial context, but exclude them from labels and from the unmatched statistics.
- **Neighbouring fires:** per-fire energy is noisy partly because hotspots are wrongly assigned between nearby fires. The final matching should cluster detections in space and time first, then link clusters to ICNF fires.

### 2.3 How often are fires and hotspots close in space and time? (2026-10-08)

Script: `src/proximity_analysis.py` (reuses the helpers and the exploratory matching of `src/eda_figures.py`, see 2.2); figures `figures/eda/10_neighbouring_fires.png`, `11_shared_hotspots.png`; numbers in `outputs/proximity_analysis_log.txt`.

Question: how often does a large fire have another fire nearby that could mix up the hotspots or the labels? Focus fires: the 2,311 fires ≥ 100 ha (2004–2019, valid coordinates, UTC start).

**A. Neighbouring ignition points** (fig. 10). Share of focus fires for which another fire starts within D km and ±24 h of its start. Control: the same with the focus fire's start shifted by ±7 days (same place and season, no real simultaneity), averaged over both shifts. Similar fires are mostly relevant for hotspots if they are ≥ 10 ha (SEVIRI rarely sees smaller fires, see fig. 07).

| Neighbour | 5 km | 10 km | 20 km | 30 km | 50 km |
|---|---|---|---|---|---|
| ≥ 100 ha, real (control) | 11 % (0.6) | 23 % (3.7) | 41 % (11) | 55 % (19) | 69 % (30) |
| ≥ 10 ha, real (control) | 23 % (4.4) | 43 % (13) | 65 % (28) | 76 % (41) | 88 % (57) |
| any size, real (control) | 57 % (36) | 80 % (61) | 95 % (82) | 98 % (91) | 100 % (97) |

- With ±72 h, ≥ 10 ha neighbours: 34 % within 5 km, 58 % within 10 km, 78 % within 20 km (control 13 / 28 / 51 %).
- Fires cluster in time far more than chance (control 3–10× lower at short distances), so simultaneous fires are not a rare edge case: **about one in four large fires has another ≥ 100 ha fire starting within 10 km and a day, and about two in three has a ≥ 10 ha fire within 20 km.**
- "Any size" is not informative: small fires are so frequent (especially in the north) that even the control finds a neighbour for most fires.
- Outbreaks dominate: 83 % of the ≥ 100 ha fires start on a calendar day (UTC) with at least one other ≥ 100 ha fire; the busiest days are 2017-10-15 (51 fires), 2016-08-07 (32), 2005-08-20 (32), 2005-08-04 (29), 2016-08-08 (27), 2004-07-26 (25). Not corrected for chance. 61 % of the close pairs (below) fall in 2005, 2016 and 2017.
- Close pairs of ≥ 100 ha fires (≤ 10 km, ≤ 24 h apart): 402 pairs involving 527 of the 2,311 fires. Only 14 pairs are within 3 km and 1 h, so there is little sign of one fire being split into several records.

**B. Shared hotspots** (fig. 11). For each focus fire, the detections inside its search area (radius R(A) and window [start − 1 h, end + 24 h], 2.2) that also fall inside the search area of at least one other fire:

| Competing fires | Focus fires with detections | Share of hotspots shared (median) | Fires with any sharing | > 50 % shared | all shared |
|---|---|---|---|---|---|
| ≥ 100 ha | 2,014 | 0 % | 38 % | 24 % | 9 % |
| ≥ 10 ha | 2,014 | 2 % | 51 % | 33 % | 14 % |

- Most fires (about half) are clean, but a sizeable group is not: **one in three large fires shares more than half of its hotspots with another ≥ 10 ha fire**, and 14 % have no hotspot of their own in this definition (small fire next to a larger one, or overlapping search areas).
- This depends on the generous radius and the end + 24 h window, so it is an upper bound for what a proper clustering would face. It shows the size of the problem: for roughly a third to a half of the labelled fires, hotspots cannot be assigned by proximity to the ICNF point alone.

**C. Shared ignition coordinates (new data-quality finding, not corrected).**
- 72 % of all 477,674 fires with coordinates share their exact coordinates with at least one other fire; 59 % share them with at least four others. There are only 175,357 distinct points. The most used points occur 578, 483, 443, 395 and 393 times (all in the north-west, around Porto/Braga), which suggests that many coordinates are locality positions (place names geocoded) rather than GPS positions of the ignition.
- The coordinate precision is DMS-like (e.g. 41.155556 = 41°09′20″); the share with 2 decimals or fewer is 3 % in 2001–2010 and ≤ 1 % later.
- For fires ≥ 100 ha (2004–2019) it is much less severe: 10 % (about 235 fires) sit at a shared point, 0.4 % at a point used ≥ 5 times. Large fires therefore have less generic coordinates, but the positional accuracy of the labels is still unknown (probably tens to hundreds of metres to several km). This limits what can be demonstrated below one SEVIRI pixel (~3–5 km).

Implications for the next steps:
- Simultaneous fires are common enough (a quarter to a half of the large fires, depending on the radius) that the clustering step cannot be skipped. A fire-by-fire evaluation must report results separately for isolated fires and fires with close neighbours, and the choice of isolated-only samples is a legitimate first model (about half the fires).
- Both numbers must be reported with the matching definition, because they depend on R(A) and the time window.
- Consider checking the labels of the large fires at shared points (about 235) and those within a few km of other ignitions before using them as ground truth.

## 3. Event construction

### 3.1 Clustering SEVIRI hotspots into events (2026-10-08)

Script: `src/cluster_hotspots.py`; outputs `data/processed/seviri_frp_events.parquet` (all detections + `event_id`, `persistent`, `in_pt`) and `data/processed/seviri_events.parquet` (one row per event); figures in `figures/clustering/`; numbers in `outputs/cluster_hotspots_log.txt` and `outputs/cluster_hotspots_sensitivity.txt`.

Goal: group detections that belong to the same fire, **using SEVIRI only**. ICNF is used only to evaluate the result, never to form the events, so events can later define the model input without label leakage (see the input design discussion: the ICNF point may only choose samples and give the label).

Method:
- Two detections are linked if their pixel centres are ≤ D km apart and their times ≤ gap hours apart (including the same time step). Events = connected groups of linked detections (single linkage in space and time, implemented with union–find over the time-sorted detections).
- Distances in local metric coordinates (x = lon · 111.32 · cos(lat), y = lat · 110.57 km).
- Measured before choosing the parameters:
  - Pixel spacing of the active SEVIRI pixels: nearest neighbour ~3.2 km (E–W), 4th nearest ~5.2 km, 8th ~6.3 km (median) → **D = 6 km links a pixel to its 8 neighbours** (one pixel ~3.2 km E–W × ~4.4 km N–S over Portugal).
  - Time since the previous detection within 6 km: 89–92 % within 1 h, ~93 % within 24 h, the rest > 48 h ("new" fires). Only ~2 % of detections have gaps between 3 and 24 h, but these links decide whether neighbouring fires are chained together (see sensitivity).
- Events get ids in order of their first detection. Per event: first/last time, duration, detections, time steps, pixels, FRE, max FRP, median confidence, FRP-weighted centroid (all detections and first time step only), share of detections on persistent off-season pixels (2.2), share inside mainland Portugal.
- `eda_figures.py` got a helper `pixel_stats()` (same persistent-pixel definition as fig. 06), used here to flag detections.

Evaluation measures (ICNF fires ≥ 100 ha, 2004–2019; diagnostic only):
- A fire's event = the event of its earliest detection within 5 km of the ignition point in [start − 1 h, start + 12 h] (found for 78.5 % of the fires, independent of the clustering parameters).
- **Merging:** share of these fires whose event also holds another ≥ 100 ha fire.
- **Fragmentation**, measured on *isolated* fires only (no other fire ≥ 10 ha within 20 km and ±72 h; 515 of 2,311): share of the hotspots matched to the fire (radius R(A), [start − 1 h, end + 24 h], 2.2) that lie in its own event; reported as share of fires with ≥ 90 % in their own event.

Sensitivity (`--sensitivity`):

| D [km] | gap [h] | events | events ≥ 10 det. | merged fires [%] | isolated fires ≥ 90 % intact [%] |
|---|---|---|---|---|---|
| 4.5 | 3 | 32,140 | 7,789 | 21.7 | 71.2 |
| 4.5 | 12 | 28,456 | 6,963 | 29.0 | 79.8 |
| 6 | 3 | 26,504 | 6,964 | 27.4 | 84.0 |
| **6** | **6** | **24,690** | **6,620** | **30.8** | **88.7** |
| 6 | 12 | 23,378 | 6,150 | 36.5 | 93.1 |
| 6 | 24 | 19,511 | 5,226 | 44.7 | 98.0 |
| 9 | 6 | 22,779 | 6,293 | 36.6 | 89.7 |
| 9 | 24 | 17,617 | 4,852 | 50.1 | 98.8 |

- D = 6 km is better than 9 km for every gap (fewer merges for the same fragmentation); D = 4.5 km splits too many isolated fires.
- The gap is a direct trade-off between merging and fragmentation. **Chosen: D = 6 km, gap = 6 h** (middle of the trade-off). Merges cannot be avoided completely with any setting: some fires really do burn into each other.

Result (D = 6 km, gap = 6 h): 24,690 events; 6,620 with ≥ 10 detections hold 91.4 % of all detections (median 26 detections, 4.2 h); 409 of them lie mostly (≥ 50 %) on persistent off-season pixels. Largest event: 11,319 detections.

Case study 2017-10-15 (figures `events_2017-10-15_overview.png`, `events_2017-10-15_panels.png`):
- 51 ICNF fires ≥ 100 ha started that day (UTC), 250,890 ha in total. 39 were linked to an event; the 12 without an early detection are all small (150–354 ha, 2,568 ha together).
- The 39 fires fall into 23 events; **10 events hold ≥ 2 fires (26 fires)**. Example: event #4 holds Lousã (53.6 kha), Figueira da Foz (19.0 kha), Mealhada and Condeixa-a-Nova, linked over > 50 km during the night.
- SEVIRI activity in the large events lasted only 8–16 h. Some very large fires produced few detections: Vouzela (15.7 kha) 4, Alcobaça (17.3 kha) 28, Arganil (38.0 kha) 58 (smoke/cloud likely; not checked).
- Distance between the ICNF ignition and the FRP-weighted centroid of the event's first time step, and lag of the event start after the ICNF start (medians):

| Fire's position in its event | Fires | Distance | Lag |
|---|---|---|---|
| alone in its event | 13 | 3.4 km | +1.3 h |
| first fire of a shared event | 10 | 3.6 km | +0.9 h |
| later fire of a shared event | 16 | 16.7 km | −5.8 h |

Conclusions:
- For isolated fires and for the first fire of a merged event, the event start is a good proxy of the ignition (~1 pixel, ~1 h), consistent with 2.1.
- **For fires that join an event that is already burning, the event start belongs to another fire** (17 km away, ~6 h earlier). These fires cannot be labelled with "their" event as is. Options for the next step: (a) exclude them / label only the first fire of each event; (b) keep the merge history of the clustering: every time a new group of detections appears (a new connected component) it is a separate "nucleus" with its own birth time and position, even if it later merges with others. Nuclei would be the candidate ignitions to link to ICNF fires.
- Fires with few detections (smoke/cloud) remain a limitation that clustering cannot fix.

### 3.2 Nuclei: separate starts inside merged events (2026-10-08)

Same script (`src/cluster_hotspots.py`, run in the same pass as the events); output `data/processed/seviri_nuclei.parquet` (one row per nucleus) and a `nucleus_id` column in `seviri_frp_events.parquet`; figures `figures/clustering/nuclei_vs_events.png` and the updated `events_2017-10-15_panels.png`; numbers in `outputs/cluster_hotspots_log.txt`.

Motivation (3.1): for fires that join an event that is already burning, the event start belongs to another fire. Nuclei keep track of where and when each separate group of detections first appeared, even if the groups later merge into one event.

Method (SEVIRI only, same D = 6 km and gap = 6 h as the events):
- A detection with an earlier linked detection (≤ 6 km, ≤ 6 h) inherits the nucleus of the **nearest** such detection.
- A detection without one, but next to a detection of the same time step that already has a nucleus, inherits from the nearest of those (fire growing within one 15-min step).
- Remaining detections are a **birth**: neighbouring new detections of the same time step form one new nucleus.
- Every nucleus lies inside exactly one event; ids follow birth order. Per nucleus: birth time, birth location (FRP-weighted centroid of the birth time step), size, FRE, duration, order of birth within its event, and two context fields:
  - `dist_active_km`: distance from the birth to the nearest detection of another nucleus active in the previous 6 h (searched up to 100 km);
  - `reappeared`: the birth place had detections within 6 km 6–48 h earlier, i.e. a fire reappearing after a gap (night, cloud, smoke) rather than a new ignition.

Results:
- 25,954 nuclei (24,690 events); 7,264 nuclei with ≥ 10 detections. 10.8 % of the events with ≥ 10 detections contain ≥ 2 nuclei (median 1, max 44).
- Nuclei with ≥ 3 detections: 14,346. 1,069 of them are born after the first nucleus of their event; at birth they are a median 8.9 km from the nearest active detection (just beyond the 6 km link distance: separate fires, spotting or fronts reappearing).
- **32 % of the nuclei with ≥ 3 detections are reappearances** (same place active 6–48 h before), not new ignitions. They must not be used as ignition candidates.
- Diagnostics (as in 3.1): share of ICNF fires ≥ 100 ha sharing their group with another fire: events 30.8 % → nuclei 23.0 %. Isolated fires with ≥ 90 % of their hotspots in one group: events 88.7 % → nuclei 87.2 % (nuclei split a few isolated fires, e.g. after jumps > 6 km).

Distance and lag from each ICNF fire's ignition to its event start (centroid of the event's first time step) vs. to the birth of its own nucleus (ICNF fires ≥ 100 ha 2004–2019 with an event; position = order of ICNF start among the fires linked to the same event):

| Position in event | Fires | Median distance event → nucleus | ≤ 5 km event → nucleus | Median lag event → nucleus | Nucleus shared with another fire |
|---|---|---|---|---|---|
| alone in event | 1,254 | 4.3 → 4.2 km | 65 % → 68 % | +0.2 → +0.2 h | 0 % |
| first in shared event | 198 | 5.5 → 4.9 km | 43 % → 54 % | +0.2 → +0.3 h | 77 % |
| later in shared event | 361 | 11.1 → 6.3 km | 18 % → 36 % | −8.5 → −3.4 h | 73 % |

Same for 2017-10-15: later fires 16.7 → 5.0 km, ≤ 5 km 13 % → 50 %, lag −5.8 → −0.8 h (16 fires).

Conclusions:
- Nuclei roughly **halve the error for fires that join a burning event** (median 11.1 → 6.3 km; twice as many within 5 km) and leave alone/first fires unchanged (~4–5 km, +0.2 h: the "first hotspot" baseline, consistent with 2.1).
- They do not solve everything: about three quarters of the fires in shared events still share their nucleus with another ICNF fire. These fires start within ~1–2 SEVIRI pixels (≤ 6 km) of a front that is already burning, so their first nearby detection belongs to the existing nucleus. At SEVIRI resolution and with this rule they cannot be separated.
- For step 2 (linking to ICNF): link ICNF fires to nuclei, not events; ignore reappeared nuclei as ignition candidates; when several ICNF fires point to the same nucleus, give it to the fire whose start fits the birth time and flag the others as unresolved. The 1,254 fires alone in their event (plus the first fires of shared events) are the cleanest labelled set.

## 4. Model data design (draft)

### 4.0 Change of approach (2026-10-08)

Discussion: if the clustering of 3.1–3.2 decides which hotspots the model sees, it already does part of the work the model should do (assigning hotspots to ignitions), and its errors become the model's errors. Decision (to confirm with the supervisor):
- **Main model: ignition heatmaps from raw FRP sequences.** Input = all detections in a space–time window, no clustering. Target = a heatmap with a Gaussian bump at *every* ICNF ignition in the window (0, 1 or many), i.e. the "ignition probability map" of the proposal. The assignment of hotspots to ignitions is left to the model; no hotspot is assigned by hand. Evaluation: peaks of the predicted map matched to ICNF ignitions within k km → detection rate, false alarms, localisation error (matching only for scoring).
- The clustering (events/nuclei) stays as **baseline** (first hotspot / nucleus birth, ~4–5 km for clean fires), as **analysis tool** (results for isolated vs. merged fires) and as **ablation** (clustered vs. raw input).
- Other options kept in mind: predicting a per-pixel burn time (ignition = earliest), burned-area perimeters (ICNF/EFFIS) as extra supervision, self-supervised pre-training on the unlabelled hotspots.

### 4.1 Is the available data enough? (2026-10-08)

Script: `src/tiling_design.py`; numbers in `outputs/tiling_design_log.txt`.

ICNF fires 2004–2019 with valid coordinates that SEVIRI sees, defined as a detection within 5 km of the ignition point in [start − 1 h, start + 6 h] (6 h so that the signal falls inside the input window of 4.2; a 12 h version gave slightly higher numbers, e.g. 2,813 instead of 2,705 fires ≥ 50 ha):

| Size | Fires | Seen by SEVIRI | Days with ≥ 1 seen fire |
|---|---|---|---|
| ≥ 10 ha | 10,911 | 5,490 (50 %) | 1,382 |
| ≥ 50 ha | 3,869 | 2,705 (70 %) | 910 |
| ≥ 100 ha | 2,311 | 1,738 (75 %) | 663 |
| ≥ 500 ha | 630 | 526 (83 %) | 257 |

Seen fires per year (≥ 50 ha / ≥ 100 ha): 2004 177/118, 2005 494/347, 2006 117/80, 2007 55/26, 2008 37/16, 2009 168/97, 2010 191/117, 2011 149/82, 2012 176/109, 2013 275/187, 2014 51/27, 2015 140/82, 2016 249/170, 2017 302/209, 2018 34/19, 2019 90/52. **42 % of the seen fires ≥ 100 ha come from 2005, 2016 and 2017.**

Assessment: enough for a first model, but tight.
- For: each fire gives several training samples (crops, rotations/flips, time windows), the inputs are small (32 × 32 pixels × 96 steps, few channels), there are 568k unlabelled detections for pre-training, and the baseline to beat is modest (~4–5 km).
- Against: very uneven years (2008: 16 seen fires ≥ 100 ha, 2018: 19), coarse pixels (3.2 × 4.5 km) and uncertain ignition coordinates limit the achievable precision, and the hardest cases (ignitions next to active fronts) number only a few hundred.
- What would help most: (1) SEVIRI FRP for 2020–2023 (ICNF labels already exist; 2022 was a large fire year) — to ask the supervisor; (2) using fires ≥ 50 ha as targets (~1,000 more than ≥ 100 ha); (3) pre-training on unlabelled detections; (4) later the European extension.

### 4.2 Sample design (draft, 2026-10-08)

Script and numbers: as 4.1; example sample `figures/tiling/example_sample.png`.

**Grid: the native SEVIRI pixel grid.** The SEVIRI pixel centres (irregular in lat/lon) can be mapped back exactly to the satellite's own pixel grid with the geostationary projection (`+proj=geos +h=35785831 +a=6378169 +b=6356583.8 +lon_0=0 +sweep=y`, pixel size 3,000.403 m at nadir; row = y / pixel size, col = x / pixel size, rounded):
- Tested sub-satellite longitudes 0°, −3.4°, 3.5°, 9.5°: only **0°** fits (93–96 % of the pixels within ±0.25 px of a grid node in every year, vs. 28–47 % for the others, ~25 % expected by chance), so all of 2004–2019 is on one fixed grid.
- All 8,402 distinct pixels map to distinct native pixels (no collisions); max residual 0.42 px (from the 2-decimal rounding of lat/lon).
- Domain: 177 rows × 106 columns. One pixel over Portugal ≈ 3.2 km E–W × 4.3–4.8 km N–S.
- Advantage: no resampling, each input cell is one real SEVIRI pixel; ICNF ignitions get continuous (sub-pixel) native coordinates.

**Time windows.**
- 12 h target periods starting at 06:00 and 18:00 UTC (06 UTC is the diurnal minimum of fire activity, fig. 04, so few fires straddle the boundary). Each ICNF fire belongs to exactly one period.
- Input per sample: 6 h history + 12 h target period + 6 h tail = 24 h = 96 time steps of 15 min. The history shows whether something was already burning; the tail guarantees ≥ 6 h of observation after every target ignition.
- Input channels per step (first version): FRP (log(1 + FRP)), confidence, detection flag. Later: static layers (DEM, land cover, persistent-pixel flag) and meteorology (ERA5-Land), objective 4. Limitation: "no detection" and "cloud" cannot be distinguished (no cloud mask in the data).

**Targets.**
- Positive: ICNF fires ≥ 50 ha that start in the target period and are seen by SEVIRI (4.1) → Gaussian bump (σ = 1 pixel) at the continuous native position of the ignition. Optionally a second output with the ignition time within the period.
- Ignore (no loss within 2 pixels): fires 10–50 ha and unseen fires ≥ 50 ha (they may or may not produce hotspots, the model should not be punished either way), and the pixels outside mainland Portugal (no labels in Spain).
- Everything else: target 0, including ongoing fires that started before the period.

**Crops and inference.** Train on 32 × 32-pixel crops (≈ 100 × 145 km; positives sampled more often, random flips/rotations), apply the (fully convolutional) model to the whole domain at test time, so no tiling is needed for evaluation.

Numbers (2004-01-20 to 2019-12-30):
- 11,648 target periods; 5,399 with detections in mainland Portugal; **1,147 with ≥ 1 target**; 4,254 with detections but no target (hard negatives, e.g. ongoing or small fires).
- Targets per positive period: 1 → 624 periods, 2–5 → 429, 6–20 → 89, > 20 → 5 (max 31). 2,705 targets and 8,206 ignore regions in total.
- Crops centred on a target fire:

| Crop | ≈ km | Another target in the crop | Earlier detections in the crop (before the fire started) |
|---|---|---|---|
| 16 × 16 | 51 × 72 | 38.6 % | 68.5 % |
| 32 × 32 | 102 × 144 | 56.5 % | 81.1 % |
| 64 × 64 | 205 × 288 | 69.7 % | 90.6 % |

  → Multi-target samples and ongoing activity in the input are the normal case, not the exception, which supports the heatmap formulation over "one fire per sample". The example (Lousã, 2017-10-15 06–18 UTC) has 12 target ignitions in one 32 × 32 crop.

**Split (chronological, no information from the future):**

| Split | Years | Target fires | Positive periods | Hard-negative periods |
|---|---|---|---|---|
| train | 2004–2014 | 1,890 (70 %) | 775 | 2,814 |
| validation | 2015–2016 | 389 (14 %) | 160 | 543 |
| test | 2017–2019 | 426 (16 %) | 212 | 897 |

Note: the test set contains 2017, the most extreme year (Pedrógão, 2017-10-15). Alternative for robustness: leave-one-year-out cross-validation in addition.

Open questions:
- Target size threshold (50 vs. 100 ha) and σ of the bumps (1 px vs. scaled with label uncertainty).
- 15-min steps vs. hourly aggregation (24 steps, fewer gaps, less detail).
- Whether to use the 6 h-"seen" criterion for selecting targets in training only (evaluation should report all fires ≥ threshold, seen or not).
- Handling of the 1,069 nuclei / fires born next to active fronts in the evaluation (report separately).

## 5. Dataset and baselines

### 5.1 Sample index and sample builder (2026-10-08)

Scripts: `src/grid.py` (native grid, shared by all scripts; `tiling_design.py` now uses it, output unchanged), `src/build_dataset.py` (sample index), `src/dataset.py` (dense samples on demand + checks); outputs in `data/processed/model/`; logs `outputs/build_dataset_log.txt`, `outputs/dataset_check_log.txt`; figure `figures/dataset/example_full_domain.png`.

Storage: sparse. A dense full-domain sample is 96 × 3 × 177 × 106 float32 = 21.6 MB, so tensors are built on demand from:
- `detections.parquet`: all 568,123 detections with image indices (i, j) on the native grid (i = north→south, j = west→east, all inside the 177 × 106 grid) and 15-min slot.
- `fires.parquet`: 10,911 ICNF fires ≥ 10 ha (2004–2019) with continuous image position, seen flag (4.1), period, split, `isolated` (no other fire ≥ 10 ha within 20 km and ±72 h; 657 of the 2,705 targets) and size class.
- `labels.parquet`: (period, fire, role) pairs: 2,705 `target` and 8,427 `ignore` entries.
- `periods.parquet`: 11,648 periods with split and counts; `static.npz`: per-cell lat/lon, mainland mask (6,127 cells), loss mask (mainland + 1 px border for coastal/border ignitions, 6,760 cells; all targets inside), persistent off-season cells (142).

Label rules (as 4.2, with two refinements):
- Seen fires ≥ 50 ha that start within 1 h *outside* a period are `ignore` in that neighbouring period (ICNF start times are uncertain by about that much; 221 extra ignore entries).
- **Targets win over ignore regions:** the first version gave 295 of 2,705 targets (11 %) zero weight at their own pixel because they lay within 2 px of an ignore fire. The loss weight is now kept within 1 px of every target.

Sample (`FireData.sample(period, crop=None)`): `x` (96, 3, h, w) with log(1 + FRP)/log(1 + 1100), confidence/100, detected flag; `y` = max of Gaussians (σ = 1 px) at the targets; `w` = loss weight; plus the labels inside the crop. `crop_origin()` places a crop around a position with random jitter.

Checks (300 random periods / all positive periods / 200 crops): every detection of the input window is in `x` (0 mismatches); every target has a heatmap peak at its pixel (0 failures); 0 targets with zero weight; 200 of 200 jittered 32 × 32 crops contain their target.

### 5.2 Evaluation protocol (2026-10-08)

Script: `src/evaluate.py`. Predictions are points (period, lat, lon, score); a heatmap model will be converted by peak extraction.
- Per period, predictions in descending score order: matched to the nearest unmatched target within k km → true positive; otherwise within k km of an ignore fire → ignored; otherwise false positive (duplicates too). Predictions outside the loss mask (e.g. Spain) are dropped.
- k = 5 km (about one pixel diagonal) and 10 km.
- Metrics per split: average precision (AP, area under the interpolated precision–recall curve), recall and precision with all predictions, best F1 (with P, R, false alarms per day), median error of true positives, recall by subgroup (isolated / size class), and the distance from each target to the nearest prediction of its period (no matching).
- **The test split (2017–2019) stays locked** until the final comparison; baselines and model choices use train and validation only.

### 5.3 Baselines from nucleus births (2026-10-08)

Script: `src/baselines.py`; results `outputs/baselines_log.txt`; figure `figures/baselines/pr_curves_val.png`.

Every nucleus (3.2) born inside a target period = one predicted ignition, located at its birth (FRP-weighted centroid of the birth time step). Score = fire radiative energy of the nucleus inside the 24 h input window (no future information). Variants: without reappeared nuclei; also without nuclei on persistent pixels; first-hour centroid instead of birth location.

Validation 2015–2016 (389 targets; train 2004–2014 in brackets, same picture):

| Baseline | AP @ 5 km | AP @ 10 km | Recall @ 10 km (all preds) | Best F1 @ 10 km (P / R) | False alarms/day @ best F1 | Median error TP @ 10 km |
|---|---|---|---|---|---|---|
| **birth** | **0.24** (0.22) | **0.57** (0.52) | **0.79** (0.81) | 0.64 (0.68 / 0.59) | 0.15 | 4.3 km |
| birth, no reappeared | 0.23 | 0.55 | 0.64 | 0.63 (0.71 / 0.56) | 0.12 | 4.3 km |
| birth, no reappeared/persist. | 0.23 | 0.54 | 0.63 | 0.63 (0.73 / 0.56) | 0.11 | 4.3 km |
| first-hour centroid | 0.26 | 0.55 | 0.63 | 0.63 (0.74 / 0.56) | 0.11 | 4.3 km |

Recall with all predictions, "birth", validation: isolated targets 0.72 @ 5 km / 0.95 @ 10 km; non-isolated 0.45 / 0.74. Nearest prediction within 5 km for 53 % of the targets (median 4.8 km).

Findings:
- **Reference baseline: "birth" (all nuclei, FRE score): AP 0.24 @ 5 km, 0.57 @ 10 km on validation.** This is the bar for the model.
- Removing reappeared nuclei raises precision at low recall but caps recall at ~0.63 (non-isolated targets drop from 0.74 to 0.54 @ 10 km). Many ICNF fires start where something burned 6–48 h before (probably rekindles, which ICNF records as new occurrences, and busy fire areas), so "reappeared" is not a safe "not an ignition" flag. It may still be useful as a model input.
- Localisation is the weak point: median error of matches ~3.5 km @ 5 km and ~4.3 km @ 10 km, only about half of the targets have a prediction within 5 km. A model must improve AP @ 5 km in particular.
- Isolated fires are almost always found within 10 km (95 %); the gap is in fires with neighbours (74 %), where the model has the most room.

## 6. Models

### 6.1 ConvLSTM v1: setup (2026-10-08)

Scripts: `src/model.py` (network, loss), `src/train.py` (features, training crops, training loop, full-domain prediction and peak extraction), `src/evaluate_model.py`. PyTorch 2.14.1 installed into `.venv` (added to `requirements.txt`); training on the Apple M1 Pro GPU (MPS).

Input features per model step (7 channels): per-hour maximum of log(1 + FRP), confidence and detection flag (the 96 15-min steps aggregated to **24 hourly steps** to keep training cheap; 15-min steps are an open question), a "target period" flag (hours 6–17 of the window), the normalised time, the mainland mask and the persistent-pixel flag.

Diagram: `figures/model/convlstm_v1_diagram.png` / `.pdf` (`src/plot_model_diagram.py`; parameter counts read from the model: encoder 3,344, ConvLSTM 1 55,424, ConvLSTM 2 73,856, head 9,281).

Network (`IgnitionConvLSTM`, 141,905 parameters): per-step encoder (2 × conv 3 × 3, 16 channels) → 2 ConvLSTM layers (Shi et al., 2015; 32 hidden channels, 3 × 3 kernels) → head (conv 3 × 3, 32 → 1) on the final hidden state → ignition heatmap logits. Fully convolutional: trained on crops, applied to the full 177 × 106 grid. Output bias initialised to a prior probability of 0.01.

Loss: penalty-reduced focal loss of CenterNet (Zhou et al., 2019; α = 2, β = 4) on the Gaussian targets, positives = pixels with y ≥ 0.75 (the pixel nearest each sub-pixel ignition always qualifies), multiplied by the loss weight `w` (ignore regions, outside Portugal = 0), normalised by the number of positives.

Training: per epoch 3,780 crops of 32 × 32 pixels from 2004–2014: one crop per target ignition (1,890; random shift ±8 px) and as many hard negatives (crop around a random detection in Portugal in a period without target, redrawn every epoch); random horizontal/vertical flips. AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 16, 20 epochs, gradient clipping at 5, seed 0. ~1.2 min per epoch on MPS.

Validation every 2 epochs: full-domain prediction for the 844 validation periods with detections → local maxima (3 × 3) with probability ≥ 0.05, max. 50 per period, sub-pixel position from the probability-weighted 3 × 3 centroid → points scored with `src/evaluate.py` (5.2). The checkpoint with the best mean of AP @ 5 km and AP @ 10 km is kept (`models/convlstm_v1.pt`). Periods without any detection get no predictions. The test split is not touched.

#### 6.1.1 Detailed description of the model (for the write-up)

**Task.** For each 12 h target period (06–18 or 18–06 UTC) the model looks at 24 h of SEVIRI detections and outputs an ignition probability map on the native SEVIRI grid: per pixel, the probability that an ICNF fire ≥ 50 ha started there during the target period. Zero, one or many ignitions per map; no hand-made hotspot ↔ fire assignment.

**Input.** One sample = 6 h history + 12 h target period + 6 h tail (history: was something already burning; tail: ≥ 6 h of observation after every possible ignition). The 96 15-min slots are aggregated to 24 hourly steps (maximum per hour). Seven channels per step: (1) log(1 + FRP)/log(1 + 1100); (2) confidence/100; (3) detected flag; (4) target-period flag (hours 7–18); (5) normalised time 0 → 1; (6) mainland mask; (7) persistent-pixel flag. Channels 4–5 tell the model *when* each step is, 6–7 *where*. A pixel without detection is 0 — "no fire" and "cloud" cannot be told apart. Tensor: (batch, 24, 7, H, W); training on 32 × 32 crops, inference on the full 177 × 106 grid.

**Network** (141,905 parameters):
- *Encoder* (3,344), applied to every hour with shared weights: Conv 3×3 (7→16) → ReLU → Conv 3×3 (16→16) → ReLU. Turns the raw channels into 16 local features per pixel (receptive field 5 × 5 px ≈ 16 × 22 km).
- *Two ConvLSTM layers* (32 channels, 3×3 kernels; 55,424 and 73,856 parameters; Shi et al., 2015): an LSTM in which all matrix products are 3×3 convolutions, so hidden state and memory are maps (one memory per pixel, updated from its neighbourhood). Per hour t: [i, f, o, g] = Conv3×3([x_t, h_{t−1}]); c_t = σ(f) ⊙ c_{t−1} + σ(i) ⊙ tanh(g); h_t = σ(o) ⊙ tanh(c_t). Forget gate f decides what memory to keep, input gate i what to write, output gate o what to pass on. Layer 2 reads layer 1's h_t; both start at zero and run through all 24 hours.
  Rationale: an ignition is a space–time pattern (a pixel lights up where nothing burned, then the fire spreads to its neighbours). The per-pixel memory can keep "this spot lit up first, at hour 9, and the fire spread north-east from it" — the signal for the origin — and from the history whether something was already burning.
- *Head* (9,281), on the final hidden state of layer 2 only (after hour 24): Conv 3×3 (32→32) → ReLU → Conv 1×1 (32→1) → logit per pixel, sigmoid → probability. The output bias starts at probability 0.01 (ignitions are rare).
- Fully convolutional: the same weights work on crops and on the full grid.

**Target and loss.**
- Target y: a Gaussian (σ = 1 px) at the sub-pixel position of every target ignition of the period (maximum where bumps overlap).
- Loss weight w: 1 inside Portugal (+ 1-px border); 0 outside and within 2 px of every ignore fire (4.2, 5.1), but kept within 1 px of every target.
- Loss: CenterNet focal loss (Zhou et al., 2019; α = 2, β = 4). Positive pixels (y ≥ 0.75, includes the pixel nearest each ignition): −(1 − p)² log p — a confident hit costs little, a miss a lot. All other pixels: −(1 − y)⁴ p² log(1 − p) — false positives are penalised, much less close to an ignition (being 1 px off is cheap, far off is expensive). Multiplied by w and divided by the number of positives. The focal terms prevent the ~1,000 easy "no ignition" pixels per crop from swamping the one or two positives.

**Training.** Years 2004–2014. Per epoch 3,780 crops of 32 × 32 px (~100 × 145 km): 1,890 positives (one per target ignition, shifted ±8 px at random so targets are not always centred) and 1,890 hard negatives (crop around a random Portuguese detection in a period without target, redrawn each epoch, so the model learns that "fire present" ≠ "new ignition"). Random horizontal/vertical flips. AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 16, gradient clipping 5, 20 epochs, seed 0, ~1.2 min/epoch on the M1 GPU. Every 2 epochs a full validation (2015–2016); the checkpoint with the best mean AP @ 5/10 km is kept (epoch 20).

**Inference: heatmap → points.** For every period with ≥ 1 detection: full-grid probability map → local maxima (3 × 3) with p ≥ 0.05, max. 50 per period → sub-pixel position from the probability-weighted 3 × 3 centroid → lat/lon via the geostationary projection → (period, lat, lon, score).

**Scoring.** As for the baselines (5.2): per period, predictions by descending score match the nearest unmatched target within k = 5 / 10 km (hit), else ignored if within k km of an ignore fire, else false alarm; AP, recall, best F1, false alarms/day, median error, recall by subgroup.

**Strengths.** Sees all detections at once, including neighbouring fires, and can learn which new hotspot is a new ignition and which a spreading front — the largest gain over the clustering is for fires with neighbours (recall 0.83 vs. 0.74 @ 10 km). Small (~140k parameters) for ~1,900 training targets; fast to train.

**Limitations.**
- Hourly steps blur the first 15-min observations of a fire, often the most informative for the location.
- Single readout after hour 24: early information must survive ~18 h of updates. Reading out all steps or attention over time (the planned CNN + Transformer) would remove this bottleneck.
- Ignitions closer than ~2 px merge into one peak (18 points for 31 ignitions on 2016-08-07, 6.3).
- Probabilities peak at ~0.55; calibration not checked.
- No cloud mask ("no fire" vs. "not visible"), no covariates (terrain, land cover, wind).
- One seed; checkpoint chosen on validation; AP @ 5 km fluctuates strongly between epochs.

#### 6.1.2 How the ConvLSTM cell works and why the receptive field grows

Diagram: `figures/model/convlstm_cell_detail.png` / `.pdf` (`src/plot_model_diagram.py`): (a) one cell, (b) receptive field within one step, (c) over time.

**One step of one layer.** Each layer keeps two maps from the previous step, both 32 channels × H × W: the hidden state h (what the layer outputs) and the memory c (what it remembers).
1. *One 3×3 convolution* over the new input x_t (16 encoder channels for layer 1, layer 1's 32-channel h_t for layer 2) stacked with h_{t−1}: 48 (or 64) channels → 4 × 32 = 128 channels, split into input gate i, forget gate f, output gate o and candidate g (code: `self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, 3, padding=1)` in `src/model.py`). This is the only spatial operation of the cell: each gate value at a pixel depends on that pixel and its 8 neighbours, in the current input and in the previous hidden state. The parameters of a ConvLSTM layer are the weights of this one convolution (layer 1: 55,424; layer 2: 73,856).
2. *Pixel-wise update* (⊙ = element-wise, no exchange between pixels): c_t = σ(f) ⊙ c_{t−1} + σ(i) ⊙ tanh(g) (keep part of the old memory, write part of the new content); h_t = σ(o) ⊙ tanh(c_t). σ (sigmoid) turns the gates into 0–1 "valves", tanh bounds values to −1…1.
So every pixel has its own memory, but what it writes, keeps and forgets is decided from its 3×3 neighbourhood. Difference to a normal LSTM: matrix products over a flat vector are replaced by convolutions, so the state stays a map and the spatial structure is kept (Shi et al., 2015).

**Receptive field** (the input area that can influence one output pixel). Every 3×3 convolution extends it by one pixel in each direction.
- *Stacking, within one time step:* encoder conv 1 → 3 × 3, encoder conv 2 → 5 × 5, ConvLSTM 1 gates → 7 × 7, ConvLSTM 2 gates → 9 × 9 px (≈ 29 × 40 km), head conv → 11 × 11. With a single ConvLSTM layer it would stop at 7 × 7 per step.
- *Recurrence, over time:* h_{t−1} passes the 3×3 convolution again at step t, so information from a neighbour arrives one step later, from a neighbour's neighbour two steps later, etc.; in theory the whole 32 × 32 crop is reachable after 24 steps.
- *Effective vs. theoretical:* gates and repeated squashing damp distant/old signals, so the effective receptive field is much smaller than the theoretical one. This motivates more context per step (layers, channels, kernel size) or direct access to all time steps (attention, CNN + Transformer).

**Why two ConvLSTM layers (a default, not yet tested).** (1) More spatial context per step (9 × 9 instead of 7 × 7 px) for a relational question ("which nearby hotspot came first, where did it spread from"); (2) stacked recurrent layers often split the work into fast/local (layer 1) and slower/abstract state (layer 2), as in the original ConvLSTM paper — not verified here; (3) still small (129k of 142k parameters; one 32-channel layer would be ~55k). Arguments for one layer: fewer parameters (overfitting with ~1,900 targets), easier to interpret, faster (relevant for 15-min steps). **Open experiment (ablation):** 1 vs. 2 vs. 3 layers and 16/32/64 channels, several seeds each (AP @ 5 km varies too much between single runs); until then "2 layers × 32 channels" must be described as a default choice.

### 6.2 ConvLSTM v1: results on validation (2026-10-08)

Training log: `outputs/convlstm_v1_log.txt`, `outputs/convlstm_v1_history.csv`; evaluation: `outputs/convlstm_v1_eval.txt`; figure `figures/model/convlstm_v1_pr_val.png`. (The run was restarted manually with the same configuration; the re-evaluation of the saved checkpoint reproduces the logged numbers exactly.)

- Training loss 2.43 → 0.80; best checkpoint at epoch 20 (also the last).
- Validation AP during training (every 2 epochs): AP @ 10 km rises steadily (0.53 → 0.66); **AP @ 5 km fluctuates strongly between epochs (0.14–0.30)**, i.e. the precise position of the peaks is not stable yet.

Validation 2015–2016, 389 target ignitions:

| Method | k | AP | Recall (all preds) | Best F1 (P / R) | False alarms/day @ best F1 | Median error TP | Nearest pred ≤ 5 km |
|---|---|---|---|---|---|---|---|
| baseline: nucleus birth | 5 km | 0.237 | 0.51 | 0.40 (0.42 / 0.39) | 0.28 | 3.6 km | 53 % |
| **ConvLSTM v1** | 5 km | **0.302** | **0.60** | **0.45** (0.45 / 0.44) | 0.29 | **3.1 km** | **62 %** |
| baseline: nucleus birth | 10 km | 0.569 | 0.79 | 0.64 (0.68 / 0.59) | 0.15 | 4.3 km | |
| **ConvLSTM v1** | 10 km | **0.664** | **0.87** | **0.69** (0.72 / 0.65) | 0.13 | **4.0 km** | |

Recall (all predictions) by subgroup, 10 km / 5 km: isolated fires 0.99 / 0.76 (baseline 0.95 / 0.72); **fires with neighbours 0.83 / 0.55 (baseline 0.74 / 0.45)**; ≥ 1,000 ha 0.93 / 0.70 (baseline 0.73 / 0.57).

Interpretation:
- The first ConvLSTM, without covariates and with hourly steps, **beats the nucleus-birth baseline on every metric**: AP +0.07 (5 km) and +0.10 (10 km), its precision–recall curve lies above the baseline almost everywhere, and the median error of matches drops by ~0.3–0.4 km.
- The gain is largest where the hand-made clustering is weakest: fires with neighbours (+9–10 points recall) and large fires (+20 points at 10 km). It shows the model can separate ignitions from raw input better than the clustering rule.
- Localisation remains the limit: median error ~3 km at 5 km matching, ~1 pixel; 38 % of targets have no prediction within 5 km.

Caveats:
- One run, one seed. The checkpoint was selected on the validation split, so these validation numbers are slightly optimistic; the unbiased number will come from the locked test split.
- AP @ 5 km is unstable between epochs; more runs/seeds are needed before drawing fine conclusions.


### 6.3 Predicted vs. ICNF ignition maps (2026-10-16)

Script: `src/plot_predictions.py`; figures `figures/model/convlstm_v1_predictions_val.png` (4 periods) and `convlstm_v1_predictions_val_zoom.png` (busiest period); log `outputs/convlstm_v1_predictions_log.txt`. Validation only.
- Periods chosen by rule, not by hand: the validation period with most targets (2016-08-07 06 UTC, 31 targets), a random period with 2–5 targets (2016-09-05 18 UTC, 3), a random single-target period (2016-07-30 06 UTC, 1), a random period with detections but no target (2015-11-14 18 UTC); seed 0.
- Predicted points shown above the best-F1 threshold at 10 km on the whole validation split: p ≥ 0.38. The probability colour scale is 0–0.6 (the model's maximum is ~0.55).
- Observations: the single-target and 2–5-target periods are found exactly (1/1 and 3/3 predicted points next to the ICNF ignitions); the period without target has no prediction (max p 0.04). In the busy period the model marks the right areas, but **produces 18 points for 31 ignitions**: ignitions closer than ~2 pixels merge into one broad blob, so only one peak survives (3 × 3 peak extraction, σ = 1 px targets). Peak probabilities stay ≤ ~0.55, i.e. the model is cautious/uncalibrated.
- Ideas from this: sharper heatmaps (smaller σ, longer training), a lower peak-extraction neighbourhood, or predicting a count/offset per peak; check calibration.


### 6.4 Design discussion and open experiments (2026-10-16)

**Training period.** Train 2004–2014 (1,890 targets, 70 %), validation 2015–2016 (389), test 2017–2019 (426) — a chronological split: (a) no leakage between simultaneous fires of the same outbreak (e.g. 51 fires on 2017-10-15) across splits, (b) mirrors operational use (train on the past, apply to later years), (c) validation for choices, test touched once. Cost: 30 % of the labelled fires are not used for training. Planned remedies: once the configuration is fixed, retrain on 2004–2016 (2,279 targets) and evaluate once on 2017–2019; leave-years-out cross-validation for robust estimates; SEVIRI 2020–2023 if obtainable.

**Window length / multi-day propagation.** Every sample is independent and the ConvLSTM state is reset per sample: the model sees exactly 24 h (6 h history, 12 h period, 6 h tail) and nothing about multi-day evolution. Rationale: the label is the ignition within the 12 h; the most informative signal is the first hours of a fire; the history only has to show what already burned; short sequences keep training cheap. Weaknesses:
- a fire that burned earlier, paused for > 6 h (night, cloud) and flares up again looks new (cf. 32 % reappeared nuclei, 3.2; rekindles in ICNF);
- post-fire reconstruction of large multi-day fires could use the full spread (direction of growth points back to the origin), which the 6 h tail discards;
- the single readout after hour 24 must carry early information through ~18 h; longer windows make this harder for this architecture.
Two framings to keep apart in the thesis: *early detection* (short tail, operational) vs. *post-fire reconstruction* (long tail, full fire evolution).

**Grid.** The model already works on the native SEVIRI pixel grid (reconstructed exactly, 4.2: 8,402 pixels → 8,402 cells, no resampling; ~3.2 × 4.5 km over Portugal). A finer grid would only interpolate; sub-pixel precision comes from sub-pixel target positions, sub-pixel peak refinement and the distribution of FRP over neighbouring pixels (median error ~3 km is already below the pixel diagonal of ~5.5 km). What is coarser than native is *time*: 15-min slots are aggregated to hourly steps in v1.

**Feedback between consecutive samples.** Idea: pass information from period k into period k + 1, so the model effectively runs continuously through the season.
- *Option A — stateful ConvLSTM:* carry the hidden/memory maps (h, c) from the end of period k into period k + 1. Unbounded, learned memory at the cost of 24 steps per period; matches operational use (one map every 12 h). Requires training on chunks of consecutive periods in time order (e.g. 6 periods = 3 days) with truncated backpropagation through time (gradients within a chunk, cut between chunks); fixed tiles or the full grid instead of jittered crops (state belongs to a place); sampling so that chunks with ignitions are frequent enough; warm-up of the state before validation/test; possibly a state reset after several days without detections to avoid drift.
- *Option B — feed back as input channels:* the model's own previous prediction (risk: trained on perfect inputs, used with its own imperfect ones; errors can compound) or satellite-only summaries of the longer past, e.g. "hours since this pixel last burned (≤ 72 h)", "burned in the last 48 h". Samples stay independent; crops, shuffling and flips still work.
- *Never feed back ICNF information* (earlier ignitions, areas, labels): not available in real use and a direct label leak in evaluation.
- Recommendation: first Option B with satellite-only summary channels (cheap test whether longer memory helps), then, if it does, Option A as a stateful ConvLSTM; compare v1, v1 + summary channels and stateful on the same split.

**Open experiments (in suggested order):**
1. 15-min steps (96 steps) vs. hourly (v1).
2. Long-history summary channels (Option B).
3. Longer windows (history 24–48 h, tail 24–72 h) — early-detection vs. reconstruction framing.
4. Stateful ConvLSTM across consecutive periods (Option A).
5. Several seeds per configuration (variance; AP @ 5 km is unstable).
5b. Depth/width ablation: 1 vs. 2 vs. 3 ConvLSTM layers, 16/32/64 channels (6.1.2).
6. Final: retrain the chosen configuration on 2004–2016, evaluate once on 2017–2019; optionally leave-years-out cross-validation.


### 6.5 ConvLSTM v2: native 15-min steps (2026-10-16)

Open experiment 1 of 6.4: does the native time resolution help, especially the localisation (AP @ 5 km)?
- Change: `--step-agg 1` → 96 steps of 15 min instead of 24 hourly maxima; everything else identical to v1 (architecture, 7 channels, loss, crops, 20 epochs, seed 0, selection on mean AP @ 5/10 km). The target-period flag covers steps 25–72, the time channel runs 0 → 1 over 96 steps. `features()` now checks that the step size divides the window and the period boundaries.
- Evaluation and plotting read the step size from the checkpoint, so `evaluate_model.py` and `plot_predictions.py` work unchanged.
- Expected cost: ~4× training time (~5 min/epoch on the M1 GPU, ~1.5–2 h for 20 epochs incl. validation) and ~4× activation memory (estimated ~3 GB at batch 16; reduce `--batch` if MPS runs out of memory — then also note it here, it changes the comparison).
- Command: `.venv/bin/python src/train.py --name convlstm_v2_15min --step-agg 1 > outputs/convlstm_v2_15min_log.txt`
- Run done 2026-10-16 (by hand, configuration as above, batch 16 fit into memory): 4.4 min per training epoch (v1: 1.2), 3.5 min per full validation (v1: 1.0); best checkpoint at **epoch 14** (mean AP @ 5/10 km 0.510). Logs `outputs/convlstm_v2_15min_{log,history,eval,predictions_log}`, figures `figures/model/convlstm_v2_15min_{pr_val,predictions_val,predictions_val_zoom}.png`.

**Result (validation 2015–2016, 389 targets):**

| | baseline | v1 (hourly) | **v2 (15 min)** |
|---|---|---|---|
| AP @ 5 km | 0.237 | 0.302 | **0.348** |
| AP @ 10 km | 0.569 | 0.664 | **0.672** |
| recall (all preds) @ 5 / 10 km | 0.51 / 0.79 | 0.60 / 0.87 | **0.64 / 0.87** |
| best F1 @ 5 km (P / R) | 0.40 (0.42 / 0.39) | 0.45 (0.45 / 0.44) | **0.48 (0.53 / 0.44)** |
| best F1 @ 10 km (P / R) | 0.64 (0.68 / 0.59) | **0.69** (0.72 / 0.65) | 0.68 (0.77 / 0.60) |
| false alarms/day @ best F1, 5 / 10 km | 0.28 / 0.15 | 0.29 / 0.13 | **0.21 / 0.10** |
| median error of matches, 5 / 10 km | 3.6 / 4.3 km | 3.1 / 4.0 km | 3.2 / **3.9** km |
| nearest prediction ≤ 5 km | 53 % | 62 % | **66 %** |
| recall non-isolated, 5 / 10 km | 0.45 / 0.74 | 0.55 / 0.83 | **0.59 / 0.84** |
| recall ≥ 1,000 ha, 5 / 10 km | 0.57 / 0.73 | 0.70 / 0.93 | **0.77 / 0.93** |

Less dependent on the choice of the best epoch: mean of the last five validation evaluations (epochs 12–20): AP @ 5 km v1 0.282 ± 0.025 vs. **v2 0.316 ± 0.047**; AP @ 10 km v1 0.661 ± 0.005 vs. **v2 0.673 ± 0.007**. AP @ 5 km still jumps between evaluations (v2: 0.11–0.35 over the run; e.g. 0.23 at epoch 16).

Interpretation:
- Native 15-min steps help mainly the **precise localisation**: AP @ 5 km +0.05 (best epoch) / +0.03 (last-five mean), 4 points more targets with a prediction within 5 km, gains for fires with neighbours and large fires at 5 km. At 10 km the models are almost equal (+0.01 AP). This fits the expectation that the first 15-min observations of a fire carry the location.
- v2 is also more conservative at its operating point: fewer false alarms (0.10 vs. 0.13 per day @ 10 km) at slightly lower recall.
- The maps look like v1: the right areas, but close ignitions still merge (19 points for 31 ignitions on 2016-08-07; max p ≈ 0.54).
- Cost: ~3.7× training time and ~3.5× validation time.
- **Caveat:** one seed per model; the 5 km gain (+0.03–0.05) is about the size of the epoch-to-epoch fluctuation of AP @ 5 km (std 0.025–0.047 over the last evaluations). Encouraging but not yet conclusive — needs several seeds per configuration before choosing 15-min steps as the default.

Next steps (candidates): repeat with several seeds (variance); 15-min steps vs. hourly; longer training / learning-rate schedule; ablations (clustered vs. raw input, without time/static channels); covariates (DEM, land cover, ERA5-Land wind); then the CNN + Transformer model; final evaluation on the test split.


### 6.6 Long-term memory across periods: concept (draft, 2026-10-16)

Goal: let the model use information older than the 6 h history, so that it "sees" fires that develop over several days. Extends two items of 6.4 (window length, feedback between consecutive samples) with two further ideas: an always-on feedback loop into the next sample, and a long-term gate that writes only relevant information. Concept only, no code yet; to discuss with the supervisor. The numbers below come from a quick check outside `src/` (preliminary); they will be redone as a script with the first memory experiment.

Diagram of the recommended design (Option 4): `figures/model/convlstm_memory_concept.png` / `.pdf` (`src/plot_model_diagram.py`).

**Why: how often older context matters**
- Targets 2004–2016 (train + val, 2,279): detections near the ignition *before* the 24 h input window, i.e. information v2 cannot see. In brackets: earlier activity but an empty 6 h history ("pause > 6 h") — for v2 these fires look fresh although the place burned before.

| Radius around the ignition | In the 6 h history (v2 sees it) | Before the window: last 24 h | last 48 h | last 7 d |
|---|---|---|---|---|
| 2 px (~7 km) | 15 % | 27 % (15 %) | 35 % (22 %) | 49 % (36 %) |
| 5 px (~18 km) | 31 % | 50 % (23 %) | 59 % (31 %) | 74 % (44 %) |

- Multi-day fires carry much of the signal: only 4.5 % of the events with ≥ 10 detections (mostly in Portugal) last > 24 h, but **41 % of the detections in Portugal belong to events lasting > 24 h** (25 % > 48 h). 33 % of the nuclei are reappearances (3.2).
- **Where v2 fails** (validation; context = detections within 2 px in the 72 h before the input window / in its 6 h history):

| Context of the target | Targets | Recall @ 10 km | Recall @ 5 km |
|---|---|---|---|
| fresh (nothing in the 78 h before the period) | 227 | 0.95 | 0.71 |
| pause > 6 h (activity 6–78 h before the period, empty history) | 83 | 0.86 | 0.65 |
| active in the 6 h history | 79 | 0.68 | 0.41 |

  False alarms at the best-F1 threshold (p ≥ 0.42) by the same context: @ 10 km **31 of 71 (44 %) at "pause > 6 h" places** (fresh 22, active 18); @ 5 km 52 of 155 (fresh 77, active 26).
- Interpretation: "pause > 6 h" places are 21 % of the targets but 44 % of the false alarms @ 10 km, and their recall is 9 points lower. v2 sees "a fire appears" but cannot know that the place burned one to three days before — exactly what a long-term memory adds. The weakest class, "active in history" (ignitions next to a burning front), already has its context inside the window; memory will not fix it (spatial separation problem, 6.3).
- Caveat that shapes the design: memory cuts both ways. Earlier activity is evidence for "not new" (reappearance after night, cloud, smoke), but ICNF also records rekindles as new fires (5.3: dropping reappeared nuclei cuts the baseline recall from 0.79 to 0.64 @ 10 km). The memory gives the model the information to separate the two; whether it can depends on knowing *why* the place was quiet (extinguished or hidden by cloud), see the cloud mask below.

**Design space: four questions**
1. *What* is carried over: hand-made summaries of past detections, the model's own past predictions, a learned latent map, or the full recurrent state (h, c).
2. *At which clock*: every 15-min step (one long sequence) or once per 12 h period (a slow memory next to the fast per-window model).
3. *How it is updated*: always (plain recurrence), learned gate, sparse or event-triggered gate, time-based decay.
4. *How it is trained*: independent samples, samples with a lookback of K previous periods, or continuous streams with truncated backpropagation through time (TBPTT).
The feedback loop = "always" update with a learned map or state; the long-term gate = gated update at a slow clock. Most of the complexity and cost lies in the training protocol (question 4), not in the gate, which is one extra convolution.

**Option 1 — Hand-made long-history channels (fixed memory).** 6.4 Option B, made concrete.
- Per pixel, causally from all detections before each step: exponential moving averages of the detection flag and of log(1 + FRP) with τ = 6, 24, 72, 168 h (leaky integrators m_t = e^(−Δt/τ) m_{t−1} + x_t: a recurrence with fixed gates), hours since the last detection (log, capped at 7 d), cumulative FRE of the last 7 d; optionally a 3 × 3 maximum for "burning next door". Either per step or once at the start of the window.
- Chronology: none in training; computed from the time-sorted detection table (`searchsorted` over [s0 − 7 d, s0), as in `FireData.sample`).
- Batching unchanged (crops, jitter, flips, hard negatives, shuffling). Cost ≈ v2 (a few more input channels). Flip-equivariant by construction.
- Strength: the cheapest test of "does older context help at all"; an ablation per τ is directly interpretable; no train/test mismatch. Limit: what is remembered is fixed (no "this front moved north-east for two days"), no learned relevance.

**Option 2 — Feedback of the model's own prediction (feedback loop, output version).**
- Extra input channel(s): the predicted heatmap of periods k − 1 (and k − 2). Tells the model "I already called an ignition here" → fewer duplicates across period boundaries.
- Training by self-conditioning (Chen et al., 2022): run the current model on k − 1 without gradient and feed its detached output (zeros half of the time), so that the model is trained on its own imperfect inputs (avoids the exposure bias of training on perfect inputs, cf. scheduled sampling, Bengio et al., 2015). Never the ICNF labels (label leak, 6.4).
- Chronology: one period of lookback per sample; batches unchanged. Cost ≈ +20 % per lookback period at 15 min (one forward pass without gradient ≈ 1/5 of a training step).
- Limit: carries only what the head outputs (ignition probability), not "something burned here"; older information only through the chain of outputs. A consistency add-on, not the main memory.

**Option 3 — Always-on latent feedback / stateful ConvLSTM (feedback loop, state version).**
- 3a: the network outputs an extra M-channel map m_k (tanh) next to the heatmap; m_{k−1} enters period k as extra input channels (or as initial state). An ungated recurrence at the 12 h clock whose transition is the whole network.
- 3b: stateful ConvLSTM (6.4 Option A): no overlapping windows, one continuous stream per episode with (h, c) carried on; a heatmap is read out every 12 h at the end of each period's tail (period start + 18 h).
- Chronology: full. Episodes = runs of 12 h periods with detections in Portugal, cut after ≥ 24 h without any detection in the domain; state reset at each episode start. 2004–2019 (persistent pixels still included): 685 episodes, median 3 periods, 90 % ≤ 15 periods (7.5 d), max 105 (52 d); 95 % of the targets lie in episodes of ≥ 10 periods (5 d).
- Training: B parallel streams (episodes packed one after another, as in language-model batching); every iteration advances each stream by one chunk (1–2 periods = 48–96 steps at 15 min), the state is detached and handed over between iterations (TBPTT; Williams & Peng, 1990), masks reset it at episode boundaries; a loss at every readout inside the chunk, normalised by the positives of the whole batch (most periods have none).
- Cost: every 15-min slot is processed once (v2's 24 h windows with a 12 h stride process every slot twice), but every active period on the full grid (or fixed tiles): ~2,750–3,000 active training periods × 48 steps × 18,762 px ≈ 2.5–2.7 G pixel-steps per epoch vs. 0.37 G for v2 → **~7× v2 per epoch** (~30 min at 15 min, ~8 min hourly on the M1); roughly 2 GB of activations per stream and 48-step chunk (estimate) → 2–4 streams on the M1, more on Colab. Each epoch sees every target and every negative pixel, so fewer epochs may suffice.
- Consequences: shuffling only across streams (a batch holds correlated days); no oversampling of positives or hard negatives (focal loss only); flips fixed per stream; tiles need overlap, because fires entering a tile look new at its edge (e.g. 48 × 48 tiles, loss on the inner 32 × 32). The readout lag becomes a free parameter (6 h = early detection, 24–48 h = reconstruction) without changing the input, which unifies the two framings of 6.4. Inference = run each episode from its start, as in training (no warm-up problem).
- Risks: TBPTT only learns dependencies within a chunk (old information flows forward, but the model is never rewarded for *storing* something longer than a chunk); in 3b one forget gate has to serve both the 15-min dynamics and the multi-day memory, so the effective memory will probably stay short; 3a has no gate, so the memory must be re-encoded every period (vanishing, drift).

**Option 4 — Two timescales with a gated long-term memory (long-term gate). Recommended main design.**
- *Fast path*: v2 unchanged (24 h window, reset per period); keeps everything that works.
- *Slow memory* M (C_M channels × H × W), updated once per period by a convolutional GRU (Ballas et al., 2016):
  z_k = σ(Conv([M_{k−1}, s_k])), r_k = σ(Conv([M_{k−1}, s_k])), M̃_k = tanh(Conv([r_k ⊙ M_{k−1}, s_k])), M_k = (1 − z_k) ⊙ M_{k−1} + z_k ⊙ M̃_k.
  The update gate z is the long-term gate: z ≈ 0 keeps the old memory (new information discarded), z ≈ 1 overwrites it. With its own slow clock, the fast LSTM gates no longer have to do both jobs.
- *Writer* s_k (what is offered to the memory, per period): the final hidden state of the fast path, plus hand-made per-period maps (max FRP, number of detections, first and last detection time, FRE), so that the raw facts reach the memory even if the fast state discards them.
- *Reader* (how period k uses M_{k−1}): (i) at the readout only, head on [h_T, M_{k−1}]; (ii) as extra input channels at every fast step; (iii) as initial fast state h_0 = Conv(M_{k−1}). (ii) is the most expressive: each new hotspot is interpreted in the light of the past.
- **Key trade-off and its solution:** if the writer depends on the memory (the fast path of k − 1 reads M_{k−2}, …), the K periods must run one after the other with gradients (~K× cost and memory). Separate the two: the writer is memory-free, so the summaries s_k are independent per period (parallel, cacheable); only the reader of the current period reads the memory. Gradients flow through the reader and the K cheap GRU steps. Transformer-XL (Dai et al., 2019) uses the same idea: a cached, gradient-stopped memory of past segments.
- *Spatial reach*: a 3 × 3 GRU update widens the memory's context by only 1 px per period, slower than multi-day spread (several px per day) → dilated convolutions or a memory at half resolution (stride 2).
- *Gate variants* (ablation, rising complexity):
  - G0 no gate (z = 1; = Option 3a): the baseline for the gate.
  - G1 learned soft gate (ConvGRU as above).
  - G2 event-triggered: z_k = a_k ⊙ σ(·) with a_k = "detection within r px during period k"; the memory changes only where something happened, elsewhere it is kept or decays with a learned rate γ^Δt (as in GRU-D; Che et al., 2018). No drift in quiet periods; inactive periods can be skipped.
  - G3 sparse gate: L1 penalty on z, or a binary gate with a straight-through estimator (Skip RNN; Campos et al., 2018).
  - G4 several memory groups with different fixed rates or clocks (Clockwork RNN, Koutník et al., 2014; Phased LSTM, Neil et al., 2016).
  The gate maps z_k are a direct explainability output (objective v): where and when the model decided that something was worth remembering.
- **Training = samples with lookback (the batching proposal).** A sample = (crop, target period k, lookback K): the current 24 h window plus the summaries of the K previous periods at the same crop. Positives and hard negatives exactly as now (one crop per target, as many hard negatives, redrawn every epoch), so class balance, shuffling and the epoch definition stay. **Chronology is needed only inside a sample (K periods back, trivial on the regular 12 h period grid), not across batches.** The memory starts at zero K periods back (or at the episode start if that is closer); K is drawn at random (e.g. 2–14 periods = 1–7 d, sometimes longer) so that the model learns to handle memories of any age. Loss on the current period (or on the last few). Summaries are computed on the full grid and then cropped (no tile-edge effects in the memory).
- *Flips*: cached learned summaries are not exactly flip-equivariant → cache the four flip variants, flip only the hand-made summaries, or switch off flips for the memory stage.
- *Inference*: carry M through the season, per period one writer pass + one GRU step + one reader pass (constant cost, as in operational use). Check length generalisation: performance and memory statistics vs. the age of the episode.
- *Cost* (relative to v2, 4.4 min per epoch at 15 min on the M1):
  - writer cached (frozen v2 + hand-made maps): one-off full-grid pass over the ~5,300 train + val periods with detections ≈ 20–25 min (0.25 s per period, as in validation); storage at 32 channels fp16 ≈ 1.2 MB per period ≈ 6 GB (memory-mapped; half with 16 channels); training ≈ **1.1–1.2× v2** (K ConvGRU steps on a crop next to 96 ConvLSTM steps);
  - writer recomputed per sample without gradient: + K inference windows ≈ +K/5 × v2 (K = 6: ~2×, K = 14: ~4×);
  - end-to-end (gradients through all K writers): ~K× time and activation memory → only as a short fine-tune, with hourly steps or a small K.
- *Stages*: (1) fast path = v2 (existing checkpoint); (2) freeze the writer, cache the summaries, train memory + reader, with the reader initialised from v2 and the memory branch initialised to zero, so that training starts exactly at v2 and any gain is the memory's; (3) optional joint fine-tune with a short K.

**Option 5 — Memory bank with attention (explicit recall).**
- Keep the summaries of the last N periods (e.g. 14 = 7 d) per pixel; the current period queries them with per-pixel attention over time (keys/values from s_{k−N}, …, s_{k−1} plus an embedding of their age, query from the current fast state). Comparable to the recall gate of E3D-LSTM (Wang et al., 2019).
- Pros: direct access to every past period (no vanishing gradients, no drift); quiet periods are simply not attended to; attention weights show which past day mattered. Cons: fixed horizon N, more parameters. Same batching and cache as Option 4.
- Natural bridge to the planned CNN + Transformer (objective ii): attention over 15-min steps inside the window and over period summaries across days.

**Other information that acts as memory (complements, not alternatives)**
- *Longer, multi-resolution history*: the last 6 h at 15 min plus the preceding 42–66 h at hourly or 3-hourly steps; +40–100 % steps. Simple, but a fixed horizon and a worse single-readout bottleneck (6.1.1).
- *Season-scale burn memory*: cumulative FRE or detections per pixel in the last 30 d or since the start of the season (burnt pixels have less fuel left); Option 1 with τ = 30 d.
- *Satellite climatology*: per-pixel detection frequency of the training years (static prior; training years only).
- **Cloud mask (SEVIRI CLM, EUMETSAT)**: with a memory, "no detection after a fire" is ambiguous (extinguished or hidden), and the rekindle vs. reappearance question above depends on it. The most important companion input for any memory option; to ask the supervisor.
- Never: ICNF labels, earlier ICNF ignitions or burned areas of the current season (6.4).

**Comparison**

| Option | Carried over | Clock | Chronology needed in training | Batching vs. v2 | Cost per epoch (× v2) | Main risk |
|---|---|---|---|---|---|---|
| 1 hand-made channels | EMAs, time since last detection | per step | none | unchanged | ~1.0 | fixed, not learned |
| 2 own prediction | heatmap of k − 1 | 12 h | 1-period lookback per sample | unchanged | ~1.2 | weak memory |
| 3a latent loop (no gate) | learned map | 12 h | lookback or streams | unchanged or streams | 1.2–7 | drift, vanishing |
| 3b stateful ConvLSTM | (h, c) | 15 min | full: episodes, ordered streams, TBPTT, resets | parallel streams, full grid / fixed tiles, no oversampling | ~7 (hourly ~2) | one gate for two timescales; horizon = chunk |
| **4 two timescales + gate** | slow map M | 12 h | K-period lookback per sample | unchanged (+ cached summaries) | **1.1–1.2** cached, 2–4 recomputed | frozen writer; length generalisation |
| 5 memory bank + attention | last N summaries | 12 h | N-period lookback per sample | unchanged (+ cache) | ~1.2 | fixed horizon N |

**Recommendation and order of experiments**
0. The diagnostic above as a script; repeat it for every memory model (recall and false alarms by context class).
1. Option 1 (τ = 6/24/72/168 h + time since the last detection) on v2: the cheapest test. If the "pause > 6 h" class does not improve, check the cloud mask before building a learned memory.
2. Option 4 with a frozen v2 writer + hand-made maps: gate G0 vs. G1 vs. G2, reader (i) vs. (ii), K random 2–14. The main contribution; it must beat Option 1 to show that a learned memory adds more than fixed features.
3. Option 5 if Option 4 helps (lead-in to the Transformer).
4. Option 3b only as a reference for continuous operation, if time allows (most expensive, least flexible training).
Option 2 as a small add-on to the chosen model. Several seeds per configuration (AP @ 5 km fluctuates, 6.5); same split.

**Evaluation additions**
- Recall, AP and false alarms by context class (fresh / pause > 6 h / active in history); the expected gain is in "pause > 6 h", above all fewer false alarms.
- Performance vs. day of the episode (memory should help more later in an episode).
- Rekindle subset: targets with activity within 2 px 6–48 h before.
- Gate and attention maps for a validation case (2016-08-07/08; 2017-10-15 is in the locked test split).
- Implementation check for stateful and memory models: identical outputs whether an episode is processed in one go or in chunks.

**Pitfalls**
- Leakage: memory only from satellite data before the readout. The summary of period k includes its 6 h tail (= the first 6 h of k + 1): fine for k + 1, never for k. Cached learned summaries must come from a writer trained on the training years only (v2 is).
- Split boundaries: episodes do not cross years (1 January is off-season); the memory for validation and test starts empty, as in deployment.
- Persistent off-season pixels keep episodes alive and fill the memory → exclude them when cutting episodes; give the memory the persistent flag.
- Inactive periods (about half of all periods have no detection in Portugal in their window): skip them (with decay γ^Δt in G2) instead of running empty windows.
- A fire ignited in the tail of k − 1 is in the memory at k: no label change needed (it belongs to k − 1), but the model must not predict it again in k (Option 2 helps).

### 6.7 Ablation study: plan and tooling (2026-10-20)

**Protocol.** Every experiment changes *one* setting of the reference `ref` (= v1: hourly steps, 7 channels, 2 × 32 ConvLSTM, last-state readout, focal loss, σ = 1 px, ignore radius 2 px, targets ≥ 50 ha, 32 × 32 crops, 1 hard negative per positive, flips, 20 epochs, AdamW 1e-3). ≥ 3 seeds per experiment (AP @ 5 km varies by ±0.03–0.05 between epochs alone, single runs cannot separate small effects); reported as mean ± std of (a) the best validation epoch (= saved checkpoint) and (b) the mean of the last five validation evaluations; plus recall for isolated / non-isolated fires, median error, F1. Validation 2015–2016 only; the test split stays locked until the final configuration is chosen. Once 15-min steps are confirmed (`ref15`), later ablations can switch their reference to v2.

**Experiments** (`src/run_ablations.py`, names → `train.py` options):

| Tier | Name | Change | Question |
|---|---|---|---|
| 1 | `ref` | — | reference (v1) |
| 1 | `ref15` | `--step-agg 1` | is the 15-min gain real (6.5)? |
| 1 | `births_plus` | + `births` channel | does the hand-made clustering (nucleus births, 3.2) add information to the raw input? |
| 1 | `births_only` | `births` instead of FRP/confidence/detected | how good is the clustering alone, with a learned model on top? (core argument of 4.0) |
| 1 | `no_time` / `no_static` / `no_conf` / `no_frp` / `presence_only` | drop period+time / mainland+persistent / confidence / FRP / FRP+confidence | which inputs matter; is FRP intensity informative or only presence? |
| 1 | `hist0` / `tail0` | `--history-h 0` / `--tail-h 0` | do the 6 h before / after the period help? |
| 2 | `layers1` / `layers3` | 1 / 3 ConvLSTM layers | is the second layer justified (6.1.2)? |
| 2 | `hidden16` / `hidden64` | 16 / 64 channels | capacity vs. overfitting (~1,900 targets) |
| 2 | `kernel5` | 5 × 5 kernels | more context per step vs. depth |
| 2 | `readout_max` / `readout_mean` | head on max / mean of the top hidden state over all steps | is the single final readout a bottleneck? |
| 2 | `cnn3d` | 3D CNN, no recurrence | is the recurrent memory needed at all? |
| 2 | `crop16` / `crop64` | training crop size | how much spatial context is needed? |
| 3 | `sigma05` / `sigma2` | target σ 0.5 / 2 px | sharper peaks → fewer merged ignitions (6.3)? |
| 3 | `loss_bce` / `loss_mse` | weighted BCE / MSE | is the CenterNet focal loss the right choice? |
| 3 | `neg0` / `neg05` / `neg2` | 0 / 0.5 / 2 hard negatives per positive | do "fire but no ignition" crops reduce false alarms? |
| 3 | `ignore0` / `ignore4` | ignore radius 0 / 4 px (training) | effect of the ignore regions |
| 3 | `target100` | targets ≥ 100 ha only (smaller → ignore) | fewer, clearer labels vs. more, weaker ones |
| 3 | `noflip` | no flips | does the augmentation help? |
| — | `src/peak_ablation.py` | peak window 3/5, threshold 0.02/0.05/0.1, sub-pixel on/off | post-processing only, no retraining |

Not ablations but extensions (later, 6.4/6.6): long-history summary channels, longer windows, stateful/long-term memory model, covariates, cloud mask, CNN + Transformer.

**Code changes for the ablations.**
- `src/model.py`: `build_model(cfg)`; `IgnitionConvLSTM` with configurable layers/width/kernel and readout (`last` = v1/v2, `max`, `mean`); new non-recurrent `TemporalCNN3D` (3 × Conv3d, time pooled by max and mean); losses `focal` (v1), `bce`, `mse`.
- `src/dataset.py`: `sample(..., sigma, ignore_r, min_target_ha, births)`; per-detection nucleus-birth flag for the `births` channel (first time step of each nucleus, 3.2). Defaults reproduce the v1/v2 samples.
- `src/train.py`: all settings as options (`--channels`, `--history-h`, `--tail-h`, `--arch`, `--layers`, `--hidden`, `--kernel`, `--readout`, `--crop`, `--sigma`, `--ignore-r`, `--min-target-ha`, `--loss`, `--neg-ratio`, `--no-flip`, `--workers`); `DEFAULTS` = v1; `config_from()` fills missing keys of older checkpoints, so v1/v2 load unchanged (re-evaluation of v1 through the new code reproduces AP 0.302 / 0.664 exactly). Positive-pixel threshold of the loss scales with σ (`pos_threshold`, = 0.75 for σ ≥ 1 px as in v1/v2). Run names may contain a folder (`ablations/ref_s0`). History written after every epoch; `<run>_done.json` marks a finished run. DataLoader workers get their own random streams. Validation records also recall for isolated / non-isolated fires.
- `src/evaluate_model.py`, `src/plot_predictions.py`: use `load_model()` and the stored configuration.
- `src/run_ablations.py`: experiment list (above), runs experiment × seed via `train.py` into `models/ablations/`, `outputs/ablations/`; skips finished runs (resumable); `--smoke` (1 epoch, 3 batches, 10 validation periods) and `--dry-run`.
- `src/summarize_ablations.py`: mean ± std per experiment, Δ vs. `ref` → `outputs/ablations/summary.{csv,md}`, `figures/ablations/summary.png`.
- `src/peak_ablation.py`: post-processing grid on one checkpoint.
- `colab/ablations_colab.ipynb`: same setup as `colab/train_colab.ipynb` (clone from GitHub, `models/` and `outputs/` on Drive), then dry run → smoke test → runs → summary → peak ablation. Resumable after a disconnect.

**Cost.** Locally (M1) an hourly run ≈ 35 min, `ref15` ≈ 2 h. Tier 1 with 3 seeds = 33 runs (~23 h locally incl. `ref15`); Tiers 1–3 = 96 runs. On Colab the time per run is to be measured with the smoke test; a free session lasts several hours at most, so the runs will span several sessions (the runner resumes).

**Validation speed (2026-10-20).** Profile of one validation of v1 on the M1 (844 periods): sample building 3 s, features 4 s, **model 52 s**, peaks 0.5 s, scoring 0.5 s. One full-domain period (177 × 106 px) costs about as much compute as one training batch (16 × 32 × 32 px), so a validation is comparable to an epoch of training (Colab T4: 0.49 min training vs. 1.35 min validation per epoch). Changes: `predict_maps()` runs `--val-batch` periods per model call (default 16; identical scores — v1 0.3023 / 0.6644, v2 0.3482 / 0.6722 at batch 1 and 16; M1 speed-up only 1.3×: 58 → 45 s, 222 → 171 s); optional `--val-amp` = half-precision validation on CUDA (expected 2–3× on a T4, scores not bit-identical, untested on CUDA at the time of writing). If still too slow, `--val-every 4`. Whatever is chosen must be used for **all** ablation runs.

**Ablation run settings (fixed 2026-10-20).** All ablation runs on Colab use `EXTRA = "--workers 2 --val-amp"` (2 data-loading workers, half-precision validation on the GPU), validation every 2 epochs, `--val-batch 16`. Measured on the Colab T4 for `ref_s0`: 0.48 min training per epoch, 0.73 min per validation (1.35 min before batching/AMP) → ≈ 17 min per hourly run, `ref15` ≈ 60 min (estimate). Epoch-2 scores of `ref_s0` (AP@5 0.182, AP@10 0.524, recall@10 0.82, median error 3.4 / 4.7 km) are in the range of v1 at epoch 2 (0.164 / 0.531), so half-precision validation does not visibly distort the scores. Consequence: ablation results are compared with each other and with `ref` from the same batch, not directly with the v1/v2 numbers (full-precision validation). Estimated cost with 3 seeds: Tier 1 ≈ 11–12 h, Tiers 1–3 ≈ 28–30 h.

**Suggested order.** (1) `ref`, `ref15` (+ the peak ablation, no training); (2) rest of Tier 1 (`births_*`, channels, history/tail); (3) Tier 2 (`layers*`, `readout_*`, `cnn3d` first); (4) Tier 3 (`sigma*`, `neg*`, `target100` first); then fix the configuration, extensions, retrain on 2004–2016, single test evaluation.

### 6.8 Ablation results, session A1 (2026-10-20)

Colab T4, `EXTRA = "--workers 2 --val-amp"`, seeds 0–2, validation 2015–2016; summary `outputs/ablations/summary_A1.md` (copied from the Colab run). Mean ± std over 3 seeds; "best" = best validation epoch (saved checkpoint), "last5" = mean of the last five validation evaluations (epochs 12–20).

| Experiment | AP @ 5 km best | AP @ 5 km last5 | AP @ 10 km best | AP @ 10 km last5 |
|---|---|---|---|---|
| `ref` (v1 configuration) | 0.376 ± 0.030 | 0.265 ± 0.033 | 0.657 ± 0.022 | 0.668 ± 0.009 |
| `births_plus` (raw + nucleus-birth channel) | 0.389 ± 0.039 | 0.311 ± 0.015 | 0.677 ± 0.005 | 0.678 ± 0.011 |
| `births_only` (births instead of raw hotspots) | 0.123 ± 0.046 | 0.091 ± 0.027 | 0.295 ± 0.012 | 0.292 ± 0.020 |

Rough Welch t-values (n = 3 per group, only indicative): `births_plus` vs. `ref`: AP@5 best 0.5, AP@5 last5 2.2, AP@10 best 1.5, AP@10 last5 1.2; `births_only` vs. `ref`: AP@10 best −25.

Interpretation:
- **Raw input is essential; birth locations alone are not enough.** `births_only` is far below `ref` (AP@10 0.30 vs. 0.66) and even below the non-learned nucleus-birth baseline (0.24 / 0.57, 5.3). That baseline ranks births by the fire radiative energy of the whole nucleus in the input window; `births_only` sees only *where* nuclei start, not how strong they become, so it cannot tell the few large fires from the many small ones. Hence `births_only` is a "birth locations only" model, not a fair "clustering-only" comparison — the fair one remains the baseline, which `ref` clearly beats (AP 0.38 / 0.66 vs. 0.24 / 0.57).
- **The clustering adds at most a little on top of the raw input.** `births_plus` is higher on every measure (+0.01 to +0.05 AP), most clearly in the last-five mean at 5 km (0.311 vs. 0.265, t ≈ 2.2) and with lower spread, but none of the differences is conclusive with 3 seeds. Keep as a candidate for the final configuration; it uses satellite data only, so no leakage.
- **Large best-vs-last5 gap at 5 km** (`ref`: 0.376 best vs. 0.265 last5) while AP @ 10 km is stable (0.657 vs. 0.668): the exact peak position varies strongly from epoch to epoch, and choosing the best of ten validation epochs inflates AP @ 5 km. The last-five mean is the more honest comparison; checkpoint averaging (EMA/SWA) or a decaying learning rate at the end could stabilise it (idea for later).
- `ref` here (0.376 / 0.657) is higher than v1 (0.302 / 0.664, seed 0, full-precision validation on the M1). Not directly comparable (seeds, half-precision validation, Colab GPU); within the ablation batch all runs share the same settings.


### 6.9 Ablation results, session A2 and peak extraction (2026-10-21)

**`ref15` (native 15-min steps)** — summary `outputs/ablations/summary_A2.md`; **only 2 of 3 seeds finished**:

| Experiment | AP @ 5 km best | AP @ 5 km last5 | AP @ 10 km best | AP @ 10 km last5 | min/epoch |
|---|---|---|---|---|---|
| `ref` (hourly, 3 seeds) | 0.376 ± 0.030 | 0.265 ± 0.033 | 0.657 ± 0.022 | 0.668 ± 0.009 | 0.50 |
| `ref15` (15 min, 2 seeds) | 0.327 ± 0.012 | 0.259 ± 0.075 | 0.649 ± 0.052 | 0.661 ± 0.028 | 1.97 |

Rough Welch t (`ref15` − `ref`): AP@5 best −2.5, AP@5 last5 −0.1, AP@10 best −0.2.
- **No evidence that 15-min steps help.** The last-five means are practically equal (0.259 vs. 0.265 at 5 km, 0.661 vs. 0.668 at 10 km); at the best epoch `ref15` is even lower at 5 km. The spread between seeds is large (AP@5 last5 ± 0.075, AP@10 best ± 0.052).
- The single-seed advantage of v2 over v1 (6.5: AP@5 0.348 vs. 0.302) was within the seed/epoch noise — exactly the risk noted there. Lesson for the thesis: single runs are not enough to compare configurations.
- Cost: ~4× training time (1.97 vs. 0.50 min/epoch).
- **Decision: hourly steps stay the reference** for all further ablations and the final configuration (same accuracy, a quarter of the cost). The third `ref15` seed would only matter if it were far above the other two; optional.

**Peak-extraction ablation** (`src/peak_ablation.py`, checkpoint `convlstm_v1`, full-precision validation on the Colab GPU; `outputs/convlstm_v1_peak_ablation.txt`). The default reproduces v1 exactly (0.302 / 0.664).

| Window | Threshold | Sub-pixel | Points | AP @ 5 km | AP @ 10 km | Recall @ 10 km | Median error @ 5 km |
|---|---|---|---|---|---|---|---|
| **3** | 0.02 / **0.05** / 0.1 | **yes** | 42,200 / 4,144 / 3,110 | **0.302** | **0.664** | **0.871** | 3.14 km |
| 3 | 0.02 / 0.05 / 0.1 | no | (same) | 0.275 | 0.654 | 0.866 | 3.09 km |
| 5 | 0.02 / 0.05 / 0.1 | yes | 42,196 / 3,961 / 2,976 | 0.293 | 0.654 | 0.848 | 3.16 km |
| 5 | 0.02 / 0.05 / 0.1 | no | (same) | 0.266 | 0.644 | 0.843 | 3.16 km |

- **Threshold has no effect** on any metric (0.02–0.1): the extra low-probability peaks are all false alarms ranked last, so AP and recall do not change; 0.05 keeps the point list ten times shorter than 0.02.
- **Sub-pixel refinement helps:** +0.027 AP @ 5 km, +0.010 AP @ 10 km.
- **A 5 × 5 window is worse** (−0.009 AP @ 5 km, −0.023 recall @ 10 km): it suppresses more nearby peaks, i.e. merges more close ignitions.
- **Decision: keep the default** (3 × 3, threshold 0.05, sub-pixel). The merging of close ignitions (6.3) is not a post-processing issue; it has to be addressed in training (σ, Tier 3).

Next (changed 2026-10-21): **Phase D first** (Tier 3: D1 `sigma05`, `sigma2`, `neg0`, `neg2`, `target100`; D2 `loss_bce`, `loss_mse`, `neg05`, `ignore0`, `ignore4`, `noflip`), compared with the existing `ref` runs. Phases B (inputs/window) and C (architecture) postponed, still open.

### 6.10 Ablation results, session D1 — run locally (2026-10-22)

Colab time ran out; D1 and the following sessions run on the M1 with `src/run_ablations.py --folder ablations_local` (option added 2026-10-21), no `--workers`, full-precision validation (`--val-amp` only works on CUDA). Because hardware and validation precision differ from Colab, **`ref` was re-run locally** and local runs are only compared with the local `ref`. Summary: `outputs/ablations_local/summary.{md,csv}`; 3 seeds each; ~32 min per run.

| Experiment | AP @ 5 km best / last5 | AP @ 10 km best / last5 | Recall @ 10 km non-isolated (best) | Median error @ 5 km (best) | min/epoch |
|---|---|---|---|---|---|
| `ref` (local) | 0.346 ± 0.060 / 0.309 ± 0.039 | 0.667 ± 0.016 / 0.665 ± 0.016 | 0.826 | 3.11 km | 1.19 |
| `sigma05` | 0.390 ± 0.072 / 0.342 ± 0.059 | 0.674 ± 0.031 / 0.669 ± 0.026 | 0.861 | 2.81 km | 1.19 |
| `sigma2` | 0.327 ± 0.030 / 0.291 ± 0.016 | 0.637 ± 0.008 / 0.637 ± 0.007 | 0.771 | 3.16 km | 1.19 |
| `neg0` | 0.358 ± 0.043 / 0.273 ± 0.044 | 0.644 ± 0.025 / 0.650 ± 0.012 | 0.822 | 3.22 km | 0.60 |
| `neg2` | 0.406 ± 0.024 / 0.301 ± 0.036 | 0.683 ± 0.028 / 0.686 ± 0.011 | 0.844 | 2.91 km | 1.78 |
| `target100` | 0.381 ± 0.030 / 0.299 ± 0.030 | 0.654 ± 0.009 / 0.662 ± 0.020 | 0.821 | 3.14 km | 0.76 |

Rough Welch t vs. `ref` (AP@5 best / last5, AP@10 best / last5): `sigma05` +0.8 / +0.8 / +0.3 / +0.2; `sigma2` −0.5 / −0.7 / −3.0 / −2.9; `neg0` +0.3 / −1.0 / −1.4 / −1.3; `neg2` +1.6 / −0.3 / +0.8 / +1.9; `target100` +0.9 / −0.3 / −1.3 / −0.2.

Interpretation:
- **Target width σ matters, narrower is better.** σ = 2 px is clearly worse at 10 km (t ≈ −3) and finds fewer non-isolated fires (0.77 vs. 0.83): wide targets make the model blur close ignitions together. σ = 0.5 px is better on every measure (AP @ 5 km +0.03–0.04, recall non-isolated +3.5 points, median error −0.3 km). Not significant with 3 seeds (large spread), but consistent in direction and with the merging problem of 6.3 — the most promising change so far.
- **Hard negatives help a little.** Without them (`neg0`) AP @ 10 km drops by ~0.02 (t ≈ −1.4), at half the training time. With two per positive (`neg2`) AP @ 10 km is the highest (last5 0.686, t ≈ 1.9); caveat: `neg2` also trains on 1.5× more crops per epoch, so part of the gain may be more training.
- **Targets ≥ 100 ha only: no gain** (AP @ 10 km slightly lower) → keep ≥ 50 ha (more labels).
- The local `ref` differs from the Colab `ref` (AP@5 last5 0.309 vs. 0.265) — another reason to compare only within one machine.

Next: D2 (`loss_bce`, `loss_mse`, `neg05`, `ignore0`, `ignore4`, `noflip`) locally; 2 extra seeds (3, 4) for `ref`, `sigma05`, `neg2`; then a combination run (σ = 0.5 + 2 hard negatives, possibly + births channel) as candidate final configuration; possibly σ = 0.75.

### 6.11 Ablation results, session D2 — and a systematic SEVIRI position offset (2026-10-23)

Local runs (M1, `--folder ablations_local`, 3 seeds, compared with the local `ref`); summary `outputs/ablations_local/summary_D1_D2.md`.

| Experiment | AP @ 5 km best / last5 | AP @ 10 km best / last5 | Recall @ 10 km (non-isol.) | Median error @ 5 km | t vs. `ref` (AP5b / AP5l / AP10b / AP10l) |
|---|---|---|---|---|---|
| `ref` | 0.346 ± 0.060 / 0.309 ± 0.039 | 0.667 ± 0.016 / 0.665 ± 0.016 | 0.869 (0.826) | 3.11 km | |
| `loss_bce` | 0.415 ± 0.052 / 0.343 ± 0.023 | 0.668 ± 0.007 / 0.667 ± 0.007 | 0.812 (0.767) | 3.02 km | +1.5 / +1.3 / +0.1 / +0.2 |
| `loss_mse` | 0.428 ± 0.014 / 0.347 ± 0.012 | 0.655 ± 0.011 / 0.660 ± 0.008 | 0.810 (0.764) | 3.02 km | +2.3 / +1.6 / −1.1 / −0.5 |
| `neg05` | 0.337 ± 0.039 / 0.283 ± 0.028 | 0.666 ± 0.022 / 0.670 ± 0.013 | 0.871 (0.830) | 3.15 km | −0.2 / −0.9 / −0.1 / +0.4 |
| `ignore0` | 0.385 ± 0.071 / 0.324 ± 0.052 | 0.682 ± 0.015 / 0.675 ± 0.010 | 0.873 (0.833) | 3.00 km | +0.7 / +0.4 / +1.1 / +0.9 |
| `ignore4` | 0.328 ± 0.021 / 0.286 ± 0.025 | 0.653 ± 0.029 / 0.649 ± 0.013 | 0.854 (0.808) | 3.09 km | −0.5 / −0.9 / −0.8 / −1.4 |
| **`noflip`** | **0.615 ± 0.012 / 0.580 ± 0.045** | 0.690 ± 0.018 / 0.657 ± 0.039 | 0.896 (0.864) | **1.79 km** | **+7.6 / +7.8** / +1.7 / −0.3 |

- `loss_bce` / `loss_mse`: higher AP @ 5 km but lower recall (0.81 vs. 0.87): sharper, more confident peaks, more fires missed. AP @ 10 km unchanged. A trade-off, not a clear win.
- `neg05`: no change; `ignore0` slightly better, `ignore4` slightly worse — the ignore regions do not help training (not significant).
- **`noflip`: AP @ 5 km almost doubles, median error 3.1 → 1.8 km** (t ≈ +7.6). Flips mirror inputs and targets consistently (checked), so this is not a bug: there is a *directional* pattern that flipping destroys.

**Cause: a systematic offset between SEVIRI and ICNF positions.** For all 2,705 targets: vector from the nucleus birth (first SEVIRI detection, 3.2) to the ICNF ignition point = **+1.49 ± 0.06 km east, −2.95 ± 0.07 km north** (median +1.72 / −3.12; |mean| 3.3 km). The real spread of the fires is small in comparison (nucleus centroid − birth: +0.03 east, −0.39 north). Median offset by year:

| Years | East | North |
|---|---|---|
| 2004–2017 (each year) | +1.4 … +2.1 km | −2.7 … −3.5 km |
| 2018 | +0.21 km | −0.58 km |
| 2019 | −0.10 km | −0.75 km |
| before 2017-12-06 (n = 2,581) | **+1.79 km** | **−3.19 km** |
| from 2017-12-06 (n = 124) | +0.10 km | −0.74 km |

- A step change at the end of 2017. The difference before − after (≈ +1.7 km east, −2.45 km north) is **about half a SEVIRI pixel in both directions** (pixel ≈ 3.2 × 4.5 km over Portugal).
- Hypothesis (from memory, **to be verified against EUMETSAT documentation**): EUMETSAT corrected a half-pixel georeferencing offset of SEVIRI Level 1.5 data in December 2017; the FRP extract's lat/lon before that date would then be shifted by ~half a pixel to the north-west.
- `noflip` learns "move the prediction ~3 km south-south-east" — correct for 2004–2017 data, but it would be **wrong for 2018–2019** (part of the test split). So `noflip` is not a fix; it exploits a data artefact.

Consequences (to be worked through):
1. **Most of the "first hotspot is ~4 km from the ignition" error so far is this offset**, not fire physics or label noise (2.1, 3.2, 5.3). After correction, the baseline and the models should become much more precise; the sub-pixel question (4.2) needs to be re-assessed.
2. **All km-based results so far are affected** (baselines, v1/v2, ablations): they compare against ICNF with a ~3 km bias before 2018. Relative comparisons between ablations of the same batch stay meaningful, except for anything that interacts with direction (flips) or with the peak precision (σ, losses at 5 km).
3. Fix in the data, not in the model: correct the SEVIRI positions before the change date (once the size and date are confirmed), rebuild the sample index, re-run the baselines and a new `ref`, then re-check `noflip` (should lose its advantage) and `sigma05`.

**Verification of the offset (2026-10-23).**
- satpy SEVIRI Level 1.5 reader documentation (https://satpy.readthedocs.io/en/latest/api/satpy.readers.seviri_l1b_native.html): "Until December 2017, the data is shifted by 1.5km SSP North and West against the nominal GEOS projection." Since December 2017 the offset is corrected; a flag in the data indicates whether the correction was applied. Same statement and an illustration of the back-shift (to the south-east) in the satpy developer memo (https://github.com/pytroll/satpy/wiki/SEVIRI-georeferencing-offset-correction).
- EUMETSAT service notices (relayed by NOAA, seen only in search results, pages not reachable): a new Image Processing Facility release correcting the geo-referencing offset of SEVIRI Level 1.5 data, planned for March 2017, then postponed to 2017-09-20. **The exact operational day in December 2017 was not found** (the "6 December" in 6.11 was from memory and is unconfirmed).
- 1.5 km at the sub-satellite point = half a native SEVIRI pixel (sampling distance 3 km). Measured in native pixels (ICNF ignition − nucleus birth, targets): before Dec 2017 +0.46 px east, −0.69 px north (n = 2,581); after +0.00 px east, −0.16 px north (n = 124); **difference +0.46 px east, −0.52 px north = the documented half-pixel shift**, measured independently from fire records. The remaining −0.16 px (≈ 0.7 km south) is present in both periods (fire behaviour or ICNF bias, not geolocation).
- The exact day hardly matters here: between 2017-10-16 and 2018-02-28 there are only 2 target fires (2017-11-22, 2018-02-25) and 1,056 detections in Portugal.

**Planned correction.** Correct the labels, not the detections: detections stay on the exact native grid; for periods before the change date (setting, default 2017-12-01) the ICNF ignition positions are moved half a pixel north-west into the shifted sensor frame, and predictions / nucleus births of those periods are moved half a pixel south-east before scoring against ICNF. Then re-run the baselines, a new `ref` and the key ablations (`noflip`, `sigma05`).

### 6.12 Geo-referencing correction implemented (2026-10-23)

Code:
- `src/grid.py`: `GEOREF_CHANGE = "2017-12-01"` (first day treated as corrected; exact day undocumented, see 6.11), `GEOREF_SHIFT_PX = 0.5`; `true_to_sensor_ij/latlon`, `sensor_to_true_ij/latlon` (sensor = true + half a pixel north-west before the change).
- `src/build_dataset.py` (default on, `--no-georef-fix` reproduces the old build): detections stay on their native pixels; `labels.parquet` holds the ignition position in the sensor frame of each period; the "seen" test compares ICNF and SEVIRI in the same frame; `fires.parquet` keeps the true positions; `build_info.json` records the setting. `src/dataset.py` reads it (`FireData.georef_fix`).
- `src/train.py` (`maps_to_points`), `src/baselines.py`, `src/peak_ablation.py`: predictions / nucleus births of periods before the change are moved half a pixel south-east (true frame) before scoring against ICNF.
- Not changed: the exploratory analyses of sections 2–4 (time zone check, proximity, clustering diagnostics, tiling design) still compare uncorrected positions; their qualitative conclusions stand, distances in km there include the offset.

Rebuilt sample index (`outputs/build_dataset_log.txt`): **2,821 targets** (before: 2,705; +116 fires whose first detection had been pushed outside the 5 km "seen" radius by the offset), 8,324 ignore entries; train / val / test targets 1,979 / 403 / 439. All checks pass (`outputs/dataset_check_log.txt`). Label − nucleus birth in the sensor frame, median: before the change −0.01 px east, −0.22 px north (n = 2,697); after +0.00 / −0.16 px (n = 124) → **the offset is gone**.

Nucleus-birth baseline on the corrected data (validation 2015–2016; uncorrected numbers in `outputs/baselines_log_uncorrected.txt`):

| | uncorrected | **corrected** |
|---|---|---|
| target ignitions | 389 | 403 |
| AP @ 5 km | 0.237 | **0.448** |
| AP @ 10 km | 0.569 | 0.577 |
| recall @ 5 km (all predictions) | 0.51 | **0.71** |
| recall @ 5 km isolated / non-isolated | 0.72 / 0.45 | **0.92 / 0.65** |
| median error of matches @ 5 / 10 km | 3.6 / 4.3 km | **2.2 / 2.3 km** |
| nearest prediction ≤ 5 km / median distance | 53 % / 4.8 km | **73 % / 2.8 km** |

Consequences:
- **The first SEVIRI detection is ~2.2 km from the ICNF ignition, not ~4 km**; well below one pixel diagonal (~5.5 km). The 3.3 km offset rarely mattered at 10 km but cost most of the precision at 5 km.
- The uncorrected models (v1/v2, all ablations so far) are now below the corrected baseline at 5 km (`ref` AP@5 ≈ 0.35–0.38 vs. 0.45): they learned against shifted labels. **All models have to be retrained on the corrected data**; the earlier results stay valid as documentation of the uncorrected setting and for relative comparisons within their batch.
- Next: new `ref` + `noflip`, `sigma05`, `neg2`, `births_plus` (3 seeds each) on the corrected data in `--folder ablations_georef`.

### 6.13 Ablations on the corrected data (2026-10-23)

Local (M1), `--folder ablations_georef`, 3 seeds each, corrected build (6.12); summary `outputs/ablations_georef/summary_georef1.md`. Corrected nucleus-birth baseline for comparison: AP@5 0.448, AP@10 0.577, recall@10 0.804 (non-isolated 0.753), median error @ 5 km 2.19 km.

| Experiment | AP @ 5 km best / last5 | AP @ 10 km best / last5 | Recall @ 10 km (non-isol.) | Median error @ 5 km | t vs. `ref` (AP5b / AP5l / AP10b / AP10l) |
|---|---|---|---|---|---|
| **`ref`** | **0.604 ± 0.016 / 0.597 ± 0.015** | **0.687 ± 0.003 / 0.681 ± 0.007** | 0.887 (0.853) | 2.08 km | |
| `births_plus` | 0.590 ± 0.047 / 0.577 ± 0.057 | 0.669 ± 0.044 / 0.661 ± 0.050 | 0.883 (0.848) | 2.12 km | −0.5 / −0.6 / −0.7 / −0.7 |
| `sigma05` | 0.604 ± 0.002 / 0.596 ± 0.004 | 0.681 ± 0.011 / 0.678 ± 0.009 | 0.897 (0.867) | 2.08 km | −0.0 / −0.1 / −0.9 / −0.4 |
| **`neg2`** | **0.618 ± 0.009 / 0.614 ± 0.009** | **0.705 ± 0.011 / 0.699 ± 0.009** | 0.891 (0.859) | 2.09 km | **+1.3 / +1.7 / +2.8 / +2.6** |
| `noflip` | 0.633 ± 0.011 / 0.613 ± 0.021 | 0.686 ± 0.007 / 0.672 ± 0.015 | 0.892 (0.861) | 1.83 km | +2.5 / +1.1 / −0.3 / −1.0 |

Interpretation:
1. **The geo-referencing correction is the largest single improvement of the project**: `ref` AP @ 5 km 0.60 (uncorrected 0.35–0.38), well above the corrected baseline (0.45); non-isolated recall 0.85 vs. 0.75 for the baseline.
2. **Training is now stable**: seed spread at 5 km ±0.016 (before ±0.03–0.06), best vs. last5 0.604 vs. 0.597 (before 0.346 vs. 0.309). Much of the earlier epoch-to-epoch instability came from the inconsistent labels (shifted before 2018, not after).
3. **`noflip` lost almost all of its advantage** (+0.03 instead of +0.27 at 5 km, none at 10 km) — confirms 6.11. The remainder fits the small, stable southward offset (~0.2 px) present in both periods (fire behaviour or ICNF positioning); learning it would be legitimate.
4. **`sigma05`'s earlier gain was an artefact of the offset**; no effect on corrected data → keep σ = 1 px.
5. **The clustering adds nothing** (`births_plus` slightly worse, more variable): the model extracts from the raw input what the hand-made clustering provides (supports 4.0).
6. **`neg2` is the only consistent gain** (+0.014–0.018 on all four measures, significant at 10 km, t ≈ 2.7); cost 1.5× training time; part of the gain may be more crops per epoch.

**Status of the earlier ablations:** all uncorrected results (6.8–6.10, 6.11 table) are superseded for anything location-sensitive; their conclusions on σ and flips have reversed. They remain as documentation of the uncorrected setting.

Next: combination run `neg2` (+ optionally `noflip`) as candidate final configuration; on corrected data still open: `layers1`, `hidden16` (also 1.7× faster if accuracy holds), `readout_max`, `cnn3d`, `presence_only`, `hist0`; then fix the configuration, retrain on 2004–2016, single test evaluation.

### 6.14 Remaining ablations on the corrected data (2026-10-24)

Local (M1), `--folder ablations_georef`, 3 seeds each; summary `outputs/ablations_georef/summary_georef2.md` (36 runs, all with the corrected build). t = rough Welch t vs. `ref` (AP5 best / AP5 last5 / AP10 best / AP10 last5).

| Experiment | Change | AP @ 5 km best / last5 | AP @ 10 km best / last5 | Recall @ 10 km non-isol. | min/epoch | t |
|---|---|---|---|---|---|---|
| `ref` | — | 0.604 / 0.597 | 0.687 / 0.681 | 0.853 | 1.25 | |
| `cnn3d` | 3D CNN, no recurrence | 0.628 / 0.617 | 0.700 / 0.691 | 0.868 | 1.19 | +2.4 / +2.3 / +4.9 / +1.6 |
| `hidden16` | 16 instead of 32 channels | 0.619 / 0.612 | 0.699 / 0.696 | 0.859 | 0.79 | +1.3 / +1.5 / +3.4 / +3.1 |
| `layers1` | 1 ConvLSTM layer | 0.611 / 0.604 | 0.691 / 0.689 | 0.857 | 0.81 | +0.6 / +0.7 / +0.5 / +1.0 |
| `readout_max` | head on max over time | 0.609 / 0.602 | 0.702 / 0.696 | 0.861 | 1.30 | +0.5 / +0.5 / +3.2 / +2.5 |
| `neg2` | 2 hard negatives / positive | 0.618 / 0.614 | 0.705 / 0.699 | 0.859 | 1.86 | +1.3 / +1.7 / +2.8 / +2.6 |
| `neg2_noflip` | + no flips | 0.631 / 0.596 | 0.692 / 0.664 | 0.846 | 1.85 | +2.7 / −0.1 / +2.4 / −2.3 (vs. `neg2`: +2.0 / −1.2 / −2.0 / −4.3) |
| `presence_only` | detections only (no FRP, no confidence) | 0.595 / 0.590 | 0.679 / 0.676 | 0.857 | 1.14 | −0.8 / −0.7 / −3.4 / −1.1 |
| `hist0` | no 6 h history | 0.584 / 0.578 | 0.668 / 0.665 | 0.859 | 0.89 | −2.0 / −1.7 / −1.8 / −1.8 |
| `births_plus`, `sigma05`, `noflip` | see 6.13 | | | | | |

Findings:
1. **All design variants lie within ~±0.03 AP of `ref`**: the model is robust to architecture choices; the label correction (6.12) mattered far more than any setting.
2. **The recurrent memory is not needed:** the non-recurrent `cnn3d` is at least as good as the ConvLSTM (slightly better on all four measures) at the same training time. What makes the model work is seeing the space–time window, not the ConvLSTM memory. Supports trying the planned Transformer (direct access to all time steps).
3. **Smaller is better or equal:** `hidden16` (+0.012–0.015, significant at 10 km, 1.6× faster) and `layers1` (equal, 1.5× faster) — the 2 × 32 model was over-sized for ~2,000 targets.
4. **`readout_max` helps at 10 km** (+0.015), not at 5 km: the final-state readout loses some information.
5. **`neg2` confirmed** (+0.014–0.018). **Combining with `noflip` is worse** (less stable, AP @ 10 km last5 −0.035 vs. `neg2`) → keep flips.
6. **FRP intensity adds little:** detections only (`presence_only`) is only ~0.01 lower; the pattern of where and when fires are detected carries most of the information (thesis question on FRP).
7. **The history helps a little:** without it (`hist0`) ~0.02 lower on all measures.
- Caveat: 12 comparisons with 3 seeds, effects 0.01–0.02; single t ≈ 2 values can occur by chance. Consistent patterns (cnn3d, hidden16, neg2, readout_max, hist0) are more reliable than single numbers.

Next (proposed): two candidate final configurations, 3 seeds each — `conv_final` (ConvLSTM, `--hidden 16 --neg-ratio 2 --readout max`) and `cnn3d_final` (`--arch cnn3d --neg-ratio 2`); choose on validation, retrain on 2004–2016, single test evaluation 2017–2019.

### 6.15 Final candidate configurations (2026-10-24)

Local (M1), corrected build, 3 seeds each; summary `outputs/ablations_georef/summary_georef3_final_candidates.md`. (`conv_final_l1` — one layer × 16 channels — was discussed but not run.)

| Validation 2015–2016 | AP @ 5 km best / last5 | AP @ 10 km best / last5 | Recall @ 10 km (non-isol.) | Median error @ 5 km | min/epoch |
|---|---|---|---|---|---|
| baseline (nucleus birth) | 0.448 | 0.577 | 0.804 (0.753) | 2.19 km | — |
| `ref` | 0.604 / 0.597 | 0.687 / 0.681 | 0.887 (0.853) | 2.08 km | 1.25 |
| `hidden16` | 0.619 / 0.612 | 0.699 / 0.696 | 0.891 (0.859) | 2.11 km | 0.79 |
| `neg2` | 0.618 / 0.614 | 0.705 / 0.699 | 0.891 (0.859) | 2.09 km | 1.86 |
| `cnn3d` | 0.628 / 0.617 | 0.700 / 0.691 | 0.898 (0.868) | 2.05 km | 1.19 |
| `conv_final` (ConvLSTM, 16 ch, 2 neg., max readout) | 0.613 ± 0.017 / 0.609 ± 0.016 | 0.696 ± 0.016 / 0.692 ± 0.017 | 0.888 (0.855) | 2.03 km | 1.23 |
| **`cnn3d_final`** (3D CNN, 2 neg.) | **0.635 ± 0.008 / 0.621 ± 0.009** | **0.704 ± 0.001 / 0.696 ± 0.008** | **0.897 (0.867)** | 2.06 km | 1.81 |

Rough Welch t (AP5 best / AP5 last5 / AP10 best / AP10 last5): `cnn3d_final` vs. `ref` +2.9 / +2.4 / +8.4 / +2.4; `cnn3d_final` vs. `conv_final` +2.0 / +1.1 / +0.8 / +0.4; `cnn3d_final` vs. `cnn3d` +1.4 / +0.6 / +2.1 / +0.8; `conv_final` vs. `hidden16` −0.5 / −0.2 / −0.2 / −0.4; vs. `neg2` −0.5 / −0.4 / −0.8 / −0.6.

Findings:
1. **The single ConvLSTM gains do not add up**: `conv_final` is not better than `hidden16` or `neg2` alone — consistent with the models being close to what the labels allow (each change ~+0.01–0.015, no stacking).
2. **`cnn3d_final` is the best configuration**: clearly above `ref` (highly significant at 10 km), +0.02 above the ConvLSTM candidate at 5 km (t ≈ 2); compared with plain `cnn3d` the hard negatives add little (+0.004–0.007) at 1.5× training time.
3. All good variants are within ~0.01 at 10 km; the baseline is beaten everywhere (+0.19 AP @ 5 km, +0.13 @ 10 km).

Proposed final evaluation: `cnn3d_final` (main model) and `conv_final` (ConvLSTM of the proposal), each retrained on 2004–2016 (3 seeds, last epoch — best and last epoch barely differ since the correction), evaluated once on 2017–2019 together with the baseline; breakdowns by isolation, size class and year (2017 extreme; 2018–2019 after the SEVIRI geo-referencing change). Needs `train.py --train-splits train,val --no-val` and a test-evaluation script.

### 6.16 Model review, inference-only experiments and error analysis (2026-10-24)

Full write-up: `docs/model_review.md` (review of all corrected-data ablations, literature scan 2023–2026, pre-/post-processing suggestions, proposed order of work). Validation only; numbers from scratch scripts on the saved `models/ablations_georef/` checkpoints (no retraining); not yet part of `src/`.
- **Post-processing** (`cnn3d_final`, 3 seeds): 3-seed ensemble of probability maps 0.640 / 0.714 AP @ 5 / 10 km (single models 0.634 / 0.703); + log-parabola sub-pixel decoding (separable DARK) **0.655 / 0.713, median error @ 5 km 1.89 km** (3×3 centroid 2.00 km). Flip TTA: fewer false alarms, no AP gain; mixed CNN + ConvLSTM ensemble: no gain.
- **Misses:** targets with another target ≤ 2 px in the same period (52 of 403) have recall 0.44 @ 10 km vs. 0.97 for the rest (~70 % of all misses). Such pairs (196 in 2004–2019) are separate records, median 4.2 km and 1.7 h apart → motivates a space–time (t, i, j) ignition output. Model vs. baseline recall @ 10 km for targets next to active fires (6 h history): 0.75 vs. 0.31.
- **False alarms** at the best-F1 threshold @ 10 km (76): 30 % near an ICNF fire < 10 ha of the same period, 28 % near a ≥ 10 ha fire started 1–72 h before; 2,163 seen 10–50 ha training fires are currently ignore regions.
- **Calibration:** peak probabilities rank well but are not calibrated (0.5–0.6 → 68 % hits, 0.6–0.8 → 92 %).
- **Localisation floor:** nearest detected pixel centre is a median 1.63 km from the ICNF point (pure pixel quantisation 1.51 km); the model reaches 1.89 km.
- **Data:** 105 whole-domain empty slots between slots with ≥ 10 detections on busy summer days (missing scans/files). Residual north–south label offset (~0.2 px) is larger in mountain districts (Guarda 0.28 px) than in lowlands (Beja 0.14) → hypothesis: uncorrected terrain parallax (h · tan VZA, VZA ≈ 47°); to test with a DEM.

### 6.16 Final training and test evaluation: implementation (2026-10-26, not yet run)

Code:
- `src/train.py`: `--train-splits` (comma list, default `train`) and `--no-val` (no validation; the checkpoint of the last epoch is saved). Guards: training on `test` is impossible; training on `val` requires `--no-val`. Justification for the last epoch: on the corrected data best-epoch and last-five means barely differ (e.g. `cnn3d_final` AP@5 0.635 vs. 0.621, AP@10 0.704 vs. 0.696).
- `src/evaluate.py`: `evaluate(..., periods=...)` scores a subset of a split's periods (used for AP per year).
- `src/evaluate_test.py`: evaluates all final checkpoints (`models/final/<experiment>_s<seed>.pt`) and the nucleus-birth baseline at 5 and 10 km — per seed and mean ± std over seeds; recall by isolation, size class and year; AP per year (2017 extreme year before the SEVIRI geo-referencing change, 2018–2019 after). **Safeguard: the test split is only used with `--confirm-test`**; without it the script evaluates the validation split (code check without test numbers). Outputs `outputs/final/<split>_results_{per_seed,summary}.csv`, `<split>_results.md`, `figures/final/<split>_pr.png`.
- Smoke test (3 batches, validation split, files deleted): final-mode training uses 2,382 target crops (train 1,979 + val 403; 7,146 crops/epoch with 2 negatives); the val guard fires; the evaluation script reproduces the corrected baseline on validation (0.448 / 0.577).

Commands (≈ 4–5 h locally for the 6 trainings):
1. `caffeinate -i .venv/bin/python -u src/run_ablations.py --only cnn3d_final conv_final --seeds 0 1 2 --folder final --extra="--train-splits train,val --no-val" 2>&1 | tee outputs/final_progress.txt`
2. once: `.venv/bin/python src/evaluate_test.py --confirm-test > outputs/final/test_log.txt`

### 6.17 Final test results 2017–2019 (2026-10-27)

Final models trained on 2004–2016 (train + val), 20 epochs, last epoch, 3 seeds each (`models/final/`, all converged; cnn3d ~2.15 min/epoch, ConvLSTM ~1.45 min/epoch with 2 hard negatives); evaluated **once** on the test split with `src/evaluate_test.py --confirm-test` (corrected build). Outputs: `outputs/final/test_results.md`, `test_results_{per_seed,summary}.csv`, `test_log.txt`; figure `figures/final/test_pr.png`. Mean ± std over seeds (baseline deterministic).

| Test 2017–2019 | AP @ 5 km | AP @ 10 km | Recall @ 10 km (all preds) | Best F1 @ 10 km (P / R) | False alarms/day @ best F1 (10 km) | Nearest pred ≤ 5 km | Median error of matches @ 5 km |
|---|---|---|---|---|---|---|---|
| baseline (nucleus birth) | 0.501 | 0.593 | 0.856 | 0.649 (0.639 / 0.661) | 0.150 | 0.813 | 1.86 km |
| **cnn3d_final** | **0.623 ± 0.012** | **0.711 ± 0.004** | **0.938 ± 0.004** | 0.693 (0.719 / 0.669) | 0.105 | **0.891** | 1.96 km |
| **conv_final** (ConvLSTM) | **0.620 ± 0.013** | **0.714 ± 0.013** | 0.933 ± 0.008 | 0.688 (0.745 / 0.640) | 0.088 | 0.888 | 2.00 km |

Recall (all predictions) by subgroup, 5 / 10 km:

| | non-isolated | isolated | 50–100 ha | 100–500 ha | 500–1k ha | ≥ 1k ha |
|---|---|---|---|---|---|---|
| baseline | 0.734 / 0.810 | 0.903 / 0.963 | 0.787 / 0.853 | 0.811 / 0.876 | 0.750 / 0.864 | 0.733 / 0.800 |
| cnn3d_final | 0.833 / 0.911 | 0.945 / 1.000 | 0.907 / 0.953 | 0.843 / 0.924 | 0.871 / 0.970 | 0.839 / 0.922 |
| conv_final | 0.826 / 0.904 | 0.953 / 1.000 | 0.902 / 0.949 | 0.843 / 0.919 | 0.894 / 0.962 | 0.817 / 0.917 |

AP by year, 5 / 10 km (2017 before, 2018–2019 after the SEVIRI geo-referencing change):

| | 2017 | 2018 | 2019 |
|---|---|---|---|
| baseline | 0.454 / 0.551 | 0.711 / 0.774 | 0.680 / 0.747 |
| cnn3d_final | 0.583 / 0.683 | 0.848 / 0.849 | 0.729 / 0.797 |
| conv_final | 0.591 / 0.693 | 0.859 / 0.859 | 0.680 / 0.771 |

Findings:
1. **Both models clearly beat the baseline on unseen years**: AP +0.12 at 5 km and +0.12 at 10 km; recall @ 10 km 0.94 vs. 0.86; fewer false alarms at the best-F1 operating point (~0.1 vs. 0.15 per day). The curves lie above the baseline at all recall levels.
2. **Results generalise from validation to test**: cnn3d_final 0.635 / 0.704 (val) vs. 0.623 / 0.711 (test) — no sign that the model choices overfitted the validation years.
3. **The 3D CNN and the ConvLSTM are equivalent on test** (0.623 vs. 0.620 at 5 km, 0.711 vs. 0.714 at 10 km; differences far below the seed spread). The +0.02 advantage of the 3D CNN at 5 km on validation does not replicate. Conclusion for the thesis: the recurrent memory brings no advantage, but no disadvantage either; both architectures reach the same level.
4. **Largest gains where the hand-made approach is weakest**: fires with neighbours (recall @ 5 km 0.83 vs. 0.73, +10 points) and large fires ≥ 1,000 ha (+10–12 points).
5. **Localisation per matched fire is not better than the first hotspot**: median error of matches ~2.0 km (models) vs. 1.9 km (baseline), but the models match more fires within 5 km (89 % vs. 81 %) and rank them better. The gain is in finding and ranking the right ignitions, not in sub-pixel precision for fires the baseline already finds — the ~2 km level looks like a floor set by pixel size and ICNF position accuracy.
6. **The geo-referencing correction generalises**: models trained almost entirely on pre-2018 (corrected) labels work very well after the SEVIRI change (2018: AP @ 5 km 0.85–0.86 vs. 0.71 baseline). 2017 (extreme year, many simultaneous fires) is the hardest year for all methods but shows the largest absolute gain (+0.13–0.14). 2019 shows the smallest gain (cnn3d +0.05 at 5 km, ConvLSTM ±0); per-year numbers rest on few targets (2018: ~35, 2019: ~90), so they are noisy.
- Caveats: one test evaluation, 3 seeds; test years 2017–2019 only (2017 dominates); ICNF ignition positions are themselves uncertain.
