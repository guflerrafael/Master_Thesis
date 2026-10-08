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
src/               all code
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

## 1. Data

### 1.1 ICNF

- Portugal's official fire database, supplied by the supervisor as `ignic_ICNF.xlsx`
- One row per fire: start (`Data`) and end (`DataFim`) date/time, municipality code (`Dicofre`), district (`Distrito`), municipality (`Concelho`), burned area in ha (`AA`), and from 2001 on the estimated ignition point (`lat`, `lon`)
- 813,384 fires, 1980–2023
- A ".csv" copy was made at first for easier handling but was found **not** to be lossless (checked 2026-10-07), and has since been deleted; the ".xlsx" is the only source:
  - 31,799 `Data` and 39,286 `DataFim` values were shifted by −1 s (e.g. 14:30:00 → 14:29:59), all in 1989–1997 (floating-point rounding of Excel date serials during the export).
  - The 1900-01-01 sentinel in `DataFim` became 1899-12-31 (Excel 1900 leap-year bug).
  - All other columns were identical. The affected rows have no coordinates and lie outside the SEVIRI period, so the first exploration was not affected.
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
