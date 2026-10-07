# Master Thesis - Notes/Actions

This document serves as a basis for the write up. All notes (more specifically actions taken) will be noted here. Each step and decision will be documented to make sure the final work is complete and reproducable.

## 0. Repository layout and how to reproduce

```
data/raw/          ignic_ICNF.xlsx, FRP_SEVIRI_PT_2004_2019.mat   (as received, never modified)
data/processed/    icnf_fires.parquet, seviri_frp.parquet          (src/clean_data.py)
data/external/     ne_10m_admin_0_countries.zip (country borders, see 2.2)
outputs/profiling/ ydata-profiling reports in raw/ and processed/  (src/eda.py)
outputs/           eda_figures_log.txt (numbers printed by src/eda_figures.py)
figures/           plots                                           (src/timezone_check.py, src/plot_largest_fires.py)
figures/eda/       descriptive figures 01–09                       (src/eda_figures.py)
src/               all code
```

Environment: Python 3.11, `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (only packages imported by `src/`, pinned).

Run from the repository root, in this order:
1. `.venv/bin/python src/clean_data.py`: raw → processed parquet (1.3, 1.5)
2. `.venv/bin/python src/eda.py`: profiling reports (1.6)
3. `.venv/bin/python src/timezone_check.py`: time zone check (1.4)
4. `.venv/bin/python src/plot_largest_fires.py`: largest fires in lat–lon–time (2.1)
5. `.venv/bin/python src/eda_figures.py > outputs/eda_figures_log.txt`: descriptive figures (2.2)

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
