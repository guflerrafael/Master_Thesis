# Model review: which model, what limits it, what to try next (2026-10-24)

> Purpose: a stand-back review of all ablation results on the corrected data (notes §6.12–6.15), plus new inference-only experiments, a data/error analysis, and a literature scan of 2023–2026 work. It ends in a ranked list of next steps. **Validation 2015–2016 only — the test split stays locked.** New numbers in this file come from scratch scripts (`predict_val.py`, `analyze_val.py`; inference on the saved `models/ablations_georef/*` checkpoints, no retraining). The literature in §5 comes from web searches on 2026-10-24. Items tagged [abstract] were checked only against the abstract or search snippets, not the full text, and must be verified before they are cited.

---

## 1. Bottom line

1. **Use `cnn3d_final` as the main model** and keep `conv_final` as the ConvLSTM of the proposal. All architecture variants lie within ±0.03 AP. Validation plateaus after ~12 epochs. Recall @ 10 km reaches ~0.90 from epoch 2 onwards. Ensembling three seeds adds only +0.006 / +0.011 AP. **The model is no longer architecture-limited. It is limited by how the targets and outputs are defined, and by the data.**
2. **The dominant error is close ignitions.** 52 of the 403 validation targets (13 %) have another target within 2 px in the same 12 h period. Their recall is **0.44 @ 10 km, against 0.97 for all other targets**, which accounts for about 70 % of all misses. These pairs are real, separate records (median 4.2 km and **1.7 h** apart), and a single 2D map per period cannot represent two ignitions in neighbouring pixels. **The most promising model change is to predict ignitions in space *and time*** (§4.1).
3. **About 30 % of the false alarms at the best-F1 threshold are real fires below the label threshold** (an ICNF fire < 10 ha within 5 km in the same period). Another 28 % are fires ≥ 10 ha that started 1–72 h earlier (ongoing fires or rekindles). Training currently ignores the **2,163 seen 10–50 ha fires**. Using them with a size output would roughly double the labelled ignitions (§4.2).
4. **Localisation is close to the pixel floor.** The best possible "nearest detected pixel centre" is a median 1.6 km from the ICNF point. Pixel quantisation alone gives 1.5 km. The model reaches 2.0 km, or **1.9 km with better decoding**. Further gains need sub-pixel information: the FRP split between neighbouring pixels, finer sensors (VIIRS, MTG-FCI) or label corrections.
5. **Free post-processing gains** (no retraining) are a log-parabola sub-pixel decoder (+0.015 AP @ 5 km, −0.12 km median error) and a 3-seed ensemble (+0.006 / +0.011 AP, fewer false alarms). Flip test-time augmentation and mixing the CNN with the ConvLSTM do not help (§3).
6. **A new pre-processing hypothesis: terrain parallax.** After the geo-referencing fix there is still a ~0.2 px (~0.7–1 km) southward offset between ICNF points and first detections. It is larger in the mountain districts (Guarda 0.28 px, Vila Real/Bragança/Castelo Branco 0.25) than in the lowlands (Beja 0.14, Porto 0.15). This is what uncorrected terrain height produces in a geostationary view from the south. It can be tested and corrected with a DEM (§6.1).

## 2. Where the results stand

Validation 2015–2016, corrected build, 3 seeds each (notes §6.13–6.15). "best" = saved checkpoint; "last5" = mean of the last five evaluations.

| Model | AP @ 5 km best / last5 | AP @ 10 km best / last5 | min/epoch | Verdict |
|---|---|---|---|---|
| baseline (nucleus birth) | 0.448 | 0.577 | — | beaten everywhere |
| `ref` ConvLSTM 2×32 | 0.604 / 0.597 | 0.687 / 0.681 | 1.25 | over-sized |
| `hidden16` / `layers1` | 0.619 / 0.612 · 0.611 / 0.604 | 0.699 / 0.696 · 0.691 / 0.689 | 0.8 | smaller is as good |
| `readout_max` | 0.609 / 0.602 | 0.702 / 0.696 | 1.30 | readout bottleneck is real but small |
| `neg2` | 0.618 / 0.614 | 0.705 / 0.699 | 1.86 | only consistent sampling gain |
| `conv_final` | 0.613 / 0.609 | 0.696 / 0.692 | 1.23 | gains do not add up |
| `cnn3d` | 0.628 / 0.617 | 0.700 / 0.691 | 1.19 | no recurrence needed |
| **`cnn3d_final`** | **0.635 / 0.621** | **0.704 / 0.696** | 1.81 | best |
| `presence_only`, `hist0`, `births_plus`, `sigma05` | −0.01 … −0.02 | −0.01 … −0.02 | | FRP intensity, history and clustering add little |

What the pattern means:
- **Non-recurrent ≥ recurrent.** This agrees with the spatiotemporal-prediction benchmarks (OpenSTL), where recurrent-free models such as SimVP/TAU are competitive with ConvLSTM, and with BA-Net (Pinto et al. 2020), which also mixes temporal convolutions with recurrence. The useful signal sits in a short space–time window (where a pixel lights up first, and what was around it before), and a 3×3×3 convolution stack captures that directly.
- **Saturation.** In every run, train loss keeps falling while validation AP is flat after epoch 12–16. Capacity, depth and sampling each move AP by ≤ 0.02, and their gains do not stack (`conv_final`). This is typical of a label-limited problem: ~2,000 targets with ~1–2 km label noise on a 3×4.5 km grid.
- **The model's advantage over the baseline is separating ignitions next to active fires** (§3.3: "active in history" 0.31 → 0.75 recall @ 10 km). For fresh, isolated fires the baseline is already almost as good (0.93 vs. 0.95).

## 3. New inference-only experiments (validation)

### 3.1 Ensembles, test-time augmentation, decoding

`cnn3d_final` and `conv_final`, all three seeds, full-domain validation maps (844 periods). "single" = mean of the three single-seed scores.

| Variant | AP @ 5 | AP @ 10 | Recall @ 10 | Median error @ 5 km | False alarms/day @ best F1 (10 km) |
|---|---|---|---|---|---|
| `cnn3d_final` single | 0.634 | 0.703 | 0.897 | 2.06 km | 0.123 |
| + 3-seed ensemble (mean of probability maps) | 0.640 | **0.714** | 0.898 | 2.00 km | 0.104 |
| + log-parabola decoding | **0.655** | 0.713 | 0.898 | **1.89 km** | 0.105 |
| + flip TTA (4 flips) instead of decoding | 0.639 | 0.711 | 0.896 | 2.07 km | **0.088** |
| `conv_final` single | 0.613 | 0.696 | 0.888 | 2.03 km | 0.089 |
| `conv_final` 3-seed ensemble + TTA | 0.614 | 0.704 | 0.886 | 2.04 km | 0.078 |
| all 6 models (both architectures) | 0.635 | 0.713 | 0.893 | 2.01 km | 0.103 |

Decoders on the `cnn3d_final` ensemble: no sub-pixel 0.603 / 0.714 (2.22 km); current 3×3 centroid 0.640 / 0.714 (2.00 km); **log-parabola per axis 0.655 / 0.713 (1.89 km)**. The log-parabola decoder fits a parabola through the log-probabilities of the peak and its two neighbours, separately in i and j. It is exact for a Gaussian peak, which is the shape the targets are trained on, and is the separable form of DARK (Zhang et al. 2020).

Conclusions:
- **Adopt the log-parabola decoder and the 3-seed ensemble** for the final configuration. Fix both now, before the test run, so they are pre-registered. Together they add +0.021 AP @ 5 km and +0.011 AP @ 10 km over a single model.
- Flip TTA blurs the peaks: it gives fewer false alarms and no gain in AP. A mixed CNN/ConvLSTM ensemble is no better than the CNN alone. Report both as negative results.
- Caveat: the checkpoints were chosen on validation, so all absolute numbers are slightly optimistic. The comparisons are fair because every variant shares the same checkpoints.

### 3.2 Calibration (`cnn3d_final` ensemble, peaks vs. hit rate @ 10 km)

| Peak probability | 0.05–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.8 |
|---|---|---|---|---|---|---|---|
| Peaks | 462 | 594 | 701 | 361 | 167 | 133 | 163 |
| Share that are hits | 0.00 | 0.01 | 0.04 | 0.11 | 0.28 | 0.68 | 0.92 |

The scores rank well but are not probabilities. They are over-confident below 0.4 and under-confident above 0.5, with a very steep transition. If probabilities are reported (operational outlook), fit isotonic regression or temperature scaling on validation peaks (Guo et al. 2017). This does not change AP.

### 3.3 Where the misses are (`cnn3d_final` ensemble; baseline in brackets)

| Group | Targets | Recall @ 5 km | Recall @ 10 km |
|---|---|---|---|
| **another target ≤ 2 px in the same period** | 52 | **0.40** (0.21) | **0.44** (0.37) |
| no other target ≤ 2 px | 351 | 0.91 (0.79) | 0.97 (0.87) |
| context: fresh (nothing within 2 px in the 78 h before) | 242 | 0.91 (0.86) | 0.95 (0.93) |
| context: pause > 6 h | 89 | 0.85 (0.71) | 0.89 (0.87) |
| context: active in the 6 h history | 72 | 0.61 (0.22) | 0.75 (0.31) |
| isolated / non-isolated | 99 / 304 | 0.97 / 0.80 | 0.99 / 0.87 |
| 50–100 ha / ≥ 1,000 ha | 144 / 30 | 0.82 / 0.93 | 0.86 / 0.97 |
| start 06–18 / 18–06 UTC | 301 / 102 | 0.84 / 0.84 | 0.90 / 0.90 |

The close pairs over the whole period 2004–2019: 196 pairs of targets within 2 px in the same period (139 train, 36 val, 21 test). Distance median 4.2 km (10 % < 1.3 km). Start-time difference median **1.7 h**, and 38 % are under 1 h apart. Only 7 pairs are within 1 km and 1 h (possible duplicates), and 6 share the exact coordinates. Most pairs are therefore separate ignitions, a few kilometres and one to several hours apart, typically in the same municipality.

### 3.4 What the false alarms are (`cnn3d_final` ensemble, best-F1 threshold @ 10 km = 0.47: 264 TP, 76 FP)

| Nearest ICNF evidence | FP | Share |
|---|---|---|
| ICNF fire < 10 ha within 5 km, same period (not a label by definition) | 23 | 30 % |
| ICNF fire ≥ 10 ha within 5 km that started 1–72 h before (ongoing fire / rekindle) | 21 | 28 % |
| no ICNF fire nearby | 14 | 18 % |
| some ICNF fire within 10 km, same period | 11 | 14 % |
| coast/border cell, persistent pixel, other | 7 | 9 % |

At least ~58 % of the "false" alarms sit where something was actually burning. The precision problem is mostly a **label-definition** problem (size threshold, rekindles), not one of hallucinated fires.

### 3.5 Localisation floor (all 2,821 targets, sensor frame, detections within 2 px in [start − 1 h, start + 6 h])

| Estimate | Median distance to ICNF point |
|---|---|
| uniform point in a 3.2 × 4.5 km pixel → pixel centre (pure quantisation) | 1.51 km |
| **oracle: nearest detected pixel centre** | **1.63 km** (val 1.59) |
| FRP-weighted centroid of the first hour | 2.22 km |
| FRP-weighted centroid of the first detection slot | 2.42 km |
| `cnn3d_final` ensemble, log-parabola decoding (TP @ 5 km, val) | 1.89 km |

41 % of target fires show only one pixel in their first detection slot. The model already beats every hand-made centroid and is within ~0.3 km of an oracle that knows which detected pixel is closest. Pushing below ~1.5 km requires information below the pixel scale (§6).

## 4. Model recommendations (ranked by expected value / cost)

### 4.1 Predict ignitions in space and time (highest priority)
- **Problem it solves:** the dominant miss class (§3.3), where two ignitions fall within 2 px in one period but are typically hours apart.
- **Design:** `TemporalCNN3D` already keeps a time axis (stride 2 twice → 6 steps from 24 hourly ones) before it pools. Replace the max/mean pooling over time with a per-step head that outputs a heatmap over (t, i, j) at hourly or 2-hourly resolution inside the 12 h period. The target is a Gaussian in space (σ = 1 px) × time (σ ≈ 1 h) at each ignition's ICNF start time. Use the same focal loss, and 3D local maxima (3×3×3) as peaks. For scoring, collapse to points (unchanged evaluation), and additionally report the **time error** of matched ignitions. Benali et al. (2016) report ~12 h / 2 km agreement with the fire database for Portugal, so time accuracy is a comparison the closest predecessor already makes.
- **Cheaper variant:** keep the 2D heatmap and add an ignition-time regression head (L1 at target pixels) plus a 2-channel offset head (CenterNet). This gives the time estimate but does not separate merged peaks.
- **Literature:** this is the temporal analogue of CenterNet keypoints. It is close in spirit to per-pixel burn dating (Pinto et al. 2020, BA-Net) and fire-arrival-time estimation (Farguell et al. 2021; Shaddy et al. 2025), where the ignition is the earliest point of a space–time field.
- **Risk:** ICNF start times are uncertain by ~0.5–1 h (notes §1.4, §5.1), so use σ_t ≥ 1 h and do not require exact timing. Expected gain: up to roughly +0.04–0.06 recall @ 10 km if half of the close-pair misses are recovered. This is a rough estimate, not a measurement.

### 4.2 Use more of the labels: size-aware targets
- Training sees 1,979 targets ≥ 50 ha. The **2,163 seen 10–50 ha training fires are ignore regions**, but they are exactly what produces 30 % of the false alarms at inference.
- Options, in increasing ambition: (a) a second heatmap channel for ≥ 10 ha seen ignitions (multi-task; the ≥ 50 ha channel stays the evaluated one); (b) ordinal channels ≥ 10 / ≥ 50 / ≥ 100 ha; (c) one heatmap for all seen ≥ 10 ha ignitions plus a log-area regression at each peak, with the ranking score = p × P(area ≥ 50 ha).
- This also serves thesis objective (i)/(v): "relating patterns to wildfire attributes" (size) becomes a model output.
- Report the evaluation with the same ≥ 50 ha targets, so results remain comparable with §2.

### 4.3 Model 2 of the proposal: CNN + temporal attention, kept small
- Expect parity rather than a jump (§2: architecture is not the bottleneck). Its value lies in the proposal, interpretability (attention over hours shows *which* observations located the ignition) and longer windows for the post-fire reconstruction framing (notes §6.4).
- Concrete template: **U-TAE** (Garnot & Landrieu 2021), with a shared 2D encoder per step, lightweight multi-head temporal attention at reduced resolution, and attention masks that collapse time. U-TAE is also the time-series baseline in the wildfire-spread benchmarks (WildfireSpreadTS; WSTS+, WACV 2026). TSViT (Tarasiou et al. 2023) is the pure-transformer alternative, but it is compute-heavy and data-hungry for ~2,000 targets.
- Keep it below ~200k parameters, use the same split, seeds and decoding, and compare with `cnn3d_final`. WildfireSpreadBench (2026) [abstract] finds that wildfire model rankings change with the metric (AP vs. F1/IoU), so report AP @ 5/10 km *and* best F1 / false alarms per day.

### 4.4 Self-supervised pre-training on the unlabelled detections
- 568k detections, mostly unlabelled, against ~2,000 targets. Cheap pretext tasks on the existing tensors: masked space–time reconstruction of the detection/FRP channels (erase blocks, predict them), or **predict each pixel's first-detection time in the window** (dense, label-free, closely related to the ignition task). Then fine-tune `cnn3d_final`.
- Literature: HighFM (Girtsou et al. 2026) [abstract] is a SatMAE-style masked-autoencoder foundation model pre-trained on >2 TB of SEVIRI imagery, fine-tuned for cloud masking and active-fire detection. Rapid hotspot detection with Presto-based SSL on SEVIRI pixel time series (2024) [abstract]. Both work on SEVIRI radiances rather than the FRP product, so they are mainly evidence that pre-training on SEVIRI time series is useful. Using radiances would be a much larger change.
- Priority below 4.1–4.2, because the label-limited plateau may come from label noise rather than representation.

### 4.5 Not recommended now
- Larger or deeper models, 15-min steps (no gain, notes §6.9), the births channel (§6.13).
- Stateful or long-term memory (notes §6.6) **before** a cloud/observability mask exists. The "pause > 6 h" class is now at 0.89 recall @ 10 km, so the headroom is smaller than in the uncorrected analysis (0.86 for v2).
- A mixed CNN/ConvLSTM ensemble (§3.1).

## 5. Literature: similar problems and how they were solved

| Work | Problem / data | Approach | Relevance |
|---|---|---|---|
| Benali et al. 2016, *Remote Sens.* 8:326 | Ignition points and dates of Portuguese fires from polar-orbiting active fires | Rule-based; agreement with fire databases ~2 km / ~12 h, better for large fires, ~76 % of burned area covered [abstract] | Closest predecessor and external benchmark. The thesis model reaches ~1.9 km median with 15-min data (different fire set; compare carefully). |
| Mandel et al. 2018 (arXiv 1808.03318); Farguell et al. 2021, *Remote Sens.* 13; Shaddy et al. 2025 (arXiv 2506.10404) | Fire arrival time and ignition from MODIS/VIIRS (+ GOES ignition times) | Data likelihood over WRF-SFIRE runs (ignition ~3 km from official point in one case); SVM arrival-time surface; conditional GAN trained on simulations (Dice 0.81 on 5 fires) [abstract] | Inverse problem framed as a space–time field. Supports §4.1 and the "arrival time" target idea. Simulation-based training is a route around label scarcity. |
| Paugam, Filippi, **Benali** et al. 2026 (arXiv 2606.06016) | MTG-FCI hotspots (1 × 1.5 km over S. Europe, 10 min), Portugal and Mediterranean 2025 | Fire Event Tracker: rule-based spatio-temporal clustering, persistent event IDs, FRP and rate of spread every 10 min [abstract] | The successor-sensor version of the nucleus baseline, from the Portuguese community. Potential contact and data source. |
| LSA SAF MTG FRP product (demonstration, Oct 2025) | FCI FRP at 1 km, every 10 min, back-processed to 2025-01-01 | Operational product | Finer pixels address the 1.5 km quantisation floor (§3.5). Option for the "operational outlook" or an extension. |
| Zhao & Ban 2022, *Remote Sens.* 14:4347 | Early detection from GOES-R ABI 15-min time series | Per-pixel GRU; earlier than VIIRS, more precise early locations than the GOES product at mid/low latitudes | Geostationary DL precedent; early-detection framing. |
| HighFM (Girtsou et al. 2026); SSL hotspot detection (arXiv 2405.20093) | SEVIRI imagery time series | Masked-autoencoder foundation model; Presto-style SSL [abstract] | Pre-training (§4.4); SEVIRI-native. |
| TS-SatFire (Sci. Data 2025); WildfireSpreadTS (NeurIPS 2023); WSTS+ (WACV 2026); WildfireSpreadBench (2026) | VIIRS time-series fire detection / spread benchmarks | U-Net, ConvLSTM, U-TAE, 3D CNNs; time-series input helps; rankings depend on the metric [abstract] | Model-2 template (U-TAE), evaluation practice (AP + threshold metrics, year folds). |
| Yang et al. 2026 (arXiv 2603.09042) | Fire spread under clouds/smoke (partial observability) | Two stages: masked inpainting of fire maps (U-Net / CVAE / ViT / diffusion), then forecasting; trained with pixel/block masking 10–80 % [abstract] | Supports erase augmentation and an observability channel (§6.2). |
| OpenSTL benchmark (NeurIPS 2023 D&B) | Spatiotemporal predictive learning | Recurrent-free SimVP/TAU competitive with ConvLSTM-family models | Context for "`cnn3d` ≥ ConvLSTM". |
| DARK (CVPR 2020); SimCC (ECCV 2022); Chatzitofis et al. 2021 (arXiv 2110.10289) | Sub-pixel keypoint decoding | Taylor/log-Gaussian decoding; coordinate classification; centre of mass can be as good as DARK | §3.1 decoding results; SimCC-style binned offsets as an alternative offset head. |
| Bernhard & Schubert 2021 (*Remote Sens.*); Noised-autoencoder point refinement (arXiv 2312.07190) | Training with noisy point labels | Co-correction / learned refinement of point annotations [abstract] | ICNF points are noisy (shared coordinates, ~1–2 km). A label-refinement step could be tested once a DEM/parallax correction is in. |

Literature gaps: still no study found that *learns* ignition locations from geostationary FRP sequences (confirms literature.md §4.2). Benali et al. (2016) and Paugam et al. (2026) are the two works to read first.

## 6. Pre-processing suggestions

### 6.1 Terrain parallax correction (new, testable)
- SEVIRI geolocation assumes the ellipsoid surface. A fire at height h is seen displaced away from the sub-satellite point (0°, 0°) by ≈ h · tan(VZA). Over Portugal VZA ≈ 45–50°, so the displacement is ≈ 1.0–1.2 × h, almost due north with a small westward component. At 600–800 m this is 0.6–0.9 km ≈ 0.15–0.2 native px in the north–south direction.
- Evidence: the residual label − first-detection offset after the geo-referencing fix is −0.16 px north in both periods (notes §6.11). By district (first-slot FRP-weighted centroid, all targets with ≥ 40 fires): highest in Guarda 0.28 px, Vila Real / Bragança / Castelo Branco 0.25, Faro (Monchique) 0.27; lowest in Viana do Castelo 0.01, Beja 0.14, Porto/Leiria 0.15. The east–west offset is ≈ 0 everywhere. The pattern fits terrain parallax, but districts are a weak proxy and fire behaviour (uphill spread) could add to it.
- Test: sample a DEM (Copernicus GLO-30 or SRTM) at each ICNF ignition, regress the north–south offset on h · tan(VZA). If the slope ≈ 1, correct like the geo-referencing fix (move labels into the sensor frame by the parallax vector of their DEM height). This also explains why `noflip` kept a small advantage (+0.03 AP @ 5 km, notes §6.13).

### 6.2 Observability: missing slots, clouds, smoke
- On busy summer days (≥ 200 detections), 105 slots have zero detections in the whole domain while the slots before and after each have ≥ 10. These are almost certainly missing scans or missing product files, not fires going out. Hourly max aggregation hides them, which may be one reason why 15-min steps did not help.
- Add an **observability channel**: (a) cheap: per slot, "product present" (any detection anywhere in the full extract ±1 slot); (b) proper: the SEVIRI cloud mask (EUMETSAT Data Store collection `EO:EUM:DAT:MSG:CLM`, 15 min) or the CM SAF CLAAS record (2004–2015+). The archive start of CLM must be checked.
- **Erase augmentation** (Pinto et al. 2020; Yang et al. 2026): randomly delete detections in space–time blocks and whole slots during training. Cheap to add, and it targets the Pedrógão-type omissions.

### 6.3 Recover 15-min information inside hourly steps
15-min steps cost 4× and gave nothing (notes §6.9), but the first 15-min observation still carries timing. Keep hourly steps and add per-hour channels: the **number of 15-min slots with a detection** (0–4) and the **index of the first detected slot** in the hour. This costs nothing at run time and keeps the sub-hour onset.

### 6.4 Labels
- 10 % of the fires ≥ 100 ha sit on coordinates shared with other fires (notes §2.3). Flag targets at points used ≥ 2 times and test down-weighting them (an ablation, not a rule).
- Report recall against **all** validation fires ≥ 50 ha, not only seen ones: 497 fires, of which 94 (19 %) are unseen and currently ignored. Detection rate by size: 50–100 ha 76 %, ≥ 1,000 ha 91 %. The current numbers are conditional on SEVIRI seeing the fire; say so in the thesis.
- Rekindles: the 28 % of false alarms at places burning 1–72 h earlier need the ICNF rekindle flag, if it exists (literature.md §4.3), before any memory model is judged.

### 6.5 More and finer data
- **SEVIRI FRP-PIXEL 2020–2024** from the LSA SAF archive (same product as the extract; ICNF labels exist; 2022 was a large fire year). This is the cheapest way to add ~15–25 % more targets and a later test year.
- **VIIRS 375 m (FIRMS, 2012+)** as a finer reference: it measures the true sub-pixel limit and could serve as an auxiliary target.
- **MTG-FCI FRP (1 km, 10 min, 2025+)**: an outlook or an out-of-distribution test on a better sensor.
- Covariates (notes §3, objective iv): wind (ERA5-Land u/v), DEM/slope, FWI. The expected effect on localisation is modest (the input is already the fire itself), but they may help the "active in history" class (direction of spread) and they are required for objective (iv). Add them as static/hourly channels to `cnn3d_final` and run them as ablations.

### 6.6 Evaluation protocol
- Leave-years-out cross-validation (e.g. 4 folds over 2004–2016) for the final comparison of `cnn3d_final` vs. ConvLSTM vs. Model 2. A single validation pair of years with 403 targets gives ±0.01–0.02 AP noise.
- Pre-register the final pipeline (architecture, epochs, decoder, ensemble) before the single test run; retrain on 2004–2016 (notes §6.15).
- Report by year in the test set: 2017 (extreme), and 2018–2019 (after the EUMETSAT geo-referencing change).

## 7. Post-processing suggestions (summary)
1. Log-parabola sub-pixel decoding — adopt (§3.1).
2. 3-seed ensemble of probability maps — adopt for the final run (3× inference, still < 2 min per seed on the M1 for the validation split).
3. Flip TTA, mixed-architecture ensemble — no gain; report as negative results.
4. Calibration (isotonic regression on validation peaks) — only if probabilities are shown to users.
5. Peak merging is not solvable by NMS settings (notes §6.9); it needs §4.1.

## 8. Proposed order of work

| # | Step | Cost | Expected effect |
|---|---|---|---|
| 1 | Make the log-parabola decoder and seed ensemble options of `src/train.py` / `peak_ablation.py`; re-score existing checkpoints | hours | +0.02 AP @ 5 km, free |
| 2 | DEM-based parallax test (§6.1); if confirmed, correct the labels and re-run `ref` + `cnn3d_final` | 1 day | possibly −0.2–0.5 km error; consistency |
| 3 | Space–time ignition head on `TemporalCNN3D` (§4.1) | 2–3 days | targets 70 % of misses; adds ignition time |
| 4 | 10–50 ha seen fires as an auxiliary/size channel (§4.2) | 1–2 days | fewer "false" alarms, 2× labels |
| 5 | Observability channel + erase augmentation; hourly count/first-slot channels (§6.2–6.3) | 1 day | robustness; cheap |
| 6 | Model 2 (U-TAE-style) on the best target definition (§4.3) | 3–4 days | parity + XAI + proposal |
| 7 | Covariates (wind, DEM, FWI) as ablations | 3–5 days (data) | objective iv |
| 8 | Leave-years-out CV, retrain 2004–2016, single test evaluation | 1–2 days compute | final numbers |

## Sources (web, 2026-10-24)
- Benali et al. 2016 — https://www.mdpi.com/2072-4292/8/4/326
- Mandel et al., data likelihood of active fires / ignition estimation — https://arxiv.org/pdf/1808.03318
- Farguell et al. 2021, ML fire arrival time — https://par.nsf.gov/biblio/10280475-machine-learning-estimation-fire-arrival-time-from-level-active-fires-satellite-data
- Shaddy et al. 2025, generative wildfire progression reconstruction — https://arxiv.org/abs/2506.10404
- Assimilation of satellite active fires data (review) — https://arxiv.org/pdf/2204.00686
- Paugam et al. 2026, MTG-FCI fire event tracking — https://arxiv.org/abs/2606.06016
- LSA SAF, MTG FRP demonstration product — https://lsa-saf.eumetsat.int/en/news/news/2025/10/01/release-of-mtg-fire-radiative-power-product-as-demonstration/
- Zhao & Ban 2022, GOES-R GRU early detection — https://www.mdpi.com/2072-4292/14/17/4347/html
- HighFM 2026 — https://arxiv.org/abs/2604.04306
- SSL rapid hotspot detection on SEVIRI — https://arxiv.org/pdf/2405.20093
- TS-SatFire — https://www.nature.com/articles/s41597-025-06271-3
- WildfireSpreadBench — https://arxiv.org/pdf/2609.22191 ; WSTS+ — https://wacv.thecvf.com/virtual/2026/poster/685
- Yang et al. 2026, partial observability — https://arxiv.org/abs/2603.09042
- OpenSTL — https://arxiv.org/pdf/2306.11249
- U-TAE — https://arxiv.org/pdf/2107.07933 ; TSViT — https://ar5iv.labs.arxiv.org/html/2301.04944
- Coordinate decoding comparison — https://arxiv.org/pdf/2110.10289
- Noisy point labels — https://www.dbs.ifi.lmu.de/cms/aktuelles/paper_remote_sensing_2021/index.html ; https://arxiv.org/html/2312.07190v1
- SEVIRI cloud mask on the EUMETSAT Data Store — https://data.eumetsat.int/data/map/EO:EUM:DAT:MSG:CLM ; CLAAS-2 — https://www.knmi.nl/research/publications/the-msg-seviri-based-cloud-property-data-record-claas-2
