# Supervisor meeting: preparation (2026-10-28)

> This document is for discussion, not for presenting. Part A is the story in a few minutes. Part B goes through the decisions one by one, each with *what*, *why* and *how to defend it*. Part C covers the harder concepts (loss, metrics, architectures) in plain terms first and then with details. Part D is honest caveats, questions for the supervisor and likely questions back. Sources: `notes.md` (§ numbers below), `model_review.md`, `data_pipeline.md`, `outputs/final/test_results.md`.

---

# Part A: the story in 3 minutes

**One sentence.** I trained a small neural network that looks at 24 h of SEVIRI hotspots and outputs a map of where new fires (≥ 50 ha) started in the middle 12 h. On the unseen years 2017–2019 it finds clearly more ignitions, ranks them better and gives fewer false alarms than a hand-made rule. The typical location error is ~2 km, on pixels of 3 × 4.5 km.

**The arc, in six steps:**
1. **Feel the data** (your suggestion). Large fires are visible in SEVIRI within ~1 h, and the first hotspot sits 1–2 pixels from the ICNF point. Along the way I found and fixed problems: ICNF times are **local time, not UTC**. Simultaneous neighbouring fires are **common, not rare**: about 1 in 4 large fires has another ≥ 100 ha fire within 10 km on the same day.
2. **Clustering** of hotspots into events/nuclei. It works for isolated fires, fails for fires that start next to an active one, and makes a good **baseline**.
3. **Change of approach**: no hand-made hotspot ↔ fire assignment. The model gets the raw hotspots and predicts an **ignition heatmap** with 0, 1 or many ignitions per map, which is the "ignition probability map" of the proposal.
4. **First ConvLSTM** beats the baseline. Then a systematic **ablation study** with 3 seeds per setting. One lesson from it: single runs lie.
5. **Biggest discovery: a half-pixel geo-referencing offset** in SEVIRI data before Dec 2017 (documented by EUMETSAT/satpy). I found it because a "no flip augmentation" run looked suspiciously good. After correcting it, all models improved strongly (AP @ 5 km 0.35 → 0.60) and training became stable.
6. **Final models** (3D CNN and ConvLSTM), retrained on 2004–2016 and evaluated **once** on the locked test set 2017–2019: **AP @ 5 km 0.62 vs. 0.50 for the baseline, AP @ 10 km 0.71 vs. 0.59**.

**What I want from the meeting:** (1) agreement on the heatmap framing and the evaluation; (2) feedback on what comes next: covariates, Transformer (model 2), space–time output, explainability; (3) data: SEVIRI 2020+, cloud mask, rekindle flag in ICNF.

---

# Part B: decisions, one by one

Format: **Decision** → *why* → *alternative / how to defend*.

## B1. Data and cleaning

| Decision | Why | If asked |
|---|---|---|
| Raw files never modified; everything scripted (`src/clean_data.py`) and logged in `notes.md` | Reproducibility | Every number in the thesis can be regenerated |
| ICNF: drop exact duplicates (6,452), set invalid end times/coordinates to missing and flag them, but **keep the rows** | Start time and area are still usable; downstream code filters on the flags | 23 "Atlantic default point" rows (2014–15), 1994 end dates lost (1970 epoch) |
| SEVIRI: drop 896 duplicates, **no confidence filter** | Confidence is strongly coupled to FRP. A filter removes exactly the weak, early detections that carry the ignition location | Low-confidence hotspots are more often unmatched (artefacts?), but the model gets confidence as an input and learns how much to trust it |
| Keep Spain as **input context**, exclude it from the loss | No labels there, but fires crossing the border are real | Loss mask = mainland Portugal + 1 px border |

## B2. Time zone (§1.4)
- **Finding:** ICNF times are **Portuguese local time** (with daylight saving). Test: the lag "first SEVIRI detection − ICNF start" jumps by ~0.9 h at the DST switch, also when comparing only the same months, so it is not a seasonal effect.
- Before the fix, fires appeared ~40 min *before* they started. After the fix, SEVIRI sees fires ~10–20 min *after* the reported start, which is physically sensible.
- Everything internally in UTC. 42 start times in DST-switch hours cannot be converted and are dropped.

## B3. What SEVIRI can see → minimum fire size (§2.2, §4.1)
- Detection probability by size (vs. a ±7-day chance control): 10–50 ha 43 %, 50–100 ha 68 %, ≥ 100 ha ≥ 80 %.
- **Targets = fires ≥ 50 ha that SEVIRI "sees"** (a detection within 5 km in [start − 1 h, start + 6 h]). That gives 2,821 targets in 2004–2019 after the geo-fix.
- *Why 50 and not 100 ha:* ~1,000 more labels. The ablation `target100` showed no gain from using only the larger fires.
- 10–50 ha fires and unseen ≥ 50 ha fires → **ignore regions** (no loss). The model should not be punished either for finding them or for missing them.
- *Caveat to say openly:* the results are **conditional on SEVIRI seeing the fire**. About 19 % of ≥ 50 ha validation fires are unseen, so they are not in the denominator.

## B4. Neighbouring fires are the central difficulty (§2.3, §3)
- 83 % of the ≥ 100 ha fires start on a day with another ≥ 100 ha fire (outbreaks: 2017-10-15 had 51 such fires).
- One in three large fires shares more than half of its hotspots with another fire.
- **Implication:** "one fire = one sample" does not work. That motivates the clustering (B5), and later the heatmap formulation (B6).

## B5. Clustering into events and nuclei (§3.1–3.2) → baseline
- **Events** = single-linkage clusters in space and time (≤ 6 km ≈ the 8 neighbouring pixels, ≤ 6 h gap). D and the gap were chosen from a sensitivity grid (merging vs. fragmentation trade-off).
- **Nuclei** = separate "births" inside a merged event (a new group of detections that appears and later joins others).
- Built from **SEVIRI only**. ICNF is used only to evaluate it, which avoids label leakage.
- Result: nuclei halve the error for fires that join an already burning event (11 → 6 km), but ~3/4 of those still share a nucleus with another fire. 32 % of nuclei are reappearances (night/cloud/smoke), not ignitions.
- **Use: the baseline.** Every nucleus birth is a predicted ignition, scored by its fire radiative energy. This is a fair, non-learned competitor that uses the same information.

## B6. Main framing: ignition heatmaps from raw hotspots (§4.0) ← key decision to confirm
- If the clustering decides which hotspots the model sees, its errors become the model's errors, and it already does the hard part (assignment).
- **Instead:** input = all hotspots in a space–time window. Target = a Gaussian bump at **every** ICNF ignition in the window. The model learns the assignment itself.
- This is the "ignition probability map" of the proposal, and it handles 0, 1 or 31 ignitions per sample naturally.
- The analogy is keypoint detection in computer vision: CenterNet detects objects as heatmap peaks.
- Evidence it was right: the model's largest gain over the baseline is exactly on fires **with neighbours** and fires **next to active fronts** (recall @ 10 km 0.75 vs. 0.31 on validation).
- Ablations: adding the clustering as an extra input channel (`births_plus`) gives **nothing** on corrected data, and clustering alone as input (`births_only`) is far worse. The model extracts what the clustering provides.

## B7. Grid: the native SEVIRI pixel grid (§4.2)
- SEVIRI pixel centres (irregular in lat/lon) map back **exactly** to the satellite's own grid with the geostationary projection (sub-satellite longitude 0°). The result: 8,402 pixels → 8,402 cells, no collisions, domain 177 × 106.
- *Why:* no resampling or interpolation. Each cell is one real measurement. ICNF points get **continuous (sub-pixel)** coordinates on this grid.
- Pixel over Portugal ≈ 3.2 km E–W × 4.5 km N–S.

## B8. Time windows / sample design (§4.2)
- **12 h target periods**, 06–18 and 18–06 UTC. 06 UTC is the diurnal minimum of fire activity, so few fires straddle a boundary. Each fire belongs to exactly one period.
- **Input = 6 h history + 12 h period + 6 h tail = 24 h.** The history shows what was already burning ("not new"). The tail guarantees ≥ 6 h of observation after every ignition.
- Hourly steps (max over four 15-min slots). 15-min steps were tested with 3 seeds: **no gain at 4× cost** (B12).
- **Framing to state clearly:** this is a near-real-time / short-window setup. Post-fire reconstruction with a long tail is a different framing (§6.4).
- Input channels: log FRP, confidence, detected flag, period flag, time, mainland mask, persistent-pixel flag.

## B9. Train/validation/test split (§4.2, §6.4)
- **Chronological:** train 2004–2014, validation 2015–2016, test 2017–2019 (locked until the end).
- *Why chronological, not random:* (a) no leakage between simultaneous fires of one outbreak; (b) it mirrors real use (train on the past, apply to the future); (c) validation is used for choices and the test is touched once.
- Final models were retrained on 2004–2016 and evaluated **once** on 2017–2019 (with a `--confirm-test` safeguard in the code).
- *Weak point:* the test is dominated by 2017 (extreme year), and one validation pair of years gives ±0.01–0.02 AP noise. Leave-years-out CV would be more robust (proposed next step).

## B10. The geo-referencing offset (§6.11–6.13) ← the most interesting story
- **Symptom:** in the ablation without flip augmentation, AP @ 5 km almost doubled. Flips were implemented correctly, so the data had to contain a **directional** pattern.
- **Diagnosis:** the vector from the first SEVIRI detection to the ICNF ignition was +1.5 km E, −3.0 km N on average until 2017, then ~0 from 2018 on. That is a step change of **half a pixel**.
- **Confirmation:** satpy/EUMETSAT documentation says SEVIRI L1.5 data before December 2017 is shifted 1.5 km N and W at the sub-satellite point, i.e. half a pixel. My fire-based estimate (+0.46 px E, −0.52 px N) matches it independently.
- **Fix:** correct the **labels** into the sensor frame, not the detections, so that detections stay on exact integer pixels. Predictions are shifted back before scoring against ICNF.
- **Effect:** the baseline's error drops from 4.3 to 2.2 km, and the model's AP @ 5 km from ~0.35 to 0.60. Seed spread shrinks from ±0.05 to ±0.016. Earlier conclusions about σ and flips **reversed**.
- **Why it matters for the thesis:** most of the "the first hotspot is ~4 km off" error was a georeferencing artefact, not fire physics or label noise. It is also a methodological lesson: ablations can reveal data problems.
- **Open follow-up:** a remaining ~0.2 px southward offset that is larger in mountain districts → hypothesis: **terrain parallax** (the satellite looks from the south at ~47° zenith angle, so elevated terrain appears shifted north). It can be tested with a DEM.

## B11. Training choices (§6.1, `data_pipeline.md` §8)
- **32 × 32 crops** (~100 × 145 km) for training; the network is fully convolutional, so it runs on the full grid at test time with no tiling.
- **Positive crops:** one per target, randomly shifted ±8 px so the target is not always in the centre.
- **Hard negatives:** crops around real detections in periods without any target ("fire present, but no new ignition"). 2 per positive in the final configuration (the only consistent ablation gain).
- **Flips** (horizontal/vertical) as augmentation; they are fine once the geo-offset is fixed.
- AdamW, cosine learning rate, 20 epochs, 3 seeds. Validation plateaus after ~12 epochs, and since the correction best and last epoch barely differ, so the final models use the last epoch.

## B12. Ablation study: method and main findings (§6.7–6.15)
- **Protocol:** change *one* setting relative to a reference, 3 seeds each. Report the best epoch and the mean of the last 5 evaluations (the honest number), with rough Welch t-values.
- **Lesson 1:** 15-min steps looked better with 1 seed (v2 vs. v1) but not with 3 seeds. **Single runs are not enough.**
- **Lesson 2:** after the geo-fix, **all architecture variants lie within ±0.03 AP**. The model is label/data-limited, not architecture-limited.

| Question | Answer (corrected data) |
|---|---|
| Is recurrence (ConvLSTM memory) needed? | **No.** A 3D CNN is as good or slightly better on validation, and equal on test |
| Bigger model? | **No.** 16 channels ≥ 32; 1 layer ≈ 2 layers |
| Does FRP intensity matter? | **Little.** Detection presence alone is only ~0.01 lower. *Where and when* fires appear carries most of the information |
| Does the 6 h history help? | A little (+0.02) |
| Readout over all steps vs. the last one? | Max over time helps a bit at 10 km |
| Does the clustering help as input? | No |
| Hard negatives | 2 per positive: consistent +0.015 |
| Target width σ | 1 px is fine; 0.5 px's earlier "gain" was an artefact of the offset |
| Peak extraction | 3 × 3 local maxima + sub-pixel centroid; the threshold is irrelevant for AP |

## B13. Final configurations and test result (§6.15, §6.17)
- `cnn3d_final` (3D CNN, 2 hard negatives) = main model. `conv_final` (ConvLSTM, 16 ch, max readout, 2 hard negatives) = the ConvLSTM of the proposal.

| Test 2017–2019 (3 seeds) | AP @ 5 km | AP @ 10 km | Recall @ 10 km | False alarms/day @ best F1 | Median error |
|---|---|---|---|---|---|
| baseline (nucleus birth) | 0.501 | 0.593 | 0.856 | 0.150 | 1.86 km |
| 3D CNN | **0.623 ± 0.012** | **0.711 ± 0.004** | **0.938** | 0.105 | 1.96 km |
| ConvLSTM | **0.620 ± 0.013** | **0.714 ± 0.013** | 0.933 | 0.088 | 2.00 km |

Key messages:
1. Clear gain over the baseline (+0.12 AP at both radii); validation numbers replicate on test, so there was **no overfitting to validation**.
2. 3D CNN ≡ ConvLSTM on test.
3. Gains are largest for **fires with neighbours** (recall @ 5 km 0.83 vs. 0.73) and **large fires** (≥ 1,000 ha, +10–12 points).
4. **The model does not localise better per matched fire** (~2.0 vs. 1.9 km). It finds and ranks the right ignitions better. ~1.6 km is the floor: the nearest detected pixel centre is a median 1.6 km from the ICNF point.
5. The geo-correction generalises: 2018 (after the EUMETSAT change) works very well (AP @ 5 km 0.85).

---

# Part C: the harder concepts

## C1. Heatmap targets (what the network is asked to output)
- Per pixel, a number between 0 and 1: "did a ≥ 50 ha fire start here in this 12 h period?"
- The target is not a hard 0/1 at a single pixel. It is a **Gaussian bump** (σ = 1 px) around the sub-pixel ICNF point, with the maximum where bumps overlap.
- *Why soft:* (a) ICNF points are uncertain by ~1–2 km and pixels are coarse, so being one pixel off should be only mildly wrong; (b) a soft target carries the sub-pixel position (the bump is asymmetric over the neighbouring pixels), which the decoder recovers.
- **Loss weight map w:** 1 in Portugal, 0 in Spain, 0 within 2 px of ignore fires, but always 1 right at a target. ("Targets win over ignore", otherwise 11 % of targets had no loss at their own pixel.)

## C2. Loss function: CenterNet focal loss
**Intuition first.** In a 32 × 32 crop there are ~1,000 pixels and usually 1–2 ignitions. With plain binary cross-entropy (BCE), the loss is dominated by the ~1,000 easy "nothing here" pixels. The network learns "predict 0 everywhere" and is mostly right. The focal loss fixes three things at once:

1. **Down-weight easy examples.** A positive pixel already predicted with p = 0.95 contributes almost nothing: factor (1 − p)². A background pixel already at p ≈ 0 contributes almost nothing: factor p². The gradient goes to the hard cases.
2. **Near-misses are cheap.** For background pixels the penalty is multiplied by (1 − y)⁴. Right next to an ignition y ≈ 0.6, so the penalty is tiny. Far away y = 0, so the penalty is full. A prediction one pixel off is "almost right", not "completely wrong".
3. **Per-ignition normalisation.** The sum is divided by the number of positive pixels, so a crop with 1 or with 10 ignitions gives a comparable loss scale.

**Formula** (α = 2, β = 4, Zhou et al. 2019 / CornerNet), with p = predicted probability, y = Gaussian target, w = weight map, positive = y ≥ 0.75:

- positive pixels: −(1 − p)^α · log p
- other pixels: −(1 − y)^β · p^α · log(1 − p)
- L = Σ w·(…) / (number of positive pixels)

Plus: the output bias is initialised so that p = 0.01 at the start ("ignitions are rare"), which stabilises the first steps (as in RetinaNet).

**What the ablations say** (⚠ only run on the *uncorrected* data, §6.11): BCE and MSE give higher AP @ 5 km but lower recall (0.81 vs. 0.87). They produce sharper, more confident peaks but miss more fires. That is a trade-off, not a clear win, so the focal loss stayed. *Honest point: this ablation should be repeated on the corrected data before it goes into the thesis.*

**If asked "why not regress coordinates directly?"** The number of ignitions per sample varies (0 to 31). A heatmap handles that without assigning predictions to targets during training, and it is the standard approach in keypoint detection.

## C3. From heatmap to points (decoding)
- Local maxima in 3 × 3 windows with p ≥ 0.05, at most 50 per period.
- **Sub-pixel position:** probability-weighted centroid of the 3 × 3 neighbourhood. This was worth +0.027 AP @ 5 km.
- Pixel (i, j) → lat/lon via the geostationary projection (and the half-pixel shift back for pre-2018 periods).
- A better decoder (log-parabola, "DARK") and a 3-seed ensemble gave a further +0.02 AP @ 5 km on validation. **They were not used in the final test run** (see D1).

## C4. Evaluation: matching and AP
- **Matching** (per period): predictions sorted by score; each takes the nearest unmatched ICNF target within k km → hit. If it is within k km of an ignore fire → neither hit nor false alarm. Otherwise → false alarm.
- **k = 5 km** ≈ one pixel diagonal (strict, location quality). **k = 10 km** = "found the right fire".
- **Average precision (AP)** = area under the precision–recall curve. It measures how well the scores *rank* true ignitions above false ones, over all thresholds. There is no threshold to choose, and it is the standard in detection tasks.
- Also reported: recall with all predictions, best F1 and its false alarms per day (an operational view), median error of the hits, and recall by subgroup (isolated / with neighbours / size class / year).
- **Why AP and not just mean distance:** distance alone ignores missed fires and false alarms, and the task is detection *and* localisation.

## C5. ConvLSTM (model 1 of the proposal)
- An LSTM where every matrix multiplication is a **3 × 3 convolution** (Shi et al. 2015). Hidden state and memory are **maps**: one memory per pixel, updated from its neighbourhood.
- Gates: forget (what to keep), input (what to write), output (what to expose). Per hour: c_t = f ⊙ c_{t−1} + i ⊙ g; h_t = o ⊙ tanh(c_t).
- *Why it fits on paper:* an ignition is a space–time pattern. "This pixel lit up first, then the fire spread to its neighbours" could be stored in the per-pixel memory.
- **Receptive field:** each 3 × 3 conv adds one pixel per direction within a step. Over time, the recurrence spreads information one pixel per step.
- Architecture: per-step encoder (2 convs) → 1–2 ConvLSTM layers → head on the final (or max over time) hidden state → heatmap. ~140k parameters in v1; the final ConvLSTM (2 layers × 16 channels) has ~45k.

## C6. 3D CNN (the model that matched or beat it)
- Three **3D convolutions** over (time, height, width), with stride 2 along time twice (24 → 6 time steps). Then **max and mean over time** → 2D head → heatmap. No recurrence.
- *Why it works as well:* the relevant signal sits in a **short space–time window** (where a pixel lit up first, what was around it before). A 3 × 3 × 3 kernel stack sees that directly, with no need to carry information through 18 recurrent steps.
- Consistent with the literature: recurrence-free models (SimVP/TAU in the OpenSTL benchmark) are competitive with ConvLSTM.
- **Thesis message:** "the recurrent memory brings no advantage, but no disadvantage either". That is a valid answer to objective (iii), "assessing modelling strategies".

## C7. Statistics: seeds, "best" vs. "last5", t-values
- AP @ 5 km varied by ±0.03–0.05 between epochs and seeds before the geo-fix, so differences of 0.02 were invisible with 1 run.
- Picking the best of 10 validation epochs inflates the score. The "last5" mean is the honest comparison.
- Welch t-values with n = 3 are **only indicative**. Across 12 comparisons a single t ≈ 2 can occur by chance, so I rely on patterns that are consistent across all four measures.

---

# Part D: caveats, open points, questions

## D1. Honest caveats (better to say them myself)
1. **Results are conditional on SEVIRI seeing the fire** (≥ 50 ha, detection within 5 km in the first 6 h). About 19 % of ≥ 50 ha fires are unseen and not counted. Recall against *all* fires should be reported too.
2. **ICNF positions are uncertain.** 72 % of all ICNF fires share exact coordinates with other fires (locality geocoding). For ≥ 100 ha it is only 10 %, but the accuracy is unknown. The ~2 km error is close to this floor.
3. **The test set is small and dominated by 2017** (439 targets; 2018 has only ~35). Per-year numbers are noisy.
4. **The improved decoder/ensemble were not pre-registered for the test run.** The test used single models with the 3 × 3 centroid. Applying them now would be post-hoc. Either report them as such or leave them out of the headline.
5. **The loss-function ablation (BCE/MSE vs. focal) was only run on the uncorrected data.**
6. Probabilities are **not calibrated**: they rank well, but p = 0.5 does not mean 50 %.
7. **No cloud mask:** "no detection" can mean no fire, or a fire hidden by cloud or smoke (Pedrógão Grande was invisible for 16 h).
8. `thesis.md` (the context file) is outdated (it stops at v1). `notes.md` is current.

## D2. Status vs. proposal objectives

| Objective | Status |
|---|---|
| (i) Spatiotemporal dataset FRP + environmental data | FRP part done (native grid, labels, sample index). **Covariates not yet** (DEM, land cover, ERA5-Land) |
| (ii) Design/evaluate DL architectures | ConvLSTM + 3D CNN done and evaluated on test. **CNN + Transformer (model 2) not yet** |
| (iii) Compare modelling strategies | Largely done: raw vs. clustered input, recurrent vs. convolutional, time resolution, losses, sampling |
| (iv) Contribution of environmental variables | **Open** (needs covariates) |
| (v) Robustness, accuracy, interpretability | Accuracy and robustness by subgroup/year done. **XAI open** |
| Extension to Europe | Not started (optional) |

## D3. Proposed next steps (to agree on priorities)
Ordered by value/cost, mostly from `model_review.md` §8:
1. **Space–time ignition output** (predict *when* as well as *where*). ~70 % of the remaining misses are pairs of ignitions within 2 px in the same period, typically ~1.7 h apart. A 2D map cannot separate them; a (t, i, j) map could. It also gives an ignition-time estimate.
2. **Terrain-parallax test with a DEM** (cheap; it may explain the residual 0.2 px offset).
3. **Covariates** (wind, DEM/slope, land cover, FWI) as ablations → objective (iv). The expected effect on localisation is modest.
4. **Model 2: CNN + temporal attention** (U-TAE-style, kept small): parity is expected. Its value is the proposal and explainability (attention shows which hours mattered).
5. **Use the 10–50 ha fires** as an auxiliary/size output. 30 % of the "false alarms" are real fires < 10 ha, and the 10–50 ha fires would roughly double the labels.
6. **Explainability:** gradient/occlusion maps on the 3D CNN, attention maps for model 2.
7. Leave-years-out cross-validation for the final comparison; SEVIRI 2020+ as an extra test.

## D4. Questions to ask the supervisor
1. **Framing:** is the heatmap formulation (all ignitions per 12 h window, ≥ 50 ha, "seen" fires) acceptable as the main task? Early detection (short window) vs. post-fire reconstruction (long window): which matters more for you?
2. **Data:** can we get **SEVIRI FRP 2020–2023/24** (LSA SAF FRP-PIXEL; 2022 was a big fire year)? A **cloud mask** (SEVIRI CLM)? Does ICNF have a **rekindle ("reacendimento") flag** or burned-area perimeters?
3. **Positional accuracy of ICNF ignition points:** is there any known estimate? Who records them?
4. **Priorities** for the remaining time: covariates/objective (iv) vs. Transformer vs. space–time output vs. XAI. Is a Europe extension still expected?
5. **Geo-referencing finding:** is it worth a short dedicated section (or even a note to the data providers)? Do you know the exact EUMETSAT change date in Dec 2017?
6. Contacts: Benali et al. (2016) is the closest predecessor (rule-based ignition points for Portugal from polar orbiters, ~2 km / ~12 h agreement). Do you know the group? (Paugam/Filippi/Benali 2026 on MTG-FCI fire tracking.)
7. Writing: timeline, structure, how much of the exploration/data cleaning goes into the main text?

## D5. Likely questions and short answers

**"Why not just take the first hotspot?"** That is the baseline. It works for isolated fires (recall @ 10 km 0.96) but fails when fires burn next to each other (0.81). The model's gain is there (0.91), and it ranks better (higher AP, fewer false alarms).

**"Why is the median error not better than the baseline?"** Both are near the floor set by the pixel size (1.5 km pure quantisation) and the ICNF accuracy. The model's advantage is finding the *right* ignitions and ranking them, not sub-pixel precision for the easy ones.

**"Why does the 3D CNN do as well as the ConvLSTM, and is the ConvLSTM then pointless?"** The informative pattern is short and local in space–time, so convolutions over a 24 h window capture it. That is a result in itself (objective iii). The ConvLSTM is kept as the model of the proposal and reaches the same level.

**"Isn't the dataset too small for deep learning?"** ~2,000 training targets, but each crop holds many pixels and often several targets, and the final models are small (~45k parameters for the ConvLSTM, ~63k for the 3D CNN). Validation → test generalisation shows no overfitting. Saturation across variants suggests a label limit, not a capacity limit.

**"How do you know the geo-offset is real and not a label artefact?"** It is a sharp step change at one date in all regions, of exactly the documented size and direction (half a pixel N/W), measured independently from fire records. After the correction the residual is the same before and after the change.

**"Why chronological split?"** Simultaneous fires of one outbreak would otherwise leak across splits, and it mimics operational use.

**"Why ignore regions, and could they hide errors?"** Fires of 10–50 ha and unseen fires may or may not produce hotspots. Punishing either outcome would be teaching noise. They are excluded from training and evaluation alike (predictions near them count as neither hit nor false alarm). Ablation: radius 0/2/4 px makes little difference.

**"What does FRP add if presence alone is almost as good?"** Mostly the *timing and spread pattern* of detections locates the ignition. FRP helps a little with ranking (large fires). This is an answer to the proposal's question about what FRP time series contain.

**"What is AP @ 5 km in plain words?"** If you sort all predicted ignition points from most to least confident, AP summarises how often the top of the list is a real ignition within 5 km, averaged over all cut-offs. 1.0 is perfect, and the baseline is 0.50.

**"Could this run operationally?"** Inference on the full domain takes well under a second per 12 h period. The limits are SEVIRI's pixel size, cloud/smoke and the 6 h tail delay. MTG-FCI (1 km, 10 min, since 2025) would be the natural successor input.

---

# Part E: connections to the literature

Sources: `literature.md` (verified reference database, 2026-10-20) and `model_review.md` §5 (web scan 2023–2026). **Status** says how far each claim has been checked: *full text* = read in the PDF; *verified* = the reference exists (Crossref/arXiv), content summarised from memory; *abstract* = only abstract or snippets seen. Anything not "full text" must be checked before it is cited for a specific claim.

## E1. Where the thesis sits (the positioning sentence)
The literature has four strands, and none of them **learns ignition locations from geostationary FRP sequences** (an inferred gap; confirm with a final Scopus/WoS search before writing "first"):
1. **Geostationary active-fire detection and FRP** (Wooster 2015; Roberts 2008/2014; Koltunov 2016; Filizzola 2016). These ask *when* a fire is first seen, not *where exactly* it started.
2. **Rule-based fire events and ignition points** (Benali 2016; Andela 2019 Global Fire Atlas; Oom 2016; Balch 2020 FIRED; Artés 2019 GlobFire; Chen 2022 FEDS). The ignition is the earliest pixel of a cluster, built mostly from daily burned-area products.
3. **Deep learning on fire time series** (Huot 2022; Gerard 2023 WildfireSpreadTS; Kondylatos 2022). These are mostly *forward* problems: spread and danger. Pinto 2020 (BA-Net) is the exception: it learns *when* each pixel burned, which is an inverse timing problem.
4. **Generic methods**: ConvLSTM, temporal attention (U-TAE), heatmap keypoints (CornerNet/CenterNet/DARK).

**Framing for the introduction:** the spread-prediction literature asks where a fire *will* burn; this thesis asks where it *started*. The same covariates appear in both, with the roles flipped. Data assimilation (Rochoux 2015) gives the formal language: ignition reconstruction as state estimation from incomplete observations.

## E2. Concept → literature

| Thesis concept / decision | Literature | How it connects | Status |
|---|---|---|---|
| SEVIRI FRP product, confidence, detection limits | Wooster et al. 2015 (LSA SAF FRP Part 1, ACP); Roberts et al. 2015 (Part 2); Freeborn et al. 2014; Hall et al. 2019 | The product the extract comes from. Omission of low-FRP fires supports keeping low-confidence detections. Comparison points for the detection-by-size curve (fig. 07) | verified |
| FRE as a fire-size score (baseline ranking) | Wooster et al. 2005 (JGR) | FRE ∝ biomass burned, which justifies FRE as a proxy for size | verified |
| ICNF database | Pereira et al. 2011 (NHESS); Catry et al. 2009 | History and statistics of the Portuguese fire database; ICNF ignition points used as data. The coordinate anomalies are your own audit | verified |
| Detection latency (time-zone result) | Koltunov et al. 2016 (GOES-EFD); Hall et al. 2019 | Geostationary detection lags vs. reports, to compare with your +10–20 min | verified |
| Clustering thresholds (6 km / 6 h) and sensitivity grid | Oom et al. 2016; Balch et al. 2020 (FIRED); Artés et al. 2019 (GlobFire) | All report that event definitions depend strongly on the distance/time thresholds. This turns your "raw input instead of clustering" argument into a recognised limitation of the rule-based paradigm | verified |
| Nuclei and nucleus-birth baseline | Andela et al. 2019 (Global Fire Atlas); Chen et al. 2022 (FEDS, VIIRS tracking every 12 h with merges/splits); Benali et al. 2016 | "Ignition = earliest point of a fire object" is the standard rule. Your baseline applies it to SEVIRI, which makes it a fair baseline | verified |
| **Closest predecessor** | **Benali et al. 2016** (*Remote Sens.* 8:326) | Same country, same ICNF ground truth, ignition points and dates from polar-orbiting active fires, rule-based; ~2 km / ~12 h agreement. Your model: ~2 km with 15-min data, on a different fire set (compare carefully) | verified, numbers abstract |
| Successor-sensor tracking | Paugam, Filippi, Benali et al. 2026 (MTG-FCI Fire Event Tracker) | Rule-based tracking on 1 km / 10 min FCI data from the Portuguese community: a possible contact and data source | abstract |
| Ignition heatmaps (multi-object keypoints) | Zhou et al. 2019 (CenterNet, "Objects as points"); Law & Deng 2018 (CornerNet) | Objects as Gaussian heatmap peaks and peak extraction; CenterNet also has an offset head for sub-pixel position | verified |
| Focal loss | Lin et al. 2017 (RetinaNet focal loss); Law & Deng 2018 (penalty-reduced variant, α = 2, β = 4) | Your loss is CornerNet's. Cite all three (notes §6.1 currently cites only CenterNet). The prior-0.01 bias initialisation is from Lin 2017 | verified |
| Sub-pixel decoding | Zhang et al. 2020 (DARK, CVPR); Chatzitofis et al. 2021; SimCC (ECCV 2022) | Log-Gaussian/Taylor decoding is exact for Gaussian peaks; it gave +0.015 AP @ 5 km on validation | verified (DARK) |
| Merged close ignitions | Song et al. 2021 (P2PNet, point-set prediction) | Predict sets of points with one-to-one matching, an alternative to heatmap peaks for dense outbreaks | verified |
| ConvLSTM (model 1) | Shi et al. 2015 (NeurIPS); Shi et al. 2017 (TrajGRU); Burge et al. 2020 (ConvLSTM on simulated fire spread, preprint) | Model 1 of the proposal | verified |
| **3D CNN ≥ ConvLSTM** | OpenSTL benchmark (Tan et al. 2023): recurrent-free SimVP/TAU competitive with ConvLSTM; **Pinto et al. 2020 (BA-Net)**: factorised 3D convolutions + LSTM, with an ablation without the LSTM | Your result agrees with the benchmark literature. BA-Net asks the same question (is the recurrence needed?) on satellite time series | full text (Pinto 2020) |
| Model 2: CNN + Transformer | Vaswani et al. 2017; **Garnot & Landrieu 2021 (U-TAE)**; Tarasiou et al. 2023 (TSViT) | U-TAE is the concrete, small template: a 2D encoder per step and temporal attention masks that collapse time. Its attention maps double as explainability | verified |
| Geostationary deep learning | Zhao & Ban 2022 (GOES-R GRU, early detection); Kang et al. 2022 (Himawari DL detection); McCarthy et al. 2021 (U-Net downscaling of GOES to VIIRS-like resolution) | Precedents for DL on geostationary fire time series; McCarthy is the direct precedent for learned sub-pixel information | verified |
| Chronological split, leave-years-out | Roberts et al. 2017 (*Ecography*, blocked CV); Gerard et al. 2023 (year folds in WildfireSpreadTS) | Justifies the time split and the proposed year-fold CV | verified |
| ≥ 3 seeds, mean ± std | Bouthillier et al. 2021 (variance in ML benchmarks) | Justifies the ablation protocol (and the 15-min lesson) | verified |
| Metrics depend on choice | WildfireSpreadBench 2026 | Model rankings change between AP and F1/IoU, so report both AP and best F1 / false alarms | abstract |
| Calibration | Guo et al. 2017 | Temperature scaling / isotonic regression if probabilities are shown to users | verified |
| Space–time output / ignition time | **Pinto 2020 (per-pixel dating)**; Farguell et al. 2021 (fire-arrival-time surface); Mandel et al. 2018; Shaddy et al. 2025 | The fire as a per-pixel arrival-time field, with the ignition as its earliest point. Supports predicting (t, i, j) and a "burn time" target | full text (Pinto) / verified |
| Erase augmentation, clouds and smoke | Pinto 2020 (random erasing of the active-fire channel); Yang et al. 2026 (fire spread under partial observability); Che et al. 2018 (GRU-D: masks + time since observation) | Augmentation modelled on the sensor's failure modes (the Pedrógão omission); observability channel | full text / abstract / verified |
| Self-supervised pre-training | HighFM (Girtsou et al. 2026, SEVIRI masked autoencoder); SSL hotspot detection on SEVIRI (2024) | 568k unlabelled detections against ~2,000 targets | abstract |
| Covariates (objective iv) | Huot et al. 2022 (Next Day Wildfire Spread); Gerard 2023; Kondylatos et al. 2023 (Mesogeos datacube, Mediterranean incl. Portugal); DaCamara et al. 2014 and Pinto et al. 2018 (FWI calibrated with SEVIRI FRP) | Templates for covariate channels. Mesogeos may be a shortcut. FWI is a single, already-calibrated first covariate | verified |
| Explainability (objective v) | Sundararajan et al. 2017 (Integrated Gradients); Adebayo et al. 2018 (sanity checks for saliency); Kondylatos et al. 2022 (XAI on wildfire DL) | Attribution maps from a zero-fire baseline, checked with model-randomisation tests | verified |
| Noisy point labels (ICNF accuracy) | Bernhard & Schubert 2021; noised-autoencoder point refinement (2023) | Learned label refinement once parallax is corrected | abstract |
| Long-term memory (concept, §6.6) | Ballas 2016 (ConvGRU); Dai 2019 (Transformer-XL); Wang 2019 (E3D-LSTM); Skip/Clockwork/Phased RNN | Background for the memory options (currently deprioritised) | verified |
| Geo-referencing offset | satpy SEVIRI L1.5 reader docs; EUMETSAT service notices (2017) | Documentation, not papers: "shifted by 1.5 km north and west until Dec 2017". The exact change day was not found | docs |
| 2017 test year context | Turco et al. 2019 (climate drivers of the 2017 fires); Tedim et al. 2018 (extreme wildfire events) | One context sentence each | verified |

## E3. Points to raise
- **The IDL Lisbon / IPMA group** (Pinto, DaCamara, Trigo, Amraoui) works with the same LSA SAF SEVIRI FRP product over Portugal. Does the extract come from there? It could give access to product documentation, a cloud mask and data from 2020 on.
- **Benali 2016 as an external benchmark:** report the baseline and the model with Benali's quantities (distance and time error for large fires) so the thesis can be placed against it. This needs a close reading of their definitions.
- **Global Fire Atlas vs. ICNF** (Andela 2019): the distance between GFA ignitions and ICNF points gives an independent estimate of ICNF positional noise, i.e. the error floor.
- **Still without a source:** "ICNF records rekindles as new fires", the ICNF positional accuracy, and the "Xu et al. 2025" reference of the proposal (no such review found; ask the supervisor for the full reference).
- An earlier LLM-generated literature draft had fabricated or wrong references (4 wrong DOIs, 1 non-existent paper, 5 composites). All references above come from the verified database.

---

## Numbers cheat sheet

| | Value |
|---|---|
| SEVIRI detections (clean) | 568,123, 2004–2019, 15 min, ~3 × 4.5 km pixels |
| ICNF fires with coordinates, 2004–2019 | 361,091; ≥ 100 ha: 2,311 |
| Targets (≥ 50 ha, seen, after geo-fix) | 2,821 → train 1,979 / val 403 / test 439 |
| Grid | 177 × 106 native pixels, 8,402 with detections |
| Sample | 24 h input (6 + 12 + 6), 24 hourly steps, 7 channels, 32 × 32 training crops |
| Geo-offset | 0.5 px N/W before Dec 2017 (≈ 1.5 km at SSP, ≈ 3.3 km over Portugal) |
| Localisation floor | ~1.5–1.6 km (pixel quantisation / nearest detected pixel) |
| Test AP @ 5 / 10 km | baseline 0.50 / 0.59 → models 0.62 / 0.71 |
| Recall @ 5 km, fires with neighbours (test) | 0.73 → 0.83 |
| Training cost | ~30–45 min per run on the M1 (20 epochs) |

## Figures to have open
- `figures/pipeline/pipeline_overview.png`: the whole pipeline on one page
- `figures/largest_fires_lat_lon_time.png`: "feel the data" result
- `figures/clustering/events_2017-10-15_panels.png`: why neighbouring fires are hard
- `figures/eda/07_detection_probability.png`: why ≥ 50 ha
- `figures/pipeline/sample_labels.png`: targets, ignore regions, loss weight
- `figures/pipeline/time_aggregation.png`: hourly aggregation
- `figures/pipeline/georef_frames.png`: the geo-referencing correction
- `figures/model/model_cnn3d.png`, `figures/model/model_convlstm.png`: architectures
- `figures/model/convlstm_v1_predictions_val_zoom.png`: predicted vs. true maps (merged close ignitions)
- `figures/ablations_georef/summary.png`: ablations on corrected data
- `figures/final/test_pr.png`: final test precision–recall curves
