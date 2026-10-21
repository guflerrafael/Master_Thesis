# Master Thesis — Literature

> Purpose: the literature base of the thesis, in the same spirit as `thesis.md` (context) and `notes.md` (methods log). It consolidates a literature-review draft produced with Google Gemini (Downloads, file "Masters — ## Role and Objective …", 25 publications P01–P25), an evaluation of that draft, a verified and extended reference database, and a first alignment of the literature with what the thesis has implemented and plans to do.
>
> Created 2026-10-20. Part I (§1–4) = evaluation and database; Part II (§5–9) = application to the thesis, reading order and next steps. Section references like "notes §6.6" point to `notes.md`, "thesis §9" to `thesis.md`.

**Tags used below**
- **[verified]** — title, authors, year, venue and DOI checked against the Crossref REST API (DOI lookup or bibliographic query) or the arXiv API on 2026-10-20.
- **[full text]** — content read from the PDF in `documents/papers/`.
- **[memory]** — content summary from general knowledge, *not* checked against the paper. Must be confirmed in the full text before it is cited for a specific claim.
- **[unverified]** / **[not found]** — could not be confirmed / no matching record exists.
- *Cites* = Crossref `is-referenced-by-count` on 2026-10-20 (counts citations from Crossref-registered works only, so it is lower than Google Scholar; "—" = no DOI, typical for NeurIPS/ICML/ICLR papers).
- *Recency* (relative to 2026): VR very recent ≤ 2 y (2024–2026), R recent 3–5 y (2021–2023), O older 6–10 y (2016–2020), F foundational > 10 y (≤ 2015).
- *C / A* = connectedness / applicability, 1–5 (as defined in the draft): C = similarity of problem, data, task, method and evaluation; A = how directly results, methods, data or experimental design can be used in this thesis.

---

# Part I — Evaluation of the draft and verified database

## 1. Evaluation of the Gemini draft (2026-10-20)

### 1.1 What the draft contains
- **A. Thesis research profile** — summary of problem, objectives, data, task, methods, metrics and current results. Built from `thesis.md` and `notes.md`, and accurate with respect to them (numbers match notes §3–6.7).
- **B. Search methodology** — 5 query groups (SEVIRI FRP & Iberia; hotspot clustering & ignition reconstruction; DL for fire evolution; heatmap keypoints & multi-timescale memory; ICNF & rekindles), inclusion/exclusion criteria, a "verification protocol".
- **C. Literature database (CSV)** — truncated in the export (every row cut after the first author), not usable.
- **D. Prioritised matrix** (15 papers), **E. ranked critical assessment** (25 papers in 3 tiers), **F. landscape and gap analysis**, **G. reading list + 3 experiment recommendations**.

### 1.2 Problems found during verification
Verification: every DOI and arXiv ID of the draft was looked up (Crossref/arXiv, 2026-10-20). Result: of the 25 entries, **4 have a wrong or non-existent DOI (one of them also a non-existent paper), 1 more could not be found, 1 has a wrong arXiv ID, 5 are composite entries** that merge 2–3 papers into one ID (P20, P21, P23, P24, P25), and several papers are real but their content is misdescribed.

| Draft ID | Draft says | Verification | Consequence |
|---|---|---|---|
| P04 Benali et al. 2016 | "Determining the location and time of ignition of wildfires from MODIS and VIIRS active fire data", *Remote Sens. Environ.*, 10.1016/j.rse.2016.06.008 | DOI resolves to an unrelated paper (Shi et al., MODIS radiative transfer). Real paper: **"Determining fire dates and locating ignition points with satellite data"**, *Remote Sensing* 8(4):326, 10.3390/rs8040326 [verified] | Use the real reference; the reported numbers ("1.5–4 km") are unverified. |
| P10 Xu, Wooster et al. 2021 | "Tracking wildfire progression and ignition points using geostationary satellite active fire observations", *RSE*, 10.1016/j.rse.2020.112160 | **No such paper.** DOI resolves to an InSAR study (Jiang & Lohman 2021). Its "findings" (detection 15–45 min after ignition) restate the thesis's own time-zone result (notes §1.4). | Delete. Nearest real works: Roberts & Wooster 2014, Koltunov et al. 2016, Hall et al. 2019 (§3.1). |
| P05 Turco et al. 2019 | DOI 10.1038/s41598-019-47223-9; CSV title "Extreme wildfire events are more likely in warm and dry conditions…" | DOI does not exist. Real: **"Climate drivers of the 2017 devastating fires in Portugal"**, *Sci. Rep.* 9:13886, 10.1038/s41598-019-50281-2 [verified] | Use for climate context only. The claim that the Pedrógão pyroCb "blocked 3.9 µm radiation 18:00–22:00 UTC" is not from this paper (it is a climate-attribution study) and must not be cited. |
| P06 "Gomes Da Camara et al. 2018, Structural and weather determinants…" | — | **Not found.** The real SEVIRI-FRP/Portugal works of that group are Pinto et al. 2018 (NHESS) and DaCamara et al. 2014 (IJWF) [verified] | Replace. |
| P08 Laurent et al. 2018 (FRY) | *Earth Syst. Sci. Data*, 10.5194/essd-10-1511-2018; "flood-fill clustering of satellite fire pixels, nucleus birth tracking" | DOI does not exist. Real: *Scientific Data* 5:180132, 10.1038/sdata.2018.132 [verified]. FRY is built from **burned-area** products (MCD64A1, FireCCI) [memory]; "nuclei" is this thesis's own term (notes §3.2), not FRY's. | Cite as a fire-patch database from burned area, not as the origin of the nucleus method. |
| P03 Wooster et al. 2003 | "…comparison to MODIS **and SEVIRI**"; FRE ∝ biomass derived here | Real title ends "…comparison to MODIS fire products" [verified]. The FRE–biomass calibration (0.368 kg MJ⁻¹) is from **Wooster et al. 2005**, *JGR* 110:D24311 [verified; coefficient from memory]. | Cite 2003 for the MIR method, 2005 for FRE → biomass. |
| P07 Pereira et al. 2011 | "A database of 1980–2005 Portuguese rural fires"; "independently confirms every data-quality anomaly of notes §1.3/§2.3 (72 % shared coordinates…)" | Real title: "The history and characteristics of the 1980–2005 Portuguese rural fire database" [verified]. Covers 1980–2005, i.e. mostly before ICNF coordinates exist (2001+); it cannot confirm the coordinate sharing, the 1994 end-date corruption or the placeholder areas. | Cite for database history and size distributions; the anomalies remain the thesis's own audit findings. |
| P12 Zhou et al. 2019 (CenterNet) | "CVPR" | arXiv preprint only (arXiv:1904.07850) [verified]. The penalty-reduced focal loss was introduced by **CornerNet** (Law & Deng, ECCV 2018) on top of the focal loss of Lin et al. (ICCV 2017) [verified]. | Cite all three for the loss (notes §6.1 currently cites only CenterNet). |
| P18 Huot et al. 2022 | "Conv-autoencoders, ConvLSTM, ViT with focal loss on 32 × 32 / 64 × 64 crops" | Metadata correct [verified]. Content [memory]: single next-day step, 64 × 64 km at 1 km, models logistic regression, random forest and a convolutional autoencoder — no ConvLSTM/ViT. | Do not cite for ConvLSTM/ViT results. |
| P19 Gerard et al. 2023 (WildfireSpreadTS) | arXiv 10.48550/arXiv.2311.01940; "proves U-TAE beats ConvLSTM" | arXiv 2311.01940 is a hypergraph-colouring paper [verified]. Venue NeurIPS 2023 Datasets & Benchmarks [memory]. The "proves" claim is stronger than what I recall (multi-day input gave small, model-dependent gains). | Get the correct ID from the NeurIPS proceedings; check the claim. |
| P20 "Xu et al. 2025 / Jain 2020 / Ghali & Akhloufi 2023" | "resolves the Xu et al., 2025 placeholder" | Composite. Jain et al. 2020 and Ghali & Akhloufi 2023 are real [verified]; **no Xu et al. 2025 review was found** (Crossref query) — the placeholder (thesis §11, `infos.txt` line 3) is *not* resolved. | Ask the supervisor for the full reference. |
| P21 "di Biase & Laneve 2018 / Kang et al. 2022" | "RSE / IEEE TGRS" | Composite. Di Biase & Laneve 2018 is *Remote Sensing* 10(5):741; Kang et al. 2022 is *GIScience & Remote Sensing* 59(1) [verified]. Specific numbers (6–24 h background windows) unverified. | Cite separately. |
| P22 Tedim et al. 2018 | "Portuguese rekindles are logged as new ICNF fires" | Real (*Fire* 1(1):9) [verified], but the paper is about **defining extreme wildfire events**, not rekindles. | Do not cite for rekindles. "ICNF records rekindles as new fires" remains a hypothesis of notes §5.3 that needs its own source (see §4.3). |
| P23, P24, P25 | composite entries (Skip/Clockwork/Phased RNN; self-conditioning + scheduled sampling; Kondylatos + Integrated Gradients) | Each paper is real [verified]; self-conditioning is Chen, Zhang & Hinton, *ICLR 2023* (arXiv 2022) | Cite individually. |
| P01, P02 | correct metadata | [verified]. Specific numbers ("omits 40–60 % of small fires", "r > 0.90 FRE vs. biomass") unverified. P01 does not "explain" the native-grid reconstruction, which was derived in notes §4.2. | Check numbers in the full text. |
| all | citation counts "~N+ (Verified Offline)" | Guesses: e.g. Wooster et al. 2015 Part 1 "~350+" vs. Crossref 90; Tedim et al. 2018 "~300+" vs. 474; Turco et al. 2019 "~200+" vs. 293. | Replaced by Crossref counts in §3. |

Patterns behind the errors (relevant when using LLM-generated reviews again):
- **Circular "findings".** The "Authors' reported findings" often restate this thesis's own numbers as if the paper had found them (P10 detection lag, P02 "confirms Fig. 04/07/08", P07 "confirms §1.3", P22 rekindles, P05 pyroCb timing). None of these can be cited.
- **Over-attribution.** Many "Thesis link" claims say a paper "proves" or "validates" a design decision of the thesis. At most, the papers motivate or are consistent with them.
- **Nominal search.** The draft states that it could not query databases live; the "search methodology" is a plan, not a record.

### 1.3 What is usable
- The **thesis research profile** (A) — a correct condensed summary, useful for the introduction.
- The **five search pillars and query strings** (B) — kept as the search plan (§2).
- The **three-paradigm landscape** (F.1): rule-based space–time clustering; forward DL spread forecasting; static ignition susceptibility. Holds up with the verified papers (§4.1).
- The **three gap statements** (F.2): merging vs. fragmentation in clustering; reappearance vs. rekindle; merging of close heatmap peaks. Correct as descriptions of the thesis's own findings, but they are *inferred* gaps (from the thesis's results), not gaps stated by the cited authors.
- The **three recommendations** (G.4): CenterNet offset head, cloud mask with the memory, Integrated Gradients from a zero-fire baseline. All sensible (taken up in §6).

### 1.4 Verdict
Use the draft as a **map of topics, not as a source**. Every reference used in the thesis should come from §3, be tagged [verified], and have the cited claim checked in the full text. The draft also misses the most relevant local work: the two papers already in `documents/papers/` (Pinto et al. 2020, 2021, IDL Lisbon) and the ignition-point/fire-event literature built on burned-area products (Global Fire Atlas, FIRED, Oom et al. 2016), which is the closest methodological neighbour of the clustering and nucleus work (§3.3).

## 2. Search record (2026-10-20)

- **Search plan:** the five query groups of the draft (B), plus: "fire event individuation", "ignition point burned area", "fire arrival time active fire", "geostationary downscaling active fire", "SEVIRI Mediterranean fire activity", "temporal attention satellite image time series", "heatmap sub-pixel decoding", "point-based localisation".
- **Sources actually used:** Crossref REST API (DOI lookup and `query.bibliographic`), arXiv API (ID lookup), the two PDFs in `documents/papers/`. No access to Scopus, Web of Science or Google Scholar; no full texts other than the two local PDFs.
- **Inclusion/exclusion:** as in the draft (B.2): peer-reviewed journals and main ML venues; geostationary/polar active fire and FRP; Portuguese fire regime and ICNF data; fire-event and ignition reconstruction; DL on spatiotemporal fire data; the exact mechanisms used in `src/` and notes §6. Excluded: static susceptibility maps without temporal modelling, UAV/camera flame detection.
- **Not yet searched (to do):** rekindle statistics and the rekindle flag in the ICNF/SGIF database; positional accuracy of ICNF ignition points; SEVIRI cloud mask (CLM) and LSA SAF FRP-PIXEL product documentation (ATBD, product user manual); Meteosat Third Generation (MTG-FCI) fire products; **2024–2026 work** (only one verified entry is in the "very recent" class — the database is thin on the newest literature); forward-citation search of Benali et al. 2016 and Andela et al. 2019.

## 3. Literature database

Grouped by pillar. Keys are BibTeX-style for later use in the thesis. "Draft" = ID in the Gemini draft (— = new).

### 3.1 Active fire and FRP from (geostationary) satellites

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Content | Use in the thesis |
|---|---|---|---|---|---|---|---|
| Wooster2015 (P01) | Wooster, Roberts, Freeborn, Xu et al. — LSA SAF Meteosat FRP products, Part 1: algorithms, product contents and analysis. *Atmos. Chem. Phys.* 15 | 10.5194/acp-15-13217-2015 [verified] | 90 | F | 5/5 | The SEVIRI FRP-PIXEL product that `FRP_SEVIRI_PT_2004_2019.mat` comes from: detection (FTA), MIR radiance method, confidence, pixel-area growth with view angle. [memory] | Data chapter: what FRP and FIRE_CONFIDENCE mean, saturation, minimum detectable FRP. Check thesis §6.1 claims against it. |
| Roberts2015 (P02) | Roberts, Wooster, Xu, Freeborn et al. — LSA SAF Meteosat FRP products, Part 2: evaluation and demonstration for CAMS. *ACP* 15 | 10.5194/acp-15-13241-2015 [verified] | 47 | F | 4/4 | Evaluation of the product against MODIS; SEVIRI misses many low-FRP fires but captures the diurnal cycle. [memory] | Context for the detection-probability curve (fig. 07) and the FRE vs. burned-area result (fig. 08). |
| Roberts2008 | Roberts, Wooster — Fire detection and fire characterization over Africa using Meteosat SEVIRI. *IEEE TGRS* 46 | 10.1109/TGRS.2008.915751 [verified] | 192 | F | 3/3 | Original SEVIRI fire thermal anomaly algorithm. [memory] | Background: origin of the detection algorithm. |
| Roberts2014 | Roberts, Wooster — Multi-temporal Kalman filter approach to geostationary active fire detection & FRP estimation. *RSE* | 10.1016/j.rse.2014.06.020 [verified] | 54 | F | 4/3 | Uses the expected diurnal brightness-temperature cycle (temporal background) to detect fires in SEVIRI. [memory] | Support for temporal context separating fires from background; persistent-pixel channel. |
| Wooster2003 (P03) | Wooster, Zhukov, Oertel — Fire radiative energy … BIRD … comparison to MODIS fire products. *RSE* 86 | 10.1016/S0034-4257(03)00070-1 [verified] | 426 | F | 3/3 | MIR radiance method for FRP. [memory] | FRP definition. |
| Wooster2005 | Wooster, Roberts, Perry, Kaufman — Retrieval of biomass combustion rates and totals from FRP observations. *JGR* 110 | 10.1029/2005JD006318 [verified] | 611 | F | 3/3 | FRE ↔ biomass consumed calibration. [memory] | Justifies FRE as a fire-size proxy (notes §2.2, baseline score in §5.3). |
| Wooster2021 | Wooster, Roberts, Giglio, Roy et al. — Satellite remote sensing of active fires: history, current status, applications, future requirements. *RSE* 267 | 10.1016/j.rse.2021.112694 [verified] | 272 | R | 4/3 | Review of active-fire sensors, algorithms, FRP and applications. [memory] | Main background citation for chapter 2. |
| Freeborn2014 | Freeborn, Wooster, Roberts, Xu — Evaluating the SEVIRI fire thermal anomaly detection algorithm … using MODIS. *Remote Sens.* 6 | 10.3390/rs6031890 [verified] | 46 | F | 4/3 | SEVIRI detection probability vs. fire size/FRP compared to MODIS. [memory] | Comparison for fig. 07 (SEVIRI sees ≥ 80 % of fires ≥ 100 ha). |
| Hall2019 | Hall, Zhang, Schroeder, Huang, Giglio — Validation of GOES-16 ABI and MSG SEVIRI active fire products. *JAG* 83 | 10.1016/j.jag.2019.101928 [verified] | 34 | O | 4/3 | SEVIRI and ABI detections validated against higher-resolution data. [memory] | Omission/commission context for SEVIRI; detection-lag discussion. |
| Giglio2016 | Giglio, Schroeder, Justice — The Collection 6 MODIS active fire detection algorithm. *RSE* 178 | 10.1016/j.rse.2016.02.054 [verified] | 1376 | O | 2/2 | MODIS active fire product. | Only if FIRMS MODIS is used (European extension). |
| Schroeder2014 | Schroeder, Oliva, Giglio, Csiszar — The new VIIRS 375 m active fire product. *RSE* 143 | 10.1016/j.rse.2013.12.008 [verified] | 1038 | F | 3/3 | VIIRS I-band active fires (from 2012). | Finer reference detections for 2012–2019 (§6.2 item 9). |
| Calle2006 | Calle, Casanova, Romo — Fire detection and monitoring using MSG SEVIRI data. *JGR Biogeosciences* 111 | 10.1029/2005JG000116 [verified] | 56 | F | 3/2 | Early SEVIRI fire detection, Iberian Peninsula. [memory] | Background: SEVIRI over Iberia. |
| Laneve2006 | Laneve, Castronuovo, Cadau — Continuous monitoring of forest fires in the Mediterranean using MSG. *IEEE TGRS* 44 | 10.1109/TGRS.2006.881716 [verified] | 57 | F | 3/2 | SEVIRI fire monitoring, Mediterranean (SFIDE origin). [memory] | Background. |
| DiBiase2018 (P21a) | Di Biase, Laneve — Geostationary sensor based forest fire detection and monitoring: improved SFIDE. *Remote Sens.* 10 | 10.3390/rs10050741 [verified] | 32 | O | 3/3 | Improved SEVIRI detection using the temporal behaviour of brightness temperature. [memory] | Temporal context for early detection. |
| Filizzola2016 | Filizzola, Corrado, Marchese, Mazzeo et al. — RST-FIRES, early-fire detection … MSG-SEVIRI. *RSE* 186 | 10.1016/j.rse.2016.08.008 [verified] | 46 | O | 3/3 | Multi-temporal "robust satellite technique" for SEVIRI early fire detection, field-validated in Italy. [memory] | Early-detection framing (notes §6.4). |
| Amraoui2010 | Amraoui, DaCamara, Pereira — Detection and monitoring of African vegetation fires using MSG-SEVIRI. *RSE* 114 | 10.1016/j.rse.2009.12.019 [verified] | 70 | F | 2/2 | SEVIRI fire detection (Lisbon group). | Background. |
| Koltunov2016 | Koltunov, Ustin, Quayle, Schwind et al. — GOES Early Fire Detection (GOES-EFD) algorithm. *RSE* 184 | 10.1016/j.rse.2016.07.021 [verified] | 36 | O | 3/3 | Geostationary early detection using temporal context; detection latency vs. reports. [memory] | Detection-latency comparison (notes §1.4). |
| Wickramasinghe2016 | Wickramasinghe, Jones, Reinke, Wallace — Multi-spatial-resolution surveillance of active fire lines using Himawari-8. *Remote Sens.* 8 | 10.3390/rs8110932 [verified] | 62 | O | 3/3 | Sub-pixel fire-line localisation from geostationary data. [memory] | Sub-pixel localisation argument (notes §6.4 "Grid"). |

### 3.2 Portugal: fire regime, ICNF data, SEVIRI applications

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Content | Use in the thesis |
|---|---|---|---|---|---|---|---|
| Benali2016 (P04) | Benali, Russo, Sá, Pinto, Price, Koutsias, Pereira — Determining fire dates and locating ignition points with satellite data. *Remote Sens.* 8(4):326 | 10.3390/rs8040326 [verified] | 69 | O | **5/5** | Ignition points and dates of large Portuguese fires estimated from satellite active fires, compared with official records. [memory — check sensors, method and error numbers] | **Closest predecessor.** External benchmark for the baselines (notes §5.3) and the ConvLSTM; discussion of ICNF location/time errors. |
| Pereira2011 (P07) | Pereira, Malamud, Trigo, Alves — History and characteristics of the 1980–2005 Portuguese rural fire database. *NHESS* 11 | 10.5194/nhess-11-3343-2011 [verified] | 119 | F | 4/4 | Origin, structure and statistics of the Portuguese fire database (PRFD, ICNF's predecessor), size distributions. [memory] | Data chapter: ICNF background. |
| Catry2009 | Catry, Rego, Bação, Moreira — Modeling and mapping wildfire ignition risk in Portugal. *IJWF* 18 | 10.1071/WF07123 [verified] | 322 | F | 3/3 | Ignition-risk model from ICNF ignition points and human/land factors. [memory] | ICNF ignition points as data; human drivers of ignitions. |
| Turco2019 (P05) | Turco, Jerez, Augusto, Tarín-Carrasco et al. — Climate drivers of the 2017 devastating fires in Portugal. *Sci. Rep.* 9 | 10.1038/s41598-019-50281-2 [verified] | 293 | O | 3/2 | Drought and heat behind 2017; extreme events of June and October. [memory] | Context for the 2017 test year (notes §4.2). |
| Tedim2018 (P22) | Tedim, Leone, Amraoui, Bouillon et al. — Defining extreme wildfire events. *Fire* 1(1):9 | 10.3390/fire1010009 [verified] | 474 | O | 2/2 | Definition and classes of extreme wildfire events. [memory] | Size-class discussion; *not* for rekindles. |
| Pinto2018 | Pinto, DaCamara, Trigo, Trigo — Fire danger rating over Mediterranean Europe based on FRP derived from Meteosat. *NHESS* 18 | 10.5194/nhess-18-515-2018 [verified] | 39 | O | 4/4 | Fire-danger classes from LSA SAF SEVIRI FRP and FWI over Mediterranean Europe. [memory] | Same product and region; FWI as a candidate meteorological covariate (§6.2). |
| DaCamara2014 | DaCamara, Calado, Ermida, Trigo et al. — Calibration of the Fire Weather Index over Mediterranean Europe based on fire activity retrieved from MSG imagery. *IJWF* 23 | 10.1071/WF13157 [verified] | 42 | F | 4/3 | FWI calibrated with SEVIRI-derived fire activity. [memory] | Same as above. |
| Amraoui2013 | Amraoui, Liberato, Calado, DaCamara et al. — Fire activity over Mediterranean Europe based on Meteosat-8. *For. Ecol. Manage.* 294 | 10.1016/j.foreco.2012.08.032 [verified] | 68 | F | 4/3 | SEVIRI fire activity over Mediterranean Europe incl. Portugal: seasonal/diurnal cycle, regions. [memory] | Comparison for the exploration figures (fig. 04–06). |
| Pinto2020 | Pinto, Libonati, Trigo, Trigo, DaCamara — A deep learning approach for mapping and dating burned areas using temporal sequences of satellite images. *ISPRS J.* 160 | 10.1016/j.isprsjprs.2019.12.014 [verified, full text] | 118 | O | **4/5** | BA-Net: 3D U-Net with factorised spatial (1×7×7) and temporal (7×1×1) convolutions plus a per-pixel LSTM after the first spatial convolution; input 64 daily VIIRS images × 4 channels (red, NIR, MIR, **active fires**) × 128 × 128; output per day and pixel = burned probability → burned area **and date of burning**. Targets MCD64A1; augmentations: random patches on the reflectance channels (cloud-like noise), **random erasing of the active-fire channel to simulate omissions**. Ablation: LSTM improves over the same model without it. Dating error (MAE vs. VIIRS active fires) 1.13 days vs. 1.80 (MCD64A1) and 4.25 (FireCCI51); Portugal one of 5 regions, ICNF burned-area map as reference. [full text] | Closest DL precedent from the (probably) same research environment: spatiotemporal sequence → per-pixel time of an event. See §5–7. |
| Pinto2021 | Pinto, Trigo, Trigo, DaCamara — A practical method for high-resolution burned area monitoring using Sentinel-2 and VIIRS. *Remote Sens.* 13 | 10.3390/rs13091608 [verified, full text] | 39 | R | 2/3 | Coarse VIIRS burned area guides 10 m Sentinel-2 mapping (Google Earth Engine + DL); case studies in Portugal, France, Greece compared with Copernicus EMS maps; band importance. [full text, abstract] | Source of high-resolution perimeters for validation or extra supervision (notes §4.0, thesis §14). |

### 3.3 Fire events, ignition points and fire spread from satellite data (non-DL)

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Content | Use in the thesis |
|---|---|---|---|---|---|---|---|
| Andela2019 | Andela, Morton, Giglio, Paugam et al. — The Global Fire Atlas of individual fire size, duration, speed and direction. *ESSD* 11 | 10.5194/essd-11-529-2019 [verified] | 396 | O | **4/4** | Individual fires from MODIS burn dates; **ignition point = earliest burn date of the fire**; spread speed and direction per fire. [memory] | Closest concept to "nucleus birth" (notes §3.2); ignition points for a European extension; validation dataset for Portugal (compare GFA ignitions with ICNF). |
| Oom2016 | Oom, Silva, Bistinas, Pereira — Biome-specific sensitivity of fire size distributions to the time-gap parameter … fire event individuation. *Remote Sens.* 8 | 10.3390/rs8080663 [verified] | 38 | O | 4/4 | Fire-event individuation; results depend strongly on the time-gap parameter. [memory] | Literature support for the clustering sensitivity analysis (notes §3.1). |
| Balch2020 | Balch, St. Denis, Mahood, Mietkiewicz et al. — FIRED (Fire Events Delineation). *Remote Sens.* 12 | 10.3390/rs12213498 [verified] | 58 | O | 4/4 | Open space–time flood-fill algorithm for fire events from MCD64A1, with a sensitivity analysis of the spatial/temporal thresholds. [memory] | Same: how others choose D and gap; cite next to Oom2016. |
| Artes2019 (P09) | Artés, Oom, de Rigo, Durrant et al. — A global wildfire dataset for the analysis of fire regimes and fire behaviour (GlobFire). *Sci. Data* 6 | 10.1038/s41597-019-0312-2 [verified] | 230 | O | 3/3 | Global fire events from MCD64A1. [memory] | Event datasets; European extension (EFFIS link). |
| Laurent2018 (P08) | Laurent, Mouillot, Yue, Ciais et al. — FRY, a global database of fire patch functional traits. *Sci. Data* 5 | 10.1038/sdata.2018.132 [verified] | 103 | O | 3/3 | Fire patches and their traits from burned-area products, with a cut-off parameter. [memory] | Same group as above; not the origin of the nucleus method. |
| Chen2022 | Chen, Hantson, Andela, Coffield et al. — California wildfire spread derived using VIIRS and an object-based tracking system (FEDS). *Sci. Data* 9 | 10.1038/s41597-022-01343-0 [verified] | 82 | R | **4/4** | Fire objects tracked from **active fires** every 12 h (one VIIRS overpass), with merging/splitting of objects and fire-line extraction. [memory] | Closest active-fire analogue of the events/nuclei (notes §3) and of the 12 h periods; how merges are handled. |
| Farguell2021 | Farguell, Mandel, Haley, Mallia et al. — Machine learning estimation of fire arrival time from Level-2 active fires satellite data. *Remote Sens.* 13 | 10.3390/rs13112203 [verified] | 30 | R | 4/4 | SVM on fire / no-fire pixels in space–time → continuous **fire-arrival-time** surface, used to initialise a coupled fire model. [memory] | Literature basis for the "per-pixel burn time" target option (notes §4.0); ignition = minimum of the arrival-time field. |
| Rochoux2015 | Rochoux, Emery, Ricci, Cuenot et al. — Towards predictive data-driven simulations of wildfire spread, Part II: EnKF state estimation. *NHESS* 15 | 10.5194/nhess-15-1721-2015 [verified] | 56 | F | 2/2 | Data assimilation of observed fire fronts into a spread model. [memory] | Framing: ignition reconstruction as an inverse problem (discussion). |

### 3.4 Deep learning for wildfire data

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Content | Use in the thesis |
|---|---|---|---|---|---|---|---|
| Huot2022 (P18) | Huot, Hu, Goyal, Sankar et al. — Next Day Wildfire Spread: an ML dataset to predict wildfire spreading from remote-sensing data. *IEEE TGRS* 60 | 10.1109/TGRS.2022.3192974 [verified] | 155 | R | 3/4 | Next-day fire mask from today's mask + terrain, weather, vegetation, population; 64 km tiles at 1 km; baselines logistic regression, random forest, conv. autoencoder; PR-AUC. [memory] | Template for covariate channels (objective 4); forward counterpart of the thesis's inverse task. |
| Gerard2023 (P19) | Gerard, Zhao, Sullivan — WildfireSpreadTS: multi-modal time series for wildfire spread prediction. *NeurIPS 2023 Datasets & Benchmarks* | ID in draft wrong; NeurIPS proceedings [unverified] | — | R | 4/4 | Multi-day VIIRS active fire + weather/terrain time series (US); U-Net, ConvLSTM, U-TAE baselines; **AP as metric; year-wise cross-validation folds**. [memory] | Evaluation design: AP, year folds (leave-years-out, notes §6.4), multi-day lookback comparison. |
| Kondylatos2022 (P25a) | Kondylatos, Prapas, Ronco, Papoutsis et al. — Wildfire danger prediction and understanding with deep learning. *GRL* 49 | 10.1029/2022GL099368 [verified] | 162 | R | 3/4 | Daily fire danger in Greece from a datacube; LSTM/ConvLSTM vs. tree models; explained with attribution methods. [memory — check which XAI methods] | Precedent for XAI on spatiotemporal wildfire DL (objective 5). |
| Kondylatos2023 | Kondylatos, Prapas, Camps-Valls, Papoutsis — Mesogeos: a multi-purpose dataset for data-driven wildfire modeling in the Mediterranean. *NeurIPS 36* | 10.52202/075280-2204 [verified] | 4 | R | 3/**4** | Harmonised Mediterranean datacube (incl. Portugal; ~1 km, daily) with meteorology, vegetation, land cover, human variables, and ignitions/burned areas. [memory — check years and variables] | Possible shortcut for covariates and for the European extension. |
| McCarthy2021 | McCarthy, Tohidi, Aziz, Dennie et al. — A deep learning approach to downscale geostationary satellite imagery for decision support in high impact wildfires. *Forests* 12 | 10.3390/f12030294 [verified] | 21 | R | **4/4** | U-Net maps GOES fire observations to VIIRS-like resolution. [memory] | Direct precedent for learned sub-pixel localisation from geostationary data. |
| Kang2022 (P21b) | Kang, Jang, Im, Kwon — Deep learning model using geostationary satellite data for forest fire detection with reduced detection latency. *GIScience & RS* 59 | 10.1080/15481603.2022.2143872 [verified] | 104 | R | 3/3 | DL fire detection on geostationary (Himawari/GK-2A) time series. [memory] | Early-detection framing; geostationary DL precedent. |
| ZhaoBan2022 | Zhao, Ban — GOES-R time series for early detection of wildfires with deep GRU network. *Remote Sens.* 14 | 10.3390/rs14174347 [verified] | 28 | R | 3/3 | Per-pixel GRU on GOES-R ABI time series for early detection. [memory] | Same. |
| Burge2020 | Burge, Bonanni, Ihme, Hu — Convolutional LSTM neural networks for modeling wildland fire dynamics. arXiv:2012.06679 (preprint) | arXiv [verified] | — | O | 3/3 | ConvLSTM on simulated fire-spread sequences. [memory] | ConvLSTM-for-fire precedent (preprint, cite with care). |
| Jain2020 (P20a) | Jain, Coogan, Subramanian, Crowley et al. — A review of machine learning applications in wildfire science and management. *Environ. Rev.* 28 | 10.1139/er-2020-0019 [verified] | 802 | O | 3/2 | Broad review of ML in wildfire science. | Introduction; motivation sentence (until Xu et al. 2025 is found). |
| Ghali2023 (P20b) | Ghali, Akhloufi — Deep learning approaches for wildland fires remote sensing. *Remote Sens.* 15 | 10.3390/rs15071821 [verified] | 84 | R | 2/2 | Review: classification, detection, segmentation. | Introduction. |

### 3.5 Spatiotemporal architectures and training (already cited in notes §6)

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Use in the thesis |
|---|---|---|---|---|---|---|
| Shi2015 (P11) | Shi et al. — Convolutional LSTM network. *NeurIPS 2015* | arXiv:1506.04214 [verified] | — | F | 4/5 | Model 1 (notes §6.1). |
| Shi2017 | Shi et al. — Deep learning for precipitation nowcasting: a benchmark and a new model (TrajGRU). *NeurIPS 2017* | arXiv:1706.03458 [verified] | — | O | 3/4 | Learned, location-dependent recurrent connections; benchmark protocol and imbalance-aware loss weighting [memory]. |
| Ballas2016 (P13) | Ballas, Yao, Pal, Courville — ConvGRU. *ICLR 2016* | arXiv:1511.06432 [verified] | — | F | 3/5 | Slow memory, Option 4 (notes §6.6). |
| Che2018 (P14) | Che, Purushotham, Cho, Sontag, Liu — GRU-D. *Sci. Rep.* 8 | 10.1038/s41598-018-24271-9 [verified] | 1861 | O | 3/5 | Masks + time-since-observation + decay (Option 1, gate G2; cloud mask). |
| Dai2019 (P15) | Dai et al. — Transformer-XL. *ACL 2019* | 10.18653/v1/P19-1285 [verified] | 2006 | O | 2/4 | Cached, gradient-stopped memory (Option 4). |
| Wang2019 (P16) | Wang et al. — Eidetic 3D LSTM. *ICLR 2019* | OpenReview [unverified] | — | O | 3/4 | Recall attention over past states (Option 5). |
| Vaswani2017 (P17) | Vaswani et al. — Attention is all you need. *NeurIPS 2017* | arXiv:1706.03762 [verified] | — | F | 3/4 | Model 2. |
| Garnot2021 | Sainte Fare Garnot, Landrieu — Panoptic segmentation of satellite image time series with convolutional temporal attention networks (U-TAE). *ICCV 2021* | 10.1109/ICCV48922.2021.00483 [verified] | 176 | R | **4/5** | Template for Model 2: shared 2D encoder per date + light temporal self-attention producing attention masks that collapse time [memory]; used as a baseline in Gerard2023. |
| Campos2018, Koutnik2014, Neil2016 (P23) | Skip RNN (*ICLR 2018*), Clockwork RNN (*ICML 2014*), Phased LSTM (*NeurIPS 2016*) | arXiv:1708.06834, 1402.3511, 1610.09513 [verified] | — | O/F | 2/3 | Gate variants G3/G4 (notes §6.6). |
| Chen2023, Bengio2015 (P24) | Chen, Zhang, Hinton — Analog bits / self-conditioning (*ICLR 2023*); Bengio et al. — Scheduled sampling (*NeurIPS 2015*) | arXiv:2208.04202, 1506.03099 [verified] | — | R/F | 2/3 | Option 2 (notes §6.6). Note: notes §6.6 cites "Chen et al., 2022" — the conference version is 2023. |

### 3.6 Heatmap localisation, loss, evaluation and explainability

| Key | Reference | DOI / ID | Cites | Rec. | C/A | Use in the thesis |
|---|---|---|---|---|---|---|
| Zhou2019 (P12) | Zhou, Wang, Krähenbühl — Objects as points (CenterNet). arXiv preprint | arXiv:1904.07850 [verified] | — | O | 4/5 | Heatmap formulation, peak extraction, **offset head** (§6). |
| Law2018 | Law, Deng — CornerNet: detecting objects as paired keypoints. *ECCV 2018* | 10.1007/978-3-030-01264-9_45 [verified] | 2704 | O | 3/5 | Origin of the penalty-reduced focal loss used in `src/model.py`. |
| Lin2017 | Lin, Goyal, Girshick, He, Dollár — Focal loss for dense object detection. *ICCV 2017* | 10.1109/ICCV.2017.324 [verified] | 25889 | O | 3/4 | Focal loss; imbalance (notes §6.1.1). |
| Zhang2020 | Zhang, Zhu, Dai, Ye, Zhu — Distribution-aware coordinate representation for human pose estimation (DARK). *CVPR 2020* | 10.1109/CVPR42600.2020.00712 [verified] | 495 | O | 3/**5** | Sub-pixel decoding of Gaussian heatmaps (Taylor expansion of the log-heatmap at the peak) [memory] — post-processing only. |
| Song2021 | Song, Wang, Jiang, Wang et al. — Rethinking counting and localization in crowds: a purely point-based framework (P2PNet). *ICCV 2021* | 10.1109/ICCV48922.2021.00335 [verified] | 362 | R | 3/3 | Point-set prediction with one-to-one matching for dense, nearby points [memory] — alternative for merged ignitions. |
| Guo2017 | Guo, Pleiss, Sun, Weinberger — On calibration of modern neural networks. *ICML 2017* | arXiv:1706.04599 [verified] | — | O | 2/4 | Calibration check / temperature scaling (notes §6.1.1: "probabilities peak at ~0.55, calibration not checked"). |
| Roberts2017 | Roberts, Bahn, Ciuti, Boyce et al. — Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography* 40 | 10.1111/ecog.02881 [verified] | 3474 | O | 2/4 | Justifies the chronological split and blocked (leave-years-out) CV (notes §4.2, §6.4). |
| Bouthillier2021 | Bouthillier et al. — Accounting for variance in machine learning benchmarks. *MLSys 2021* | arXiv:2103.03098 [verified] | — | R | 2/4 | Justifies ≥ 3 seeds and reporting mean ± std (notes §6.7). |
| Sundararajan2017 (P25b) | Sundararajan, Taly, Yan — Axiomatic attribution for deep networks (Integrated Gradients). *ICML 2017* | arXiv:1703.01365 [verified] | — | O | 2/4 | XAI (objective 5). |
| Adebayo2018 | Adebayo et al. — Sanity checks for saliency maps. *NeurIPS 2018* | arXiv:1810.03292 [verified] | — | O | 2/4 | Model-randomisation test before trusting attribution maps. |

## 4. Research landscape and gaps

### 4.1 Four strands of work
1. **Active-fire detection and FRP from geostationary sensors** (§3.1). Mature for detection and FRP; early-detection work (Koltunov2016, Filizzola2016, DiBiase2018, Kang2022, ZhaoBan2022) is about *when* a fire is first seen, not *where exactly* it started. Sub-pixel localisation from geostationary data is rare (Wickramasinghe2016, McCarthy2021).
2. **Rule-based fire events and ignition points** (§3.3, Benali2016). Almost all large datasets (Global Fire Atlas, FIRED, GlobFire, FRY, Oom2016) are built from *burned-area* products at daily resolution; the ignition is the earliest-burned pixel. All report a strong dependence on the spatial/temporal thresholds — the same merging-vs-fragmentation trade-off as notes §3.1. Active-fire-based tracking (Chen2022, FEDS) works at 12 h steps from VIIRS.
3. **Deep learning on fire time series** (§3.4): mostly forward problems — next-day spread (Huot2022, Gerard2023) or danger (Kondylatos2022, Mesogeos) — and detection. Pinto2020 is the exception that comes closest: it learns *when* each pixel burned from a sequence (dating), i.e. an inverse, per-pixel timing problem.
4. **Generic methods** (§3.5–3.6): ConvLSTM/ConvGRU, temporal attention for satellite time series (U-TAE), heatmap keypoints (CornerNet/CenterNet/DARK), memory mechanisms.

### 4.2 Where this thesis sits (inferred, to confirm with a final database search)
- In the verified set, no study **learns ignition locations from geostationary FRP sequences**. The closest are Benali2016 (rule-based, polar-orbiting, Portugal, same ground truth), Andela2019 (rule-based, burned area, global) and Pinto2020 (DL, timing of burning rather than ignition location, VIIRS).
- The thesis's distinct elements: (a) 15-min geostationary input on the native grid (notes §4.2); (b) multi-ignition heatmap targets instead of one fire per sample, so hotspot ↔ fire assignment is learned (notes §4.0); (c) evaluation against official ignition records with isolated vs. non-isolated stratification (notes §5.2); (d) the reappearance-vs-rekindle analysis (notes §5.3, §6.6).
- This is an **inferred** gap, not one stated by the cited authors. Before writing "first" or "novel" in the thesis: a Scopus/Web of Science search for 2020–2026 on ("ignition" AND ("geostationary" OR "SEVIRI" OR "GOES" OR "Himawari") AND ("deep learning" OR "neural network")), plus forward citations of Benali2016 and Andela2019.

### 4.3 Open questions the literature does not answer yet
- **Rekindles in ICNF.** The hypothesis "ICNF records rekindles as new fires" (notes §5.3) has no verified source yet. To check: whether the ICNF/SGIF fire records contain a rekindle flag ("reacendimento") or fire type field [memory — not verified]; if so, it would turn the reappearance-vs-rekindle question into a measurable subset.
- **ICNF ignition-point accuracy.** Benali2016 and Catry2009 use ICNF points; whether either quantifies their positional error must be checked in the full text.
- **Cloud mask.** Needed for any memory option (notes §6.6); the relevant source is the EUMETSAT/NWC SAF product documentation, not yet collected.
- **MTG-FCI.** For the "operational monitoring" outlook, the successor of SEVIRI (Meteosat Third Generation, finer pixels and faster repeat [memory]) and its fire product need a reference.

---

# Part II — Application to the thesis (draft, 2026-10-20)

## 5. Already implemented: literature that supports, or should be cited for, existing work

| Thesis component | Where | Literature | Comment |
|---|---|---|---|
| SEVIRI data, FRP, confidence, saturation | thesis §6.1, notes §1.2 | Wooster2015, Roberts2015, Roberts2008, Wooster2003 | Data chapter. Check the 1,050 MW saturation and the confidence definition in Wooster2015. |
| Native geostationary grid (`src/grid.py`) | notes §4.2 | Wooster2015 (product geometry) | The reconstruction itself is the thesis's own result; the paper only documents the projection. |
| Confidence vs. FRP trade-off (no confidence filter) | thesis §6.1 | Wooster2015, Freeborn2014, Hall2019 | Low-FRP early detections matter most for localisation — the literature on omission of low-FRP fires supports keeping them. |
| Detection probability by fire size (fig. 07), FRE vs. burned area (fig. 08) | notes §2.2 | Freeborn2014, Hall2019, Roberts2015, Wooster2005 | Comparison points, not confirmations. |
| Seasonal/diurnal cycle (fig. 04) | notes §2.2 | Amraoui2013, Roberts2015 | Same product, same region. |
| ICNF data and cleaning | notes §1.1, §1.3 | Pereira2011, Catry2009, Benali2016 | The anomalies (coordinate sharing, 1994 end dates) are the thesis's own audit. |
| Time zone check, detection lag | notes §1.4 | Koltunov2016, Hall2019 (latency); Benali2016 (dates from satellite) | No paper is needed for the DST test itself. |
| Space–time clustering (6 km, 6 h) + sensitivity grid | notes §3.1, `src/cluster_hotspots.py` | Oom2016, Balch2020, Artes2019, Laurent2018, Chen2022 | Strong support: all report threshold sensitivity. Chen2022 is the active-fire analogue. |
| Nuclei and nucleus-birth baseline | notes §3.2, §5.3 | Andela2019 (ignition = earliest burn), Benali2016, Chen2022 (merge/split of fire objects) | The baseline = the standard rule-based approach in the literature, applied to SEVIRI — a fair baseline. |
| Multi-ignition Gaussian heatmaps, focal loss, ignore regions | notes §4.2, §6.1 | Zhou2019, **Law2018**, Lin2017 | Add CornerNet and Lin2017 to notes §6.1. |
| 3 × 3 probability-weighted sub-pixel centroid | notes §6.1 | Zhou2019 (offset head), Zhang2020 (DARK) | Both are better-founded alternatives (§6). |
| ConvLSTM encoder–recurrence–head | notes §6.1 | Shi2015, Shi2017, Burge2020, Pinto2020 | Pinto2020: a CNN + LSTM on satellite time series with an ablation without the LSTM — same question as `cnn3d` (notes §6.7). |
| Chronological split, test locked | notes §4.2, §6.4 | Roberts2017, Gerard2023 (year folds) | |
| ≥ 3 seeds per ablation, mean ± std | notes §6.7 | Bouthillier2021 | |
| Long-term memory options 1–5 (concept) | notes §6.6 | Ballas2016, Che2018, Dai2019, Wang2019, Campos2018, Koutnik2014, Neil2016, Chen2023, Bengio2015 | Already cited; correct the Chen year. |

## 6. Could be implemented (ordered by expected value / cost)

### 6.1 Cheap — post-processing or analysis, no retraining
1. **DARK sub-pixel decoding** (Zhang2020) as an extra option in `src/peak_ablation.py`: replace the 3 × 3 weighted centroid by the second-order (Taylor) estimate of the peak of the log-heatmap. Targets AP @ 5 km, the weakest metric (notes §5.3, §6.5). The targets are Gaussian (σ = 1 px), which is the assumption of DARK.
2. **Calibration check** (Guo2017): reliability diagram and expected calibration error of peak probabilities vs. hit rate on validation; temperature scaling fitted on validation. Answers the open "probabilities peak at ~0.55" limitation (notes §6.1.1). Relevant for operational use (a probability should mean something).
3. **External benchmark against Benali2016**: report the baseline and the model with the same quantities as Benali2016 (distance and time error to ICNF for large fires), so the thesis can be placed against the closest Portuguese study. Requires reading their exact definitions.
4. **Global Fire Atlas ignitions vs. ICNF** (Andela2019): for 2004–2016 large fires, distance between GFA ignition, ICNF point and SEVIRI first detection/nucleus birth — an independent estimate of how noisy the ICNF points are, i.e. the floor for the achievable error.

### 6.2 Moderate — small training changes, fit into the ablation framework (notes §6.7)
5. **Detection-erasing augmentation** (Pinto2020): during training, randomly delete detections in space–time blocks (and occasionally whole 15-min slots) of the input — simulates cloud/smoke omission, the Pedrógão case (notes §2.1) and the "no fire ≠ not visible" ambiguity, without a cloud mask. New ablation `erase` (one flag in `src/train.py`/`src/dataset.py`). Expected effect: robustness at the cost of a little clean-sky accuracy.
6. **CenterNet offset head** (Zhou2019; draft G.4): two extra output channels with the sub-pixel offset of each target, L1 loss at target pixels. Compare with item 1 — if DARK already gives the gain, the head is unnecessary.
7. **Auxiliary timing output** (Pinto2020 dating; Farguell2021 arrival time; notes §4.0 and §4.2 "optional second output with the ignition time"): a second head that predicts, per pixel, the time of first detection within the window (self-supervised from SEVIRI — no extra labels) or the ignition time of the targets. Gives the model a dense, per-step training signal (counters the single-readout bottleneck, notes §6.1.1) and a new output (ignition time) for evaluation.
8. **Fire Weather Index as the first meteorological covariate** (DaCamara2014, Pinto2018): one channel that the Lisbon SEVIRI-FRP literature has already calibrated over Mediterranean Europe, instead of 5–6 raw ERA5-Land variables. Then wind direction/speed separately, because the inverse task needs the *direction* of spread (Huot2022 uses wind as a forward driver).
9. **VIIRS 375 m detections (2012–2019) as finer reference** (Schroeder2014): estimate how far the SEVIRI-based prediction and the ICNF point are from the first VIIRS detection of the same fire; optionally as an auxiliary, finer training target. Requires FIRMS data.

### 6.3 Larger — new model or data
10. **Model 2 as a U-TAE-style network** (Garnot2021; used for fires in Gerard2023): shared 2D encoder per step (the existing encoder), lightweight temporal self-attention per pixel at low resolution producing attention masks, then a decoder to the heatmap. This is a concrete, cheap form of the planned "CNN + Transformer" (thesis §3), removes the single-readout bottleneck by design, and its attention masks are an XAI output ("which 15-min steps mattered").
11. **Mesogeos datacube** (Kondylatos2023) for covariates and the European extension: check whether its variables, grid and years make it usable instead of building the ERA5-Land/DEM/CORINE stack from scratch.
12. **Learned downscaling** (McCarthy2021): longer-term idea — learn SEVIRI → VIIRS-resolution fire maps on 2012–2019, then localise ignitions on the finer map. Probably beyond the thesis scope; useful for the discussion/outlook.
13. **Point-set prediction for dense outbreaks** (Song2021): predict a set of points with one-to-one matching (as the evaluation already does) instead of heatmap peaks; an alternative if `sigma05` and the peak ablation do not fix merged peaks (18–19 points for 31 ignitions, notes §6.3).

## 7. Concepts and methodology inspirations

- **Inverse vs. forward framing.** The fire-spread DL literature (Huot2022, Gerard2023) predicts where a fire *will* burn; the thesis asks where it *started*. The same covariates enter both, but their role flips (wind and slope tell where the fire came from). Data assimilation (Rochoux2015) gives the formal language: ignition reconstruction as state estimation from incomplete observations. Good framing for the introduction and discussion.
- **Ignition = earliest time of an arrival-time field.** Andela2019 (burn date), Farguell2021 (arrival-time surface) and Pinto2020 (dating) all represent a fire as a per-pixel time of burning, from which the ignition follows as the earliest point. This supports the alternative target noted in notes §4.0 and suggests a hybrid: predict an arrival-time map, read ignitions as its local minima, and compare with the heatmap model.
- **Two framings, two literatures** (notes §6.4): *early detection* (Koltunov2016, Filizzola2016, Kang2022, ZhaoBan2022: latency, false alarms per day) vs. *post-fire reconstruction* (Benali2016, Andela2019, Chen2022: location/time error). Keeping them apart in the thesis lets each result be compared with the right literature and metric.
- **Threshold sensitivity is a known, general problem.** Oom2016, Balch2020 and Artés2019 show that event definitions depend on the time gap and distance. This turns the thesis's argument for raw input instead of clustering (notes §4.0) from an own observation into a recognised limitation of the rule-based paradigm — with the `births_only`/`births_plus` ablations as the direct test.
- **Augmentation as a model of the sensor.** Pinto2020 designs augmentations from the error modes of the input (clouds, omitted active fires). The same logic applies here: omission (erasing), FRP noise, position jitter within a pixel — each tied to a documented SEVIRI limitation (§3.1).
- **Evaluation practice.** AP with year-wise folds (Gerard2023), blocked CV (Roberts2017), seed variance (Bouthillier2021), calibration (Guo2017) and attribution sanity checks (Adebayo2018) together form a defensible evaluation chapter; the thesis already follows most of it.
- **Possible research context.** Pinto2018/2020/2021, DaCamara2014 and Amraoui2010/2013 come from the IDL Lisbon / IPMA group that works with the same LSA SAF SEVIRI FRP product over Portugal; the supervisor's extract may come from this environment. Worth asking — it would give access to the product documentation, a cloud mask and possibly SEVIRI 2020–2023 (thesis §9 item 8).

## 8. Reading order (2026-10-20)

Ordered by value for the current stage of the thesis (models running, ablations next, write-up of data and methods chapters).

| # | Paper | Why read it now |
|---|---|---|
| 1 | Benali2016 | Same country, same ICNF ground truth, same task solved by rules. The external benchmark (§6.1 item 3) and the main point of comparison in the discussion. |
| 2 | Pinto2020 (in `documents/papers/`) | Closest DL precedent, probably from the supervisor's environment: per-pixel timing of burning from image sequences, CNN + LSTM ablation, active-fire erasing augmentation (§6.2 items 5, 7). |
| 3 | Wooster2015 (Part 1) | The manual for the data (FRP, confidence, saturation, detection limits); needed for the data chapter anyway. |
| 4 | Andela2019 | Ignition = earliest burned pixel — the burned-area version of the nucleus births; also a way to estimate ICNF point noise (§6.1 item 4). |
| 5 | Zhou2019 + Law2018 | The loss and peak extraction in use; the offset head targets AP @ 5 km, the weakest metric (§6.2 item 6). |
| 6 | Oom2016 + Balch2020 | Event definitions depend on distance/time thresholds — turns the argument for raw input over clustering (notes §4.0) into a recognised limitation of the rule-based paradigm. |
| 7 | Garnot2021 (U-TAE) | Most concrete and cheap template for Model 2 (CNN + temporal attention) (§6.3 item 10). |
| 8 | Chen2022 (FEDS) | Active-fire tracking in 12 h steps with merges/splits — closest active-fire analogue of the events/nuclei. |
| 9 | Farguell2021 | Per-pixel fire-arrival time: basis for the alternative "burn time" target (§7). |
| 10 | Wooster2021 (review) | Skim for the background chapter. |

After that: DaCamara2014 and Pinto2018 (case for FWI as the first meteorological covariate, §6.2 item 8). The memory papers (Ballas2016, Che2018, Dai2019) only when the long-term memory work starts (notes §6.6). Turco2019 and Tedim2018: one context sentence each, no deep read.

## 9. Next steps for the literature

1. Read the full texts of the top-priority papers (order in §8) and replace every [memory] tag used in the thesis by a checked statement.
2. Close the open items of §2 and §4.3 (rekindle source, ICNF accuracy, cloud mask, MTG-FCI, 2024–2026 papers, Xu et al. 2025).
3. Keep this file as the single reference list: new papers go into §3 with tags, their use into §5–7; export to BibTeX when writing starts.
