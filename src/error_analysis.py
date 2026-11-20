"""Ceilings and error analysis of the space-time model cnn3d_st (notes.md 9.21).

What limits cnn3d_st, and how far the task can go at SEVIRI resolution. Uses only the data, the labels
and the saved points of finished runs (outputs/compare_runs/, written by compare_runs.py); no training.
  val    validation 2015-2016 with the runs trained WITHOUT 2015-2016 (ablations_all/cnn3d_st_all_s0-2,
         ablations_all/cnn3d_final_all_s0-2). All conclusions and improvement choices come from here.
  test   2024-2025 with final_2023/cnn3d_st_s0-2, final_2023/cnn3d_final_s0-2 and final/ (trained to 2016):
         descriptive only, after the single test of 9.20 (needs --confirm-test), as in 6.18. Nothing is
         chosen here.
Parts (operating point = best F1 at 10 km of each run, as in 9.12):
  A  localisation: pixel quantisation, oracle nearest detected cell, first-slot centroid and the model's
     hits; per-axis spread of ICNF - detected cell for fresh, isolated, one-cell first detections against
     the spread of pure quantisation (an upper bound of the ICNF position error); AP and recall vs the
     matching radius k, with the oracle as the bound of any detected-cell estimate
  B  data: share of the fires >= 50 ha that SEVIRI sees, recall against ALL fires >= 50 ha (seen + unseen,
     nearest point <= 10 km in the fire's period), share of the burned area of the fires found
  C  target tensors (labels only): share of targets with their own peak in the 2D and in the 3D target
  D  operating point and AP: recall with all points vs at best F1; false alarms by the nearest ICNF
     evidence (duplicate of a found target, rekindle record, fire < 10 ha, fire >= 10 ha 1-72 h before,
     fire >= 10 ha starting in the 6 h tail, border cell, persistent cell, nothing); AP with these
     classes counted as neutral, cumulatively (ceilings set by the label definition)
  E  misses at best F1: no point within k / point taken by another target / matched below the threshold;
     recall by subgroup and by the SEVIRI signal at the target
  F  ranking: SEVIRI signal at every point (FRE, detections, activity in the 6 h history) for hits and
     false alarms; how well that signal alone ranks them (AUC) against the model score; score of the hits
     vs the hours of observation left after the ICNF start

Run from the repository root:
  .venv/bin/python src/error_analysis.py --split val > outputs/error_analysis/val.txt
  .venv/bin/python src/error_analysis.py --split test --confirm-test > outputs/error_analysis/test.txt
Outputs: outputs/error_analysis/<split>_{targets,points,radius,ap_ceiling,fp_classes}.csv (+ the text above)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import eda_figures as E
import grid as G
from baselines import predictions as baseline_predictions
from dataset import FireData
from evaluate import average_precision, evaluate, target_context
from tiling_design import ACQ_OFFSET_H, HISTORY_H, PERIOD_H, TAIL_H
from train import decode, heatmap_to_points

ROOT = E.ROOT
CACHE = ROOT / "outputs" / "compare_runs"
OUT = ROOT / "outputs" / "error_analysis"
SLOT_NS = 15 * 60 * 10**9
R_PX = 1.5                      # cells: detections this close belong to a point / target (as R_TIME_PX)
RADII = (1, 1.5, 2, 3, 4, 5, 7.5, 10, 15)
RUNS = {
    "val": {"cnn3d_st": [f"ablations_all/cnn3d_st_all_s{s}" for s in range(3)],
            "cnn3d_final": [f"ablations_all/cnn3d_final_all_s{s}" for s in range(3)]},
    "test": {"cnn3d_st": [f"final_2023/cnn3d_st_s{s}" for s in range(3)],
             "cnn3d_final": [f"final_2023/cnn3d_final_s{s}" for s in range(3)],
             "cnn3d_final_2016": [f"final/cnn3d_final_s{s}" for s in range(3)],
             "conv_final_2016": [f"final/conv_final_s{s}" for s in range(3)]},
}
FP_CLASSES = ("duplicate", "rekindle record", "earlier fire >= 10 ha", "fire < 10 ha", "fire in the tail",
              "border cell", "persistent cell", "no ICNF evidence")
LABEL_DEF = ("rekindle record", "earlier fire >= 10 ha", "fire < 10 ha", "fire in the tail")
SMALL = []                      # ICNF records < 10 ha of the split (set in main)
PREV_D = 7                      # days of satellite history looked at before the input window (not seen by the model)


# ---------------------------------------------------------------- inputs

def cached_points(name, split):
    """Points of a finished run as written by compare_runs.points() (asserts they are newer than the checkpoint)."""
    f = CACHE / f"{name.replace('/', '__')}{'' if split == 'val' else '_' + split}.parquet"
    ck = ROOT / "models" / f"{name}.pt"
    assert f.exists() and f.stat().st_mtime > ck.stat().st_mtime, f"run compare_runs.py first for {name} ({split})"
    return pd.read_parquet(f)


def best_f1_threshold(curve):
    f1 = 2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]).clip(lower=1e-9)
    return float(curve["score"].iloc[int(f1.values.argmax())])


def slot_of(t):
    return pd.to_datetime(t, utc=True).astype("int64") // SLOT_NS


def window_index(data, period):
    """Index range [a, b) of the detections of a period's 24 h input window (slot-sorted table)."""
    s0 = data.first_slot(period)
    return np.searchsorted(data._slot, [s0, s0 + (HISTORY_H + PERIOD_H + TAIL_H) * 4])


def cell_km(i, j):
    """Size of one native cell [km] east-west and north-south at image position (i, j)."""
    la, lo = G.ij_to_latlon(i, j)
    la_e, lo_e = G.ij_to_latlon(i, j + 1)
    la_n, lo_n = G.ij_to_latlon(i - 1, j)
    return E.haversine_km(la, lo, la_e, lo_e), E.haversine_km(la, lo, la_n, lo_n)


def km_ij(i1, j1, i2, j2):
    la1, lo1 = G.ij_to_latlon(i1, j1)
    la2, lo2 = G.ij_to_latlon(i2, j2)
    return E.haversine_km(la1, lo1, la2, lo2)


# ---------------------------------------------------------------- targets and their signal

def target_table(data, split):
    """One row per target of the split(s): context, size and the SEVIRI signal near its (sensor-frame) position."""
    splits = [split] if isinstance(split, str) else list(split)
    per = set(data.periods.loc[data.periods["split"].isin(splits), "period"])
    tg = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(per)][["period", "fire_id", "i_f", "j_f"]].copy()
    f = data.fires.set_index("fire_id")
    tg = tg.join(f[["start_utc", "lat", "lon", "burned_area_ha", "size_class", "isolated", "rekindle"]], on="fire_id")
    tg = tg.join(target_context(data), on="fire_id")
    nu = pd.read_parquet(E.PROC / "seviri_nuclei.parquet").set_index("nucleus_id")["t_birth"]
    frp = data.det["frp_mw"].values.astype(float)
    nuc = data.det["nucleus_id"].values
    rows = []
    for q in tg.itertuples():
        s = slot_of(pd.Series([q.start_utc])).iat[0]
        a, b = window_index(data, q.period)
        sl, ii, jj = data._slot[a:b], data._i[a:b], data._j[a:b]
        d = np.hypot(ii - q.i_f, jj - q.j_f)
        near = d <= R_PX
        r = {}
        # localisation (model_review 3.5): detections <= 2 cells in [start - 1 h, start + 6 h)
        loc = (d <= 2.0) & (sl >= s - 4) & (sl < s + 24)
        if loc.any():
            k = np.nonzero(loc)[0]
            r["oracle_km"] = km_ij(q.i_f, q.j_f, ii[k], jj[k]).min()
            first = k[sl[k] == sl[k].min()]
            w = frp[a:b][first]
            ci, cj = (ii[first] * w).sum() / w.sum(), (jj[first] * w).sum() / w.sum()
            r["first_slot_km"] = float(km_ij(q.i_f, q.j_f, ci, cj))
            r["first_slot_cells"] = len(first)
            if len(first) == 1:                                   # one detected cell in the first slot
                r["d_east_km"] = float(km_ij(q.i_f, q.j_f, q.i_f, jj[first][0]) * np.sign(q.j_f - jj[first][0]))
                r["d_south_km"] = float(km_ij(q.i_f, q.j_f, ii[first][0], q.j_f) * np.sign(q.i_f - ii[first][0]))
        ew, ns = cell_km(q.i_f, q.j_f)
        r["cell_ew_km"], r["cell_ns_km"] = float(ew), float(ns)
        r["quant_km"] = float(km_ij(q.i_f, q.j_f, np.round(q.i_f), np.round(q.j_f)))
        # signal at the target in the window (<= 1.5 cells)
        after = near & (sl >= s - 4)
        r["first_lag_h"] = ((sl[after].min() - s) / 4 + ACQ_OFFSET_H) if after.any() else np.nan
        r["n_det_6h"] = int((near & (sl >= s) & (sl < s + 24)).sum())
        r["fre_gj"] = float(frp[a:b][after].sum() * 900 / 1e3)
        r["max_frp"] = float(frp[a:b][after].max()) if after.any() else 0.0
        r["obs_h"] = float(PERIOD_H + TAIL_H - q.t_h)               # hours of input left after the ICNF start
        s0 = data.first_slot(q.period)
        a0 = np.searchsorted(data._slot, s0 - PREV_D * 96)
        dp = np.hypot(data._i[a0:a] - q.i_f, data._j[a0:a] - q.j_f) <= R_PX
        r["prev_h"] = float((s0 - data._slot[a0:a][dp].max()) / 4) if dp.any() else np.nan   # before the window, not seen by the model
        # 'seen' only through an older nucleus (data_pipeline.md 6.2): every seen-test detection (<= 5 km,
        # [start - 1 h, start + 6 h)) belongs to a nucleus born more than 1 h before the start
        la, lo = G.true_to_sensor_latlon(q.lat, q.lon, q.start_utc)
        sw = (sl >= s - 4) & (sl < s + 24)
        k = np.nonzero(sw)[0]
        dk = E.haversine_km(la, lo, data.det["lat"].values[a:b][k], data.det["lon"].values[a:b][k])
        k = k[dk <= 5]
        births = nu.reindex(nuc[a:b][k]).values
        r["seen_via_older"] = bool(len(k)) and bool(np.all(births < (q.start_utc - pd.Timedelta(hours=1)).tz_convert(None).to_datetime64()))
        rows.append(r)
    return pd.concat([tg.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------- points and their signal

def point_signal(pts, data):
    """SEVIRI signal at every point (sensor frame), in its period's 24 h window, <= R_PX cells."""
    pts = pts.copy()
    i, j = G.latlon_to_ij(pts["lat"].values, pts["lon"].values)
    if data.georef_fix:
        i, j = G.true_to_sensor_ij(i, j, pts["period"])
    pts["si"], pts["sj"] = i, j
    frp = data.det["frp_mw"].values.astype(float)
    out = {c: np.zeros(len(pts)) for c in ("p_fre_gj", "p_ndet", "p_hist", "p_first_h", "p_hours", "p_prev_h")}
    for p, idx in pts.groupby("period").indices.items():
        a, b = window_index(data, p)
        s0 = data.first_slot(p)
        a0 = np.searchsorted(data._slot, s0 - PREV_D * 96)          # up to PREV_D days before the window
        pi, pj, ps = data._i[a0:a], data._j[a0:a], data._slot[a0:a]
        dprev = np.hypot(pi[None] - i[idx, None], pj[None] - j[idx, None]) <= R_PX
        last = np.where(dprev, ps[None], -1).max(1) if len(ps) else np.full(len(idx), -1)
        out["p_prev_h"][idx] = np.where(last >= 0, (s0 - last) / 4, np.nan)   # hours from the last earlier detection to the window start
        sl, ii, jj, fr = data._slot[a:b] - s0, data._i[a:b], data._j[a:b], frp[a:b]
        d = np.hypot(ii[None] - i[idx, None], jj[None] - j[idx, None]) <= R_PX      # (points, detections)
        out["p_fre_gj"][idx] = (d * fr[None]).sum(1) * 900 / 1e3
        out["p_ndet"][idx] = d.sum(1)
        out["p_hist"][idx] = (d & (sl[None] < HISTORY_H * 4)).sum(1)
        first = np.where(d, sl[None], 10**6).min(1)
        out["p_first_h"][idx] = np.where(first < 10**6, first / 4 - HISTORY_H, np.nan)   # hours since the period start
        out["p_hours"][idx] = [len(np.unique(sl[row] // 4)) for row in d]
    for c, v in out.items():
        pts[c] = v
    return pts


def auc(pos, neg):
    """Area under the ROC curve of a score separating pos from neg (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


# ---------------------------------------------------------------- false-alarm classes

def icnf_small(data, split):
    """ICNF records < 10 ha with valid coordinates and UTC start, in the split's years (+- 1 day)."""
    ic = pd.read_parquet(E.PROC / "icnf_fires.parquet", columns=["fire_id", "start_utc", "lat", "lon", "burned_area_ha"])
    per = data.periods.loc[data.periods["split"] == split, "period"]
    lo, hi = per.min() - pd.Timedelta(days=1), per.max() + pd.Timedelta(days=2)
    ic = ic[ic["lat"].notna() & ic["start_utc"].notna() & (ic["burned_area_ha"] < 10) & ic["start_utc"].between(lo, hi)]
    return ic


def _between(df, col, lo, hi):
    """Rows of a table sorted by `col` with lo <= col < hi (searchsorted)."""
    v = df[col].values
    a, b = np.searchsorted(v, [lo.to_datetime64(), hi.to_datetime64()])
    return df.iloc[a:b]


def fp_classes(scored, data, k, small):
    """Class of every false alarm (FP_CLASSES, first that applies) and, for duplicates, the hours between
    the false alarm and the nearest hit of its period (space-time points only)."""
    h = lambda x: pd.Timedelta(hours=x)
    fp = scored[scored["status"] == "fp"]
    tp = scored[scored["status"] == "tp"]
    lab = data.labels[data.labels["role"] == "target"].merge(data.fires[["fire_id", "lat", "lon"]], on="fire_id")
    lab_by = {p: g for p, g in lab.groupby("period")}
    tp_by = {p: g for p, g in tp.groupby("period")}
    rk_by = {p: g for p, g in data.rekindles.groupby("period")}
    big = data.fires[["start_utc", "lat", "lon"]].sort_values("start_utc")
    small = small.sort_values("start_utc")
    i, j = G.latlon_to_ij(fp["lat"].values, fp["lon"].values)
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    border = pd.Series(~data.static["pt"][i, j], index=fp.index)
    persistent = pd.Series(data.static["persistent"][i, j], index=fp.index)
    timed = "t_h" in fp
    cls, ddt = {}, {}
    for p, g in fp.groupby("period"):
        sets = {"duplicate": lab_by.get(p), "rekindle record": rk_by.get(p),
                "earlier fire >= 10 ha": _between(big, "start_utc", p - h(72), p - h(1)),
                "fire < 10 ha": _between(small, "start_utc", p - h(1), p + h(PERIOD_H + 1)),
                "fire in the tail": _between(big, "start_utc", p + h(PERIOD_H + 1), p + h(PERIOD_H + TAIL_H))}
        T = tp_by.get(p)
        for q in g.itertuples():
            c = None
            for name, S in sets.items():
                if S is not None and len(S) and E.haversine_km(q.lat, q.lon, S["lat"].values, S["lon"].values).min() <= k:
                    c = name
                    break
            if c is None:
                c = "border cell" if border[q.Index] else "persistent cell" if persistent[q.Index] else "no ICNF evidence"
            dt = np.nan
            if c == "duplicate" and timed and T is not None and len(T):
                dd = E.haversine_km(q.lat, q.lon, T["lat"].values, T["lon"].values)
                dt = abs(q.t_h - T["t_h"].values[int(dd.argmin())])
            cls[q.Index], ddt[q.Index] = c, dt
    out = scored.copy()
    out["fp_class"] = pd.Series(cls)
    out["dup_dt_h"] = pd.Series(ddt)
    return out


def ap_with_neutral(scored, n_targets, classes):
    s = scored.copy()
    s.loc[s["fp_class"].isin(classes), "status"] = "ignored"
    return average_precision(s, n_targets)[0]


# ---------------------------------------------------------------- parts

def part_a(tg, runs_scored, data, split, base):
    print("\nA  Localisation ceiling (targets of the split; detections <= 2 cells in [start - 1 h, start + 6 h))")
    f = tg["oracle_km"].notna()
    print(f"   targets {len(tg)}, with a detection for the oracle {f.sum()}")
    q = lambda v: f"median {np.nanmedian(v):.2f} km, p25 {np.nanpercentile(v, 25):.2f}, p75 {np.nanpercentile(v, 75):.2f}"
    print(f"   pixel quantisation (ICNF point -> centre of its cell): {q(tg['quant_km'])}; cell {tg['cell_ew_km'].median():.2f} km E-W x {tg['cell_ns_km'].median():.2f} km N-S")
    print(f"   oracle: nearest detected cell centre:                 {q(tg['oracle_km'])}")
    print(f"   FRP-weighted centroid of the first detection slot:    {q(tg['first_slot_km'])}; one cell in the first slot for {np.mean(tg.loc[f, 'first_slot_cells'] == 1):.0%}")
    for name, sc in runs_scored.items():
        for k in (5, 10):
            e = np.concatenate([s[k][2].loc[s[k][2]["status"] == "tp", "err_km"].values for s in sc])
            print(f"   {name:18s} hits @ {k:2d} km (3 seeds pooled): {q(e)}")
    e = base[5][2].loc[base[5][2]["status"] == "tp", "err_km"]
    print(f"   {'nucleus birth':18s} hits @  5 km: {q(e)}")
    # AP and recall vs radius
    rows = []
    for name, sc in list(runs_scored.items()) + [("nucleus birth", None)]:
        for k in RADII:
            if sc is None:
                res = evaluate(base["points"], data, split, k, groups=())[0]
                aps, recs = [res["AP"]], [res["recall (all preds)"]]
            else:
                rs = [evaluate(p, data, split, k, groups=())[0] for p in sc[0]["points_list"]]
                aps, recs = [r["AP"] for r in rs], [r["recall (all preds)"] for r in rs]
            rows.append(dict(method=name, k=k, AP=np.mean(aps), recall=np.mean(recs)))
    ok = tg["oracle_km"].values
    for k in RADII:
        rows.append(dict(method="oracle: nearest detected cell", k=k, AP=np.nanmean(np.nan_to_num(ok, nan=np.inf) <= k), recall=np.mean(np.nan_to_num(ok, nan=np.inf) <= k)))
        rows.append(dict(method="pixel quantisation (bound)", k=k, AP=np.mean(tg["quant_km"] <= k), recall=np.mean(tg["quant_km"] <= k)))
    rad = pd.DataFrame(rows)
    print("   AP (recall with all points) by matching radius k [km]:")
    piv = rad.assign(v=rad["AP"].round(3).astype(str) + " (" + rad["recall"].round(2).astype(str) + ")").pivot(index="method", columns="k", values="v")
    print(piv.to_string())
    return rad


def part_a2(dev):
    """Per-axis offset ICNF point - detected cell centre for fresh, isolated targets whose first detection
    slot has one cell, over all non-test years (labels and detections only, no model)."""
    one = dev[(dev["first_slot_cells"] == 1) & ~dev["already_burning"] & dev["isolated"]]
    print(f"\nA2 Offset ICNF point - first detected cell, fresh isolated targets with one cell in the first slot, all non-test years "
          f"(n = {len(one)} of {len(dev)} targets; no model)")
    inside = (one["d_east_km"].abs() <= one["cell_ew_km"] / 2) & (one["d_south_km"].abs() <= one["cell_ns_km"] / 2)
    print(f"   ICNF point inside the detected cell: {inside.mean():.0%} (if the fire were always detected in its own cell and the "
          f"ICNF point were exact: 100 %)")
    rows = []
    for ax, cell in (("d_east_km", "cell_ew_km"), ("d_south_km", "cell_ns_km")):
        v = one[ax].values
        sd_obs = 1.4826 * np.median(np.abs(v - np.median(v)))
        sd_q = (one[cell] / np.sqrt(12)).median()
        ex = np.sqrt(max(sd_obs ** 2 - sd_q ** 2, 0))
        rows.append(dict(axis=ax[2:].replace("_km", ""), median_km=np.median(v), robust_sd_km=sd_obs, quantisation_sd_km=sd_q,
                         remaining_sd_km=ex, share_within_half_cell=np.mean(np.abs(v) <= one[cell].values / 2)))
        print(f"   {ax[2:].replace('_km', ''):5s}: median {np.median(v):+.2f} km, robust SD {sd_obs:.2f} km; pure quantisation SD {sd_q:.2f} km "
              f"-> remaining spread (ICNF position + detection in a neighbour cell + fire spread before the first scan) <= {ex:.2f} km")
    return pd.DataFrame(rows)


def part_b(data, split, runs_points, thr10):
    print("\nB  Data ceiling: all ICNF fires >= 50 ha of the split (nearest point <= 10 km in the fire's period)")
    per = set(data.periods.loc[data.periods["split"] == split, "period"])
    f = data.fires[(data.fires["burned_area_ha"] >= 50) & data.fires["period"].isin(per)].copy()
    print(f"   fires >= 50 ha {len(f)}, seen {f['seen'].sum()} ({f['seen'].mean():.0%}); burned area seen {f.loc[f['seen'], 'burned_area_ha'].sum() / f['burned_area_ha'].sum():.0%}")
    # unseen fires: is there any detection <= 5 km later in the window (after start + 6 h)?
    late = []
    for q in f[~f["seen"]].itertuples():
        a, b = window_index(data, q.period)
        la, lo = G.true_to_sensor_latlon(q.lat, q.lon, q.start_utc)
        s = slot_of(pd.Series([q.start_utc])).iat[0]
        k = np.nonzero(data._slot[a:b] >= s + 24)[0]
        late.append(bool(len(k)) and E.haversine_km(la, lo, data.det["lat"].values[a:b][k], data.det["lon"].values[a:b][k]).min() <= 5)
    print(f"   unseen fires with a detection <= 5 km later in the 24 h window (after start + 6 h): {np.sum(late)} of {len(late)}")
    rows = []
    for name, plist in runs_points.items():
        for s, pts in enumerate(plist):
            pts = pts.assign(period=pd.to_datetime(pts["period"], utc=True))
            by = {p: g for p, g in pts.groupby("period")}
            d = np.array([E.haversine_km(q.lat, q.lon, by[q.period]["lat"].values, by[q.period]["lon"].values).min()
                          if q.period in by else np.inf for q in f.itertuples()])
            dop = np.array([E.haversine_km(q.lat, q.lon, g["lat"].values, g["lon"].values).min()
                            if q.period in by and len(g := by[q.period][by[q.period]["score"] >= thr10[name][s]]) else np.inf
                            for q in f.itertuples()])
            for grp, m in (("seen", f["seen"].values), ("unseen", ~f["seen"].values), ("all", np.ones(len(f), bool))):
                a = f["burned_area_ha"].values[m]
                rows.append(dict(method=name, seed=s, group=grp, n=m.sum(), found=np.mean(d[m] <= 10), found_op=np.mean(dop[m] <= 10),
                                 area_found=a[d[m] <= 10].sum() / a.sum(), area_found_op=a[dop[m] <= 10].sum() / a.sum()))
            for lo_, hi_ in ((50, 100), (100, 500), (500, 1000), (1000, np.inf)):
                m = (f["burned_area_ha"].values >= lo_) & (f["burned_area_ha"].values < hi_)
                rows.append(dict(method=name, seed=s, group=f"all {lo_}-{hi_} ha", n=m.sum(), found=np.mean(d[m] <= 10), found_op=np.mean(dop[m] <= 10),
                                 seen_share=f["seen"].values[m].mean()))
    t = pd.DataFrame(rows).groupby(["method", "group"], sort=False).mean(numeric_only=True).drop(columns="seed")
    print("   share found with all points / at the operating point, and share of the burned area found (mean of 3 seeds):")
    print(t.round(3).to_string())
    return t


def part_c(data, split):
    print("\nC  Target tensors (labels only, no model): targets with their own peak (one-to-one, <= 1.5 cells, 3D also <= 1.5 h)")
    per = data.periods[(data.periods["split"] == split) & (data.periods["n_targets"] > 0)]["period"]
    ctx = target_context(data)
    rows = []
    for p in per:
        _, (y3, y2), _, lab = data.sample(p, time_bins=12)
        tg = lab[lab["role"] == "target"]
        th = ctx.loc[tg["fire_id"], "t_h"].values
        for mode in ("2d", "3d"):
            if mode == "2d":
                fi, fj, s = heatmap_to_points(y2, thr=0.3)
                tt = np.zeros(len(fi))
            else:
                fi, fj, s, tt = decode(y3, "3d", thr=0.3)
            free = np.ones(len(tg), bool)
            own = np.zeros(len(tg), bool)
            for a, b, t in zip(fi, fj, tt):                     # peaks by descending value
                d = np.hypot(tg["i_loc"].values - a, tg["j_loc"].values - b)
                ok = free & (d <= 1.5) & ((np.abs(th - t) <= 1.5) if mode == "3d" else True)
                if ok.any():
                    m = int(np.where(ok, d, np.inf).argmin()); free[m] = False; own[m] = True
            for fid, o in zip(tg["fire_id"], own):
                rows.append(dict(mode=mode, fire_id=fid, own=o, in_pair=bool(ctx.loc[fid, "in_pair"])))
    r = pd.DataFrame(rows)
    t = r.groupby(["mode", "in_pair"])["own"].agg(["mean", "size"]).unstack()
    print(f"   all: 2D {r[r['mode'] == '2d']['own'].mean():.1%}, 3D {r[r['mode'] == '3d']['own'].mean():.1%}")
    print(t.round(3).to_string())
    return r


def part_d(runs_scored, data, split, small):
    print("\nD  Operating point and the AP left on the table (k = 5 and 10 km, mean of 3 seeds)")
    rows, cls_rows = [], []
    for name, sc in runs_scored.items():
        for k in (5, 10):
            for s, run in enumerate(sc):
                res, curve, scored = run[k]
                thr = best_f1_threshold(curve)
                n = res["targets"]
                c = fp_classes(scored, data, k, small)
                ap = {"actual": res["AP"]}
                ap["duplicates neutral"] = ap_with_neutral(c, n, ("duplicate",))
                ap["+ label definition neutral"] = ap_with_neutral(c, n, ("duplicate",) + LABEL_DEF)
                ap["+ border / persistent neutral"] = ap_with_neutral(c, n, ("duplicate",) + LABEL_DEF + ("border cell", "persistent cell"))
                ap["every false alarm neutral (= recall, all points)"] = res["recall (all preds)"]
                rows.append(dict(method=name, k=k, seed=s, R_bestF1=res["R @ best F1"], P_bestF1=res["P @ best F1"], thr=thr, **ap))
                fp = c[c["status"] == "fp"]
                for above in (True, False):
                    g = fp[fp["score"] >= thr] if above else fp
                    for cl in FP_CLASSES:
                        cls_rows.append(dict(method=name, k=k, seed=s, at=("best F1" if above else "all points"), cls=cl,
                                             n=(g["fp_class"] == cl).sum(), share=(g["fp_class"] == cl).mean() if len(g) else np.nan))
                if "dup_dt_h" in c and c["dup_dt_h"].notna().any():
                    dd = c.loc[(c["fp_class"] == "duplicate") & (c["score"] >= thr), "dup_dt_h"].dropna()
                    cls_rows.append(dict(method=name, k=k, seed=s, at="best F1", cls="duplicate: median hours to the found point",
                                         n=len(dd), share=dd.median() if len(dd) else np.nan))
    t = pd.DataFrame(rows).groupby(["method", "k"], sort=False).mean(numeric_only=True).drop(columns="seed")
    print(t.round(3).to_string())
    name = next(iter(runs_scored))
    fa = []
    for s_, run in enumerate(runs_scored[name]):
        res, curve, scored = run[10]
        thr = best_f1_threshold(curve)
        c = fp_classes(scored, data, 10, small)
        fa.append(c[(c["status"] == "fp") & (c["score"] >= thr)].assign(seed=s_))
    fa = pd.concat(fa)
    top = fa.groupby("period").size().sort_values(ascending=False) / len(runs_scored[name])
    print(f"   {name}: false alarms above the threshold @ 10 km per period (mean per seed): {len(fa) / len(runs_scored[name]):.1f} in "
          f"{fa['period'].nunique()} periods; the 5 busiest periods hold {top.head(5).sum() / top.sum():.0%}: "
          + ", ".join(f"{p:%Y-%m-%d %H} UTC {v:.1f}" for p, v in top.head(5).items()))
    ct = pd.DataFrame(cls_rows)
    piv = ct.groupby(["at", "method", "k", "cls"], sort=False)[["n", "share"]].mean().unstack("cls")
    for at in ("best F1", "all points"):
        print(f"   false alarms by nearest ICNF evidence, {at}: share (mean count per seed)")
        sub = ct[ct["at"] == at].groupby(["method", "k", "cls"], sort=False)[["n", "share"]].mean()
        sub = (sub["share"].round(2).astype(str) + " (" + sub["n"].round(1).astype(str) + ")").unstack("cls")
        print(sub[[c for c in list(FP_CLASSES) + ["duplicate: median hours to the found point"] if c in sub]].to_string())
    return t, ct


def part_e(tg, runs_scored, split):
    print("\nE  Misses at the operating point (best F1 @ 10 km), mean of 3 seeds")
    rows = []
    for name, sc in runs_scored.items():
        for s, run in enumerate(sc):
            res, curve, scored = run[10]
            thr = best_f1_threshold(curve)
            tp = scored[scored["status"] == "tp"].set_index("fire_id")
            m_score = tg["fire_id"].map(tp["score"])
            hit_any = m_score.notna()
            hit_op = m_score >= thr
            by = {p: g for p, g in scored.groupby("period")}
            near = np.array([q.period in by and E.haversine_km(q.lat, q.lon, by[q.period]["lat"].values, by[q.period]["lon"].values).min() <= 10
                             for q in tg.itertuples()])
            cat = np.where(hit_op, "hit", np.where(hit_any, "found below the threshold", np.where(near, "point taken by another target", "no point within 10 km")))
            rows.append(pd.DataFrame({"method": name, "seed": s, "fire_id": tg["fire_id"].values, "cat": cat,
                                      "score": m_score.values, "thr": thr}))
    r = pd.concat(rows)
    piv = r.groupby(["method", "cat"]).size().unstack(fill_value=0) / 3
    print("   targets per outcome (mean count per seed):")
    print(piv.round(1).to_string())
    st = r[r["method"] == "cnn3d_st"].merge(tg, on="fire_id")
    st["hit"] = st["cat"] == "hit"
    st["found"] = st["cat"].isin(["hit", "found below the threshold"])
    print("   cnn3d_st: recall at the operating point / with all points, by subgroup and signal (3 seeds pooled):")
    st["day_period"] = st["period"].dt.hour == 6
    bins = {
        "period": np.where(st["day_period"], "06-18 UTC", "18-06 UTC"),
        "period x start [h]": np.where(st["day_period"], "day ", "night ") + pd.cut(st["t_h"], [0, 3, 6, 9, 12], right=False).astype(str),
        "isolated": st["isolated"].astype(str), "in_pair": st["in_pair"].astype(str), "already_burning": st["already_burning"].astype(str),
        "seen only via an older nucleus": st["seen_via_older"].astype(str), "size_class": st["size_class"].astype(str),
        "rekindle": st["rekindle"].astype(object).where(st["rekindle"].notna(), "unknown").astype(str),
        "ICNF start in the period [h]": pd.cut(st["t_h"], [0, 3, 6, 9, 12], right=False).astype(str),
        "detections <= 1.5 cells in the first 6 h": pd.cut(st["n_det_6h"], [-1, 0, 2, 5, 10, 20, 1e9], labels=["0", "1-2", "3-5", "6-10", "11-20", "> 20"]).astype(str),
        "FRE <= 1.5 cells in the window [GJ]": pd.qcut(st["fre_gj"].rank(method="first"), 4, labels=["q1 (low)", "q2", "q3", "q4 (high)"]).astype(str),
        "first detection after the ICNF start [h]": pd.cut(st["first_lag_h"], [-2, 0.5, 1, 2, 4, 24], labels=["<= 0.5", "0.5-1", "1-2", "2-4", "> 4"]).astype(str),
        f"burned <= 1.5 cells before the window (up to {PREV_D} d)": pd.cut(st["prev_h"], [0, 24, 72, 24 * PREV_D + 1], right=False,
                                                                             labels=["within 24 h", "1-3 d", f"3-{PREV_D} d"]).astype(str).replace("nan", f"not in {PREV_D} d"),
    }
    out = []
    for gname, v in bins.items():
        g = st.groupby(v)
        t = pd.DataFrame({"targets": g.size() / 3, "recall @ best F1": g["hit"].mean(), "recall all points": g["found"].mean(),
                          "median score of the match": g["score"].median()})
        t.index = pd.MultiIndex.from_product([[gname], t.index])
        out.append(t)
    t = pd.concat(out)
    print(t.round(3).to_string())
    edges = st["fre_gj"].quantile([0.25, 0.5, 0.75]).round(1).tolist()
    print(f"   FRE quartile edges [GJ]: {edges}")
    # one primary reason per miss of cnn3d_st (first that applies)
    miss = st[~st["hit"]].copy()
    weak = (miss["n_det_6h"] <= 5) | (miss["fre_gj"] <= edges[0])
    reason = np.select(
        [miss["cat"] == "no point within 10 km", miss["cat"] == "point taken by another target", miss["in_pair"],
         miss["already_burning"] | miss["seen_via_older"], weak, miss["first_lag_h"] > 2],
        ["no point within 10 km", "point taken by another target", "close pair (<= 2 cells, same period)",
         "next to an active fire (already burning / seen via an older nucleus)", "weak signal (<= 5 detections in 6 h or FRE in the lowest quartile)",
         "detected > 2 h after the ICNF start"], "other (strong, fresh signal, scored low)")
    rr = pd.Series(reason).value_counts() / 3
    print(f"   cnn3d_st misses at the operating point by primary reason (mean count per seed, of {len(miss) / 3:.1f}):")
    print((rr.round(1).astype(str) + "  (" + (rr / rr.sum() * 100).round(0).astype(int).astype(str) + " %)").to_string())
    return r, t


def part_f(runs_points, runs_scored, tg, data, split):
    print("\nF  Ranking: what separates hits from false alarms (all points, k = 10 km, 3 seeds pooled)")
    rows, keep = [], []
    for name, sc in runs_scored.items():
        for s, run in enumerate(sc):
            res, curve, scored = run[10]
            thr = best_f1_threshold(curve)
            p = scored[scored["status"] != "ignored"].copy()
            p = point_signal(p, data)
            p["method"], p["seed"], p["thr"] = name, s, thr
            keep.append(p)
    pts = pd.concat(keep, ignore_index=True)
    for name, g in pts.groupby("method", sort=False):
        tp, fp = g[g["status"] == "tp"], g[g["status"] == "fp"]
        hi = g[g["score"] >= g["thr"]]
        line = (f"   {name}: AUC score {auc(tp['score'], fp['score']):.3f}; FRE alone {auc(tp['p_fre_gj'], fp['p_fre_gj']):.3f}; "
                f"detections alone {auc(tp['p_ndet'], fp['p_ndet']):.3f}; hours with detections {auc(tp['p_hours'], fp['p_hours']):.3f}")
        print(line)
        print(f"      median FRE [GJ]: hits {tp['p_fre_gj'].median():.0f}, false alarms {fp['p_fre_gj'].median():.0f}; above the threshold: "
              f"hits {hi.loc[hi['status'] == 'tp', 'p_fre_gj'].median():.0f}, false alarms {hi.loc[hi['status'] == 'fp', 'p_fre_gj'].median():.0f}")
        print(f"      share with activity in the 6 h history: hits {np.mean(tp['p_hist'] > 0):.2f}, false alarms {np.mean(fp['p_hist'] > 0):.2f}")
        print(f"      false alarms above the threshold with FRE above the median FRE of the hits: {np.mean(hi.loc[hi['status'] == 'fp', 'p_fre_gj'] > tp['p_fre_gj'].median()):.2f}")
        bins_ = [0, 24, 72, 24 * PREV_D + 1]
        for lab_, d in (("hits (all)", tp), ("false alarms (all)", fp), ("hits above thr", hi[hi["status"] == "tp"]), ("false alarms above thr", hi[hi["status"] == "fp"])):
            c = pd.cut(d["p_prev_h"], bins_, right=False).value_counts(normalize=True, sort=False)
            print(f"      {lab_:22s} burned <= 1.5 cells before the window (not in the input): within 24 h {c.iloc[0]:.2f}, 1-3 d {c.iloc[1]:.2f}, "
                  f"3-{PREV_D} d {c.iloc[2]:.2f}, not in {PREV_D} d {d['p_prev_h'].isna().mean():.2f}")
    fc = []
    for name, sc in runs_scored.items():
        for s_, run in enumerate(sc):
            c = fp_classes(run[10][2], data, 10, SMALL[0])
            c["method"] = name
            fc.append(c)
    fc = pd.concat(fc, ignore_index=True)
    for name, g in fc.groupby("method", sort=False):
        tp = g.loc[g["status"] == "tp", "score"]
        parts = [f"{cl} {auc(tp, g.loc[g['fp_class'] == cl, 'score']):.2f}" for cl in FP_CLASSES if (g["fp_class"] == cl).any()]
        print(f"   {name}: AUC of the score, hits vs false alarms of each class: " + "; ".join(parts))
    # score of the hits vs observation time left after the ICNF start (cnn3d_st)
    st = pts[(pts["method"] == "cnn3d_st") & (pts["status"] == "tp")].merge(tg[["fire_id", "t_h", "obs_h", "burned_area_ha", "fre_gj"]], on="fire_id", suffixes=("", "_tg"))
    if len(st):
        st["start_bin"] = pd.cut(st["t_h_tg"] if "t_h_tg" in st else st["t_h"], [0, 3, 6, 9, 12], right=False).astype(str)
        st["above"] = st["score"] >= st["thr"]
        g = st.groupby("start_bin")
        print("   cnn3d_st hits by ICNF start hour in the period: median score, share above the threshold")
        print(pd.DataFrame({"hits": g.size() / 3, "median score": g["score"].median(), "above threshold": g["above"].mean()}).round(3).to_string())
        rho = pd.Series(st["score"]).corr(np.log1p(st["burned_area_ha"]), method="spearman")
        rho2 = pd.Series(st["score"]).corr(np.log1p(st["p_fre_gj"]), method="spearman")
        print(f"   cnn3d_st hits: Spearman(score, log burned area) {rho:.2f}; Spearman(score, log FRE at the point) {rho2:.2f}")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true", help="needed with --split test (descriptive, after 9.20)")
    args = ap.parse_args()
    assert args.split == "val" or args.confirm_test, "--split test needs --confirm-test"
    split = args.split
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 40)
    OUT.mkdir(parents=True, exist_ok=True)
    data = FireData()
    assert data.georef_fix
    print(f"Error analysis on {split} ({'dev models, trained without these years' if split == 'val' else 'final models, DESCRIPTIVE ONLY'})")
    tg = target_table(data, split)
    tg.to_csv(OUT / f"{split}_targets.csv", index=False)
    dev_f = OUT / "dev_targets.csv"                       # all non-test years, labels + detections only
    if dev_f.exists() and dev_f.stat().st_mtime > (E.PROC / "model" / "labels.parquet").stat().st_mtime:
        dev = pd.read_csv(dev_f)
    else:
        dev = target_table(data, ("train", "val", "train2", "train3"))
        dev.to_csv(dev_f, index=False)

    runs_points = {name: [cached_points(n, split) for n in names] for name, names in RUNS[split].items()}
    runs_scored = {}
    for name, plist in runs_points.items():
        runs_scored[name] = []
        for p in plist:
            runs_scored[name].append({k: evaluate(p, data, split, k, groups=()) for k in (5, 10)})
        runs_scored[name][0]["points_list"] = plist
    bp = baseline_predictions(data)["birth"]
    base = {k: evaluate(bp, data, split, k, groups=()) for k in (5, 10)}
    base["points"] = bp
    thr10 = {name: [best_f1_threshold(r[10][1]) for r in sc] for name, sc in runs_scored.items()}
    for name, sc in runs_scored.items():
        print(f"   {name}: AP@5 {np.mean([r[5][0]['AP'] for r in sc]):.3f}, AP@10 {np.mean([r[10][0]['AP'] for r in sc]):.3f} "
              f"(reproduces the stored scores); best-F1 thresholds @ 10 km {np.round(thr10[name], 3).tolist()}")

    rad = part_a(tg, runs_scored, data, split, base)
    rad.to_csv(OUT / f"{split}_radius.csv", index=False)
    if split == "val":
        part_a2(dev).to_csv(OUT / "dev_offset_axes.csv", index=False)
    part_b(data, split, runs_points, thr10).to_csv(OUT / f"{split}_data_ceiling.csv")
    part_c(data, split).to_csv(OUT / f"{split}_target_tensors.csv", index=False)
    small = icnf_small(data, split)
    SMALL[:] = [small]
    t, ct = part_d(runs_scored, data, split, small)
    t.to_csv(OUT / f"{split}_ap_ceiling.csv"); ct.to_csv(OUT / f"{split}_fp_classes.csv", index=False)
    r, _ = part_e(tg, runs_scored, split)
    r.to_csv(OUT / f"{split}_miss_outcomes.csv", index=False)
    pts = part_f(runs_points, runs_scored, tg, data, split)
    pts.drop(columns=["si", "sj"]).to_csv(OUT / f"{split}_points.csv", index=False)
    print(f"\nwritten: outputs/error_analysis/{split}_*.csv")


if __name__ == "__main__":
    main()
