"""Build the sample index for the heatmap model (notes.md 4.2, 5.1).

Everything is stored sparsely; src/dataset.py builds the dense tensors of a sample on demand.

Outputs (data/processed/model/):
  detections.parquet  all SEVIRI detections with image indices (i, j) on the native grid and
                      the 15-min slot; plus event_id / nucleus_id / persistent / in_pt
  fires.parquet       ICNF fires >= IGNORE_HA (2004-2019, valid coordinates, UTC start) with
                      continuous image position (i_f, j_f), seen flag, period, split, isolation
  labels.parquet      (period, fire) pairs with role 'target' or 'ignore'; i_f / j_f here are the
                      ignition position in the SEVIRI *sensor* frame of that period (see below)
  periods.parquet     one row per 12 h target period: split, detections, targets, ignores
  static.npz          per-cell lat, lon, mainland mask (pt), loss mask (pt dilated by 1 px),
                      persistent off-season pixels
  build_info.json     settings of this build (geo-referencing correction on/off, change date)

Geo-referencing correction (notes.md 6.11, default on): SEVIRI positions before
grid.GEOREF_CHANGE lie half a native pixel north-west of the true ground position. Detections are
kept on their native pixels; the ICNF ignition positions are moved into the sensor frame of each
period (labels.parquet) and the 'seen' test compares ICNF with SEVIRI in the same frame.
fires.parquet keeps the true positions. Predictions are moved back to the true frame before
scoring (train.maps_to_points, baselines.py). --no-georef-fix reproduces the uncorrected build.

Run from the repository root:
  .venv/bin/python src/build_dataset.py > outputs/build_dataset_log.txt
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

import eda_figures as E
import grid as G
from proximity_analysis import count_neighbours
from tiling_design import (HISTORY_H, IGNORE_HA, PERIOD_H, PERIOD_START_UTC, SPLIT, TAIL_H, TARGET_HA,
                           seen)

PROC = E.PROC
OUT = PROC / "model"
BOUNDARY_H = 1                 # seen target-size fires starting within 1 h outside a period: ignore
ISOLATION = (10, 20, 72)       # no other fire >= 10 ha within 20 km and +-72 h (as in 3.1)
SIZE_BINS = [10, 50, 100, 500, 1000, np.inf]
SIZE_LABELS = ["10–50", "50–100", "100–500", "500–1k", "≥1k"]


def period_of(t):
    shift = pd.Timedelta(hours=PERIOD_START_UTC)
    return (t - shift).dt.floor(f"{PERIOD_H}h") + shift


def detections():
    d = pd.read_parquet(PROC / "seviri_frp_events.parquet")
    i, j = G.latlon_to_ij(d["lat"].values, d["lon"].values)
    d["i"], d["j"] = np.round(i).astype(np.int16), np.round(j).astype(np.int16)
    assert d["i"].between(0, G.H - 1).all() and d["j"].between(0, G.W - 1).all()
    d["slot"] = (d["time_utc"].values.astype("datetime64[m]").astype(np.int64) // 15).astype(np.int64)
    return d


def fires(det, georef_fix=True):
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    f = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
             & icnf["start_utc"].between(pd.Timestamp(E.PERIOD[0], tz="UTC"), pd.Timestamp(E.PERIOD[1], tz="UTC"))
             & (icnf["burned_area_ha"] >= IGNORE_HA)].reset_index(drop=True)
    f["i_f"], f["j_f"] = G.latlon_to_ij(f["lat"].values, f["lon"].values)
    if georef_fix:     # compare ICNF and SEVIRI in the same (sensor) frame
        lat_s, lon_s = G.true_to_sensor_latlon(f["lat"].values, f["lon"].values, f["start_utc"])
        f["seen"] = seen(f.assign(lat=lat_s, lon=lon_s), det)
    else:
        f["seen"] = seen(f, det)
    f["period"] = period_of(f["start_utc"])
    f["year"] = f["start_utc"].dt.year
    f["split"] = f["year"].map({y: k for k, ys in SPLIT.items() for y in ys})
    ha, km, h = ISOLATION
    f["isolated"] = count_neighbours(f, f[f["burned_area_ha"] >= ha], [km], h)[:, 0] == 0
    f["size_class"] = pd.cut(f["burned_area_ha"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    f["is_target"] = (f["burned_area_ha"] >= TARGET_HA) & f["seen"]
    return f[["fire_id", "start_utc", "end_utc", "lat", "lon", "i_f", "j_f", "burned_area_ha", "size_class",
              "municipality", "district", "seen", "is_target", "isolated", "period", "year", "split"]]


def labels(f, georef_fix=True):
    own = f[["period", "fire_id"]].assign(role=np.where(f["is_target"], "target", "ignore"))
    # target-size fires starting within BOUNDARY_H outside a period: ignore in that period
    b = pd.Timedelta(hours=BOUNDARY_H)
    tg = f[f["is_target"]]
    near_prev = tg[(tg["start_utc"] - tg["period"]) < b]                               # early in its period
    near_next = tg[(tg["period"] + pd.Timedelta(hours=PERIOD_H) - tg["start_utc"]) <= b]  # late in its period
    extra = pd.concat([
        near_prev.assign(period=near_prev["period"] - pd.Timedelta(hours=PERIOD_H)),
        near_next.assign(period=near_next["period"] + pd.Timedelta(hours=PERIOD_H)),
    ])[["period", "fire_id"]].assign(role="ignore")
    lab = pd.concat([own, extra], ignore_index=True)
    lab = lab.merge(f[["fire_id", "i_f", "j_f"]], on="fire_id")
    if georef_fix:     # ignition position in the sensor frame of the label's period
        lab["i_f"], lab["j_f"] = G.true_to_sensor_ij(lab["i_f"].values, lab["j_f"].values, lab["period"])
    return lab


def periods(det, lab):
    shift = pd.Timedelta(hours=PERIOD_START_UTC)
    p0 = (pd.Timestamp(E.PERIOD[0], tz="UTC") - shift).floor("D") + shift
    per = pd.DataFrame({"period": pd.date_range(p0, pd.Timestamp(E.PERIOD[1], tz="UTC"), freq=f"{PERIOD_H}h")})
    per["split"] = per["period"].dt.year.map({y: k for k, ys in SPLIT.items() for y in ys})
    t = det["time_utc"].values.astype("datetime64[ns]").astype("int64")
    a = (per["period"] - pd.Timedelta(hours=HISTORY_H)).values.astype("datetime64[ns]").astype("int64")
    b = (per["period"] + pd.Timedelta(hours=PERIOD_H + TAIL_H)).values.astype("datetime64[ns]").astype("int64")
    per["n_det"] = np.searchsorted(t, b) - np.searchsorted(t, a)
    tpt = det.loc[det["in_pt"], "time_utc"].values.astype("datetime64[ns]").astype("int64")
    per["n_det_pt"] = np.searchsorted(tpt, b) - np.searchsorted(tpt, a)
    counts = lab.groupby(["period", "role"]).size().unstack(fill_value=0)
    per = per.merge(counts, left_on="period", right_index=True, how="left").fillna({"target": 0, "ignore": 0})
    per = per.rename(columns={"target": "n_targets", "ignore": "n_ignore"})
    per[["n_targets", "n_ignore"]] = per[["n_targets", "n_ignore"]].astype(int)
    return per


def static(det):
    _, mainland = E.load_borders()
    lat, lon = G.cell_latlon()
    pt = np.asarray(E.in_polygon(lon.ravel(), lat.ravel(), mainland)).reshape(G.H, G.W)
    persistent = np.zeros((G.H, G.W), bool)
    p = det[det["persistent"]].drop_duplicates(["i", "j"])
    persistent[p["i"], p["j"]] = True
    pt_loss = binary_dilation(pt, structure=np.ones((3, 3), bool))   # coastal / border cells
    return dict(lat=lat.astype(np.float32), lon=lon.astype(np.float32), pt=pt, pt_loss=pt_loss, persistent=persistent)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    georef_fix = "--no-georef-fix" not in sys.argv
    det = detections()
    f = fires(det, georef_fix)
    lab = labels(f, georef_fix)
    print(f"Geo-referencing correction: {'ON' if georef_fix else 'OFF'}"
          + (f" (labels before {G.GEOREF_CHANGE} moved {G.GEOREF_SHIFT_PX} px north-west into the sensor frame)" if georef_fix else ""))
    per = periods(det, lab)
    st = static(det)

    print(f"Grid: {G.H} × {G.W} cells; mainland Portugal cells: {st['pt'].sum():,} (loss mask incl. 1-px border: {st['pt_loss'].sum():,}); "
          f"persistent off-season cells: {st['persistent'].sum()}")
    print(f"Detections: {len(det):,} (all inside the grid)")
    print(f"Fires ≥ {IGNORE_HA} ha: {len(f):,}; targets (≥ {TARGET_HA} ha and seen): {f['is_target'].sum():,}; "
          f"isolated targets: {(f['is_target'] & f['isolated']).sum():,}")
    ti, tj = np.round(f.loc[f["is_target"], "i_f"]).astype(int), np.round(f.loc[f["is_target"], "j_f"]).astype(int)
    outside = ~st["pt_loss"][ti.clip(0, G.H - 1), tj.clip(0, G.W - 1)]
    print(f"Targets outside the loss mask: {outside.sum()}")
    lt = lab[lab["role"] == "target"]
    li, lj = np.round(lt["i_f"]).astype(int).clip(0, G.H - 1), np.round(lt["j_f"]).astype(int).clip(0, G.W - 1)
    print(f"Target labels (sensor frame) outside the loss mask: {(~st['pt_loss'][li, lj]).sum()}")
    print(f"Labels: {(lab['role'] == 'target').sum():,} target and {(lab['role'] == 'ignore').sum():,} ignore entries "
          f"({len(lab) - len(f):,} extra ignores from the ±{BOUNDARY_H} h boundary rule)")
    print("\nPeriods by split:")
    print(per.groupby("split").agg(periods=("period", "size"), with_detections=("n_det", lambda x: (x > 0).sum()),
                                   with_targets=("n_targets", lambda x: (x > 0).sum()), targets=("n_targets", "sum"))
          .loc[list(SPLIT)].to_string())

    det.to_parquet(OUT / "detections.parquet", index=False)
    f.to_parquet(OUT / "fires.parquet", index=False)
    lab.to_parquet(OUT / "labels.parquet", index=False)
    per.to_parquet(OUT / "periods.parquet", index=False)
    np.savez_compressed(OUT / "static.npz", **st)
    info = {"georef_fix": georef_fix, "georef_change": G.GEOREF_CHANGE, "georef_shift_px": G.GEOREF_SHIFT_PX,
            "targets": int(f["is_target"].sum()), "labels": len(lab)}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(f"\nwritten: {OUT.relative_to(E.ROOT)}/ (detections, fires, labels, periods .parquet, static.npz, build_info.json)")


if __name__ == "__main__":
    main()
