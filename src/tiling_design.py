"""Numbers behind the tiling / sample design for the heatmap model (notes.md 4.1-4.2).

Design (draft):
  grid     native SEVIRI grid: row = y / SSD, col = x / SSD in the geostationary projection
           (sub-satellite longitude 0 deg, valid for all of 2004-2019, see notes.md 4.2)
  periods  12 h target periods [06, 18) and [18, 06) UTC; input = target period plus 6 h
           history before and 6 h after (24 h = 96 time steps of 15 min)
  targets  ICNF fires >= TARGET_HA starting in the target period and seen by SEVIRI
           (detection within 5 km in [start - 1 h, start + 6 h]); other fires >= IGNORE_HA
           and unseen target-size fires are 'ignore' regions (no loss)
  crops    CROP x CROP native pixels for training; full domain at inference
  split    chronological: train 2004-2014, validation 2015-2016, test 2017-2019

Outputs: outputs/tiling_design_log.txt (stdout), figures/tiling/example_sample.png

Run from the repository root:
  .venv/bin/python src/tiling_design.py > outputs/tiling_design_log.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import eda_figures as E
from grid import native_rc

PROC = E.PROC
FIG = E.ROOT / "figures" / "tiling"

TARGET_HA, IGNORE_HA = 50, 10
PERIOD_H, HISTORY_H, TAIL_H = 12, 6, 6
PERIOD_START_UTC = 6          # target periods start at 06 and 18 UTC (diurnal minimum of activity)
SEEN_KM, SEEN_PRE_H, SEEN_POST_H = 5, 1, 6
CROPS = (16, 32, 64)
CROP = 32
SPLIT = {"train": range(2004, 2015), "val": range(2015, 2017), "test": range(2017, 2020)}

def load():
    sev = pd.read_parquet(PROC / "seviri_frp.parquet").sort_values("time_utc").reset_index(drop=True)
    r, c = native_rc(sev["lat"].values, sev["lon"].values)
    sev["row"], sev["col"] = np.round(r).astype(int), np.round(c).astype(int)
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    f = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
             & icnf["start_utc"].between(pd.Timestamp(E.PERIOD[0], tz="UTC"), pd.Timestamp(E.PERIOD[1], tz="UTC"))
             & (icnf["burned_area_ha"] >= IGNORE_HA)].reset_index(drop=True)
    f["row_f"], f["col_f"] = native_rc(f["lat"].values, f["lon"].values)
    f["seen"] = seen(f, sev)
    f["year"] = f["start_utc"].dt.year
    shift = pd.Timedelta(hours=PERIOD_START_UTC)
    f["period"] = (f["start_utc"] - shift).dt.floor(f"{PERIOD_H}h") + shift
    f["role"] = np.where((f["burned_area_ha"] >= TARGET_HA) & f["seen"], "target", "ignore")
    f["split"] = f["year"].map({y: k for k, ys in SPLIT.items() for y in ys})
    return sev, f


def seen(f, sev):
    t = sev["time_utc"].values.astype("datetime64[ns]").astype("int64")
    h = 3_600_000_000_000
    s0 = f["start_utc"].values.astype("datetime64[ns]").astype("int64")
    i0, i1 = np.searchsorted(t, s0 - SEEN_PRE_H * h), np.searchsorted(t, s0 + SEEN_POST_H * h)
    out = np.zeros(len(f), bool)
    for k in range(len(f)):
        if i1[k] > i0[k]:
            out[k] = (E.haversine_km(f["lat"].iat[k], f["lon"].iat[k],
                                     sev["lat"].values[i0[k]:i1[k]], sev["lon"].values[i0[k]:i1[k]]) <= SEEN_KM).any()
    return out


def grid_check(sev):
    px = sev[["lat", "lon", "row", "col"]].drop_duplicates(["lat", "lon"])
    r, c = native_rc(px["lat"].values, px["lon"].values)
    res = np.maximum(np.abs(r - px["row"]), np.abs(c - px["col"]))
    print("Native grid (geostationary projection, sub-satellite longitude 0°):")
    print(f"  {len(px):,} lat/lon pixels -> {len(px[['row', 'col']].drop_duplicates()):,} native pixels "
          f"(collisions: {len(px) - len(px[['row', 'col']].drop_duplicates())}); "
          f"max |residual| {res.max():.2f} px, p95 {np.percentile(res, 95):.2f} px")
    print(f"  rows {sev['row'].min()}–{sev['row'].max()}, cols {sev['col'].min()}–{sev['col'].max()} "
          f"-> domain {sev['row'].max() - sev['row'].min() + 1} × {sev['col'].max() - sev['col'].min() + 1} pixels")


def availability(f):
    print("\nICNF fires 2004–2019 (valid coordinates) seen by SEVIRI "
          f"(detection ≤ {SEEN_KM} km in [start − {SEEN_PRE_H} h, start + {SEEN_POST_H} h]):")
    rows = []
    for th in (10, 50, 100, 500):
        g = f[f["burned_area_ha"] >= th]
        rows.append({"size": f"≥ {th} ha", "fires": len(g), "seen": int(g["seen"].sum()),
                     "seen [%]": round(g["seen"].mean() * 100), "days with ≥ 1 seen": g[g["seen"]]["start_utc"].dt.floor("D").nunique()})
    print(pd.DataFrame(rows).to_string(index=False))
    per_year = pd.DataFrame({th: f[(f["burned_area_ha"] >= th) & f["seen"]].groupby("year").size() for th in (50, 100)})
    print("\nSeen fires per year:")
    print(per_year.T.to_string())
    big3 = per_year.loc[[2005, 2016, 2017], 100].sum() / per_year[100].sum() * 100
    print(f"  share of seen fires ≥ 100 ha in 2005, 2016, 2017: {big3:.0f} %")


def periods(sev, f):
    shift = pd.Timedelta(hours=PERIOD_START_UTC)
    p0 = (pd.Timestamp(E.PERIOD[0], tz="UTC") - shift).floor("D") + shift
    p1 = pd.Timestamp(E.PERIOD[1], tz="UTC")
    allp = pd.date_range(p0, p1, freq=f"{PERIOD_H}h")
    # detections inside mainland Portugal during the input window of each period
    _, mainland = E.load_borders()
    in_pt = np.asarray(E.in_polygon(sev["lon"], sev["lat"], mainland))
    t = sev.loc[in_pt, "time_utc"].values.astype("datetime64[ns]").astype("int64")
    a = (allp - pd.Timedelta(hours=HISTORY_H)).values.astype("datetime64[ns]").astype("int64")
    b = (allp + pd.Timedelta(hours=PERIOD_H + TAIL_H)).values.astype("datetime64[ns]").astype("int64")
    n_det = np.searchsorted(t, b) - np.searchsorted(t, a)
    tg = f[f["role"] == "target"]
    per = pd.DataFrame({"period": allp, "n_det_pt": n_det}).set_index("period")
    per["targets"] = tg.groupby("period").size().reindex(per.index, fill_value=0)
    per["split"] = per.index.year.map({y: k for k, ys in SPLIT.items() for y in ys})

    print(f"\nTarget periods of {PERIOD_H} h ({PERIOD_START_UTC:02d}/{(PERIOD_START_UTC + 12) % 24:02d} UTC), "
          f"input {HISTORY_H} h + {PERIOD_H} h + {TAIL_H} h = {(HISTORY_H + PERIOD_H + TAIL_H) * 4} time steps; "
          f"targets: fires ≥ {TARGET_HA} ha seen by SEVIRI:")
    pos = per[per["targets"] > 0]
    busy = per[(per["targets"] == 0) & (per["n_det_pt"] > 0)]
    print(f"  periods: {len(per):,}; with detections in mainland Portugal: {(per['n_det_pt'] > 0).sum():,}; "
          f"with ≥ 1 target: {len(pos):,}; with detections but no target (hard negatives): {len(busy):,}")
    k = pos["targets"]
    print(f"  targets per positive period: 1: {(k == 1).sum()}, 2–5: {k.between(2, 5).sum()}, 6–20: {k.between(6, 20).sum()}, "
          f"> 20: {(k > 20).sum()} (max {k.max()}); total targets {k.sum():,}; ignore regions {(f['role'] == 'ignore').sum():,}")
    by = pd.DataFrame({
        "years": {s: f"{min(ys)}–{max(ys)}" for s, ys in SPLIT.items()},
        "target fires": tg.groupby("split").size(),
        "positive periods": pos.groupby("split").size(),
        "hard-negative periods": busy.groupby("split").size(),
    }).loc[list(SPLIT)]
    by["target fires [%]"] = (by["target fires"] / by["target fires"].sum() * 100).round(1)
    print("\nChronological split:")
    print(by.to_string())
    return per


def crops(sev, f):
    """For every target fire, a crop centred on it: how many other targets share the crop,
    and does the crop already contain detections before the fire started (ongoing fires)?"""
    tg = f[f["role"] == "target"].reset_index(drop=True)
    t = sev["time_utc"].values.astype("datetime64[ns]").astype("int64")
    row, col = sev["row"].values, sev["col"].values
    print(f"\nCrops centred on each target fire ({len(tg):,} targets):")
    out = []
    for size in CROPS:
        half = size / 2
        multi, busy = 0, 0
        for p, g in tg.groupby("period"):
            for x in g.itertuples():
                others = ((np.abs(g["row_f"] - x.row_f) <= half) & (np.abs(g["col_f"] - x.col_f) <= half)).sum() - 1
                multi += others > 0
                i0 = np.searchsorted(t, (x.period - pd.Timedelta(hours=HISTORY_H)).value)
                i1 = np.searchsorted(t, x.start_utc.value)
                busy += ((np.abs(row[i0:i1] - x.row_f) <= half) & (np.abs(col[i0:i1] - x.col_f) <= half)).any()
        out.append({"crop [px]": f"{size} × {size}", "≈ km": f"{size * 3.2:.0f} × {size * 4.5:.0f}",
                    "with another target [%]": round(multi / len(tg) * 100, 1),
                    "with earlier activity in the input [%]": round(busy / len(tg) * 100, 1)})
    print(pd.DataFrame(out).to_string(index=False))


def fig_example(sev, f):
    """One training sample: a CROP x CROP crop around the largest target of 2017-10-15."""
    FIG.mkdir(parents=True, exist_ok=True)
    tg = f[(f["role"] == "target") & (f["period"] == f.loc[f["burned_area_ha"].idxmax(), "period"])]
    x = tg.loc[tg["burned_area_ha"].idxmax()]
    r0, c0 = int(round(x.row_f)) - CROP // 2, int(round(x.col_f)) - CROP // 2
    t0 = x.period - pd.Timedelta(hours=HISTORY_H)
    parts = [("history (6 h before the period)", t0, x.period),
             ("target period (12 h)", x.period, x.period + pd.Timedelta(hours=PERIOD_H)),
             ("tail (6 h after)", x.period + pd.Timedelta(hours=PERIOD_H), x.period + pd.Timedelta(hours=PERIOD_H + TAIL_H))]
    inside = lambda d: d[d["row"].between(r0, r0 + CROP - 1) & d["col"].between(c0, c0 + CROP - 1)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), layout="constrained")
    for ax, (title, a, b) in zip(axes, parts):
        d = inside(sev[(sev["time_utc"] >= a) & (sev["time_utc"] < b)])
        img = np.zeros((CROP, CROP))
        np.maximum.at(img, (r0 + CROP - 1 - d["row"].values, d["col"].values - c0), d["frp_mw"].values)
        im = ax.imshow(np.log1p(img), cmap=E.SEQ_BLUE, vmin=0, vmax=np.log1p(1000), interpolation="nearest")
        ax.set_title(f"input: max FRP, {title}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes[:3], shrink=0.8, pad=0.01, label="log(1 + FRP [MW])")
    # target heatmap: Gaussian (sigma 1 px) per target, ignore discs (radius 2 px) for other fires
    rr, cc = np.mgrid[0:CROP, 0:CROP]
    heat, ignore, n_in = np.zeros((CROP, CROP)), np.zeros((CROP, CROP), bool), 0
    near = f[(f["period"] == x.period)]
    for y in near.itertuples():
        pr, pc = r0 + CROP - 1 - y.row_f, y.col_f - c0
        if not (-2 <= pr < CROP + 2 and -2 <= pc < CROP + 2):
            continue
        if y.role == "target":
            heat = np.maximum(heat, np.exp(-((rr - pr) ** 2 + (cc - pc) ** 2) / 2))
            n_in += 0 <= pr < CROP and 0 <= pc < CROP
        else:
            ignore |= (rr - pr) ** 2 + (cc - pc) ** 2 <= 4
    ax = axes[3]
    ax.imshow(heat, cmap=E.SEQ_ORANGE, vmin=0, vmax=1, interpolation="nearest")
    ax.contour(ignore, levels=[0.5], colors=E.INK, linewidths=1)
    ax.set_title(f"target heatmap: {n_in} target ignitions in this crop", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Example training sample: {CROP} × {CROP} native pixels around {x.municipality.title()} "
                 f"({x.burned_area_ha:,.0f} ha), period starting {x.period:%Y-%m-%d %H:%M} UTC "
                 "(black contour = ignore region: other fires ≥ 10 ha or unseen)", fontsize=10, x=0.01, ha="left")
    fig.savefig(FIG / "example_sample.png"); plt.close(fig)
    print("\nfigure: figures/tiling/example_sample.png")


def main():
    sev, f = load()
    grid_check(sev)
    availability(f)
    periods(sev, f)
    crops(sev, f)
    fig_example(sev, f)


if __name__ == "__main__":
    main()
