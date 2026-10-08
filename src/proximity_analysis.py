"""How often are fires and hotspots close to each other in space and time?

Part A, ICNF ignition points: for each focus fire (>= 100 ha, 2004-2019, valid coordinates)
  count other fires whose ignition lies within D km and whose start is within +-T hours.
  Control: the same count with the focus fire's start shifted by +-7 days (same season and
  place, but no real simultaneity), averaged over both shifts.
Part B, SEVIRI hotspots: for each focus fire, take all detections that match it (same
  radius R(A) and window as in eda_figures.py: [start - 1 h, end + 24 h]) and compute the
  share that also matches at least one other fire.

Neighbour pools by minimum size: any, >= 10 ha, >= 100 ha (small fires are rarely seen by
SEVIRI, so >= 10 ha neighbours are the relevant ones for hotspot mixing).

Outputs: figures/eda/10_neighbouring_fires.png, figures/eda/11_shared_hotspots.png,
         numbers printed to stdout (saved to outputs/proximity_analysis_log.txt).

Run from the repository root:
  .venv/bin/python src/proximity_analysis.py > outputs/proximity_analysis_log.txt
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E   # shared helpers, palette and matching

ROOT = E.ROOT
PROC, FIG = E.PROC, E.FIG

FOCUS_HA = 100
DISTS = [5, 10, 20, 30, 50, 100]          # km
WINDOWS_H = [24, 72]                      # start-time difference
POOLS = [("any size", 0), ("≥ 10 ha", 10), ("≥ 100 ha", 100)]
SHIFT_DAYS = 7


def count_neighbours(focus, pool, dists, window_h, shift_days=0):
    """For each focus fire, the number of pool fires (excluding itself) within each distance
    and with |start difference| <= window_h, after shifting the focus start by shift_days."""
    pool = pool.sort_values("start_utc")
    t = pool["start_utc"].values.astype("datetime64[ns]").astype("int64")
    lat, lon, fid = pool["lat"].values, pool["lon"].values, pool["fire_id"].values
    w = int(window_h * 3.6e12)
    out = np.zeros((len(focus), len(dists)), int)
    for k, f in enumerate(focus.itertuples()):
        s = f.start_utc.value + int(shift_days * 24 * 3.6e12)
        i0, i1 = np.searchsorted(t, [s - w, s + w + 1])
        if i1 == i0:
            continue
        d = E.haversine_km(f.lat, f.lon, lat[i0:i1], lon[i0:i1])
        d = d[fid[i0:i1] != f.fire_id]
        out[k] = [(d <= D).sum() for D in dists]
    return out


def part_a(fires):
    focus = fires[fires["burned_area_ha"] >= FOCUS_HA].reset_index(drop=True)
    print(f"Part A: {len(focus):,} focus fires (≥ {FOCUS_HA} ha, 2004–2019)")
    res = {}
    for wh in WINDOWS_H:
        for name, ha in POOLS:
            pool = fires[fires["burned_area_ha"] >= ha]
            real = (count_neighbours(focus, pool, DISTS, wh) > 0).mean(axis=0) * 100
            ctrl = np.mean([(count_neighbours(focus, pool, DISTS, wh, sd) > 0).mean(axis=0)
                            for sd in (-SHIFT_DAYS, SHIFT_DAYS)], axis=0) * 100
            res[(wh, name)] = (real, ctrl)
    table = pd.DataFrame({(f"±{wh} h", name, kind): vals[i] for (wh, name), vals in res.items()
                          for i, kind in enumerate(["real", "control"])}, index=[f"{d} km" for d in DISTS]).T
    print("Share of focus fires with ≥ 1 other fire starting within D km and ±T h [%]:")
    print(table.round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colours = {"any size": E.BLUE, "≥ 10 ha": E.ORANGE, "≥ 100 ha": E.AQUA}
    for ax, wh in zip(axes, WINDOWS_H):
        for name, _ in POOLS:
            real, ctrl = res[(wh, name)]
            ax.plot(DISTS, real, color=colours[name], lw=2, marker="o", ms=4, label=name)
            ax.plot(DISTS, ctrl, color=colours[name], lw=1.2, ls=":")
        ax.set_xscale("log"); ax.set_xticks(DISTS, [str(d) for d in DISTS])
        ax.set_xlabel("distance between ignition points D [km]")
        ax.set_title(f"Another fire starts within ±{wh} h")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel(f"fires ≥ {FOCUS_HA} ha with a neighbour [%]")
    axes[0].legend(title="neighbour size (solid)", fontsize=8, loc="lower right")
    axes[1].plot([], [], color=E.INK2, lw=1.2, ls=":", label="control: start shifted ±7 days")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    E.save(fig, "10_neighbouring_fires.png")

    # days with many large fires
    day = focus["start_utc"].dt.floor("D")
    per_day = day.value_counts()
    print(f"\nDays with ≥ 1 fire ≥ {FOCUS_HA} ha: {len(per_day):,}; fires on a day that has ≥ 2 such fires: "
          f"{per_day[per_day >= 2].sum():,} of {len(focus):,} ({per_day[per_day >= 2].sum() / len(focus) * 100:.0f} %)")
    print("Days with most fires ≥ 100 ha:")
    print(per_day.head(8).rename("fires").to_string())
    return focus


def part_b(fires):
    print("\nPart B: hotspots claimed by more than one fire")
    out = {}
    focus = fires[fires["burned_area_ha"] >= FOCUS_HA]
    sev = pd.read_parquet(PROC / "seviri_frp.parquet")
    for name, ha in POOLS[1:]:
        pool = fires[fires["burned_area_ha"] >= ha].reset_index(drop=True)
        det, pos = E.match_pairs(pool, sev, pre_h=1, post_h=24)
        pairs = pd.DataFrame({"det": det, "fire_id": pool["fire_id"].values[pos]})
        n_claims = pairs.groupby("det")["fire_id"].transform("nunique")
        pairs["shared"] = n_claims > 1
        g = pairs[pairs["fire_id"].isin(focus["fire_id"])].groupby("fire_id").agg(
            n=("det", "size"), shared=("shared", "mean"))
        g["shared"] *= 100
        out[name] = g
        print(f"  competitors {name}: {len(g):,} focus fires have detections; "
              f"median shared share {g['shared'].median():.0f} %, "
              f"fires with > 0 %: {(g['shared'] > 0).mean() * 100:.0f} %, > 50 %: {(g['shared'] > 50).mean() * 100:.0f} %, "
              f"100 %: {(g['shared'] >= 99.9).mean() * 100:.0f} %")
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, 105, 5)
    for (name, _), c in zip(POOLS[1:], [E.ORANGE, E.BLUE]):
        ax.hist(out[name]["shared"], bins=bins, histtype="step", lw=2, color=c, label=f"competing fires {name}")
    ax.set_xlabel("share of a fire's matched hotspots that also match another fire [%]")
    ax.set_ylabel(f"fires ≥ {FOCUS_HA} ha")
    ax.set_title("Hotspots shared between fires")
    ax.legend(fontsize=8)
    fig.tight_layout()
    E.save(fig, "11_shared_hotspots.png")


def main():
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    fires = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
                 & icnf["start_utc"].between(pd.Timestamp(E.PERIOD[0], tz="UTC"), pd.Timestamp(E.PERIOD[1], tz="UTC"))].copy()
    fires["end_eff"] = fires["end_utc"].where(fires["end_utc"].notna(), fires["start_utc"])
    fires = fires.reset_index(drop=True)
    part_a(fires)
    part_b(fires)


if __name__ == "__main__":
    main()
