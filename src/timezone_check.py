"""Check the time zone of ICNF start times against SEVIRI (UTC) detections.

For each ICNF fire (2004-2019, valid coordinates, >= MIN_AREA ha) find the first SEVIRI
detection within RADIUS_KM of the ignition point in [start - PRE_H, start + POST_H],
treating start_local naively as UTC. lag = first detection - start.

If ICNF is Portuguese local time, the true UTC start is start - 1 h in summer (DST) and
start - 0 h in winter, so the naive lag should be ~1 h smaller in summer than in winter.
If ICNF is UTC, the lag should not depend on DST.

Fires with earlier nearby activity (detections within ISO_KM in the ISO_H hours before the
window) are excluded, so the first detection is likely to belong to the fire itself.

Run from the repository root:  .venv/bin/python src/timezone_check.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

MIN_AREA = 10      # ha
RADIUS_KM = 5
PRE_H, POST_H = 6, 24
ISO_KM, ISO_H = 10, 48


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def first_detections(fires, sev):
    t = sev["time_utc"].dt.tz_localize(None).values
    lat, lon = sev["lat"].values, sev["lon"].values
    rows = []
    for f in fires.itertuples():
        start = np.datetime64(f.start_local)
        # isolation: no nearby detections before the window
        i0, i1 = np.searchsorted(t, [start - np.timedelta64(ISO_H + PRE_H, "h"), start - np.timedelta64(PRE_H, "h")])
        if i1 > i0 and (haversine_km(f.lat, f.lon, lat[i0:i1], lon[i0:i1]) <= ISO_KM).any():
            continue
        j0, j1 = np.searchsorted(t, [start - np.timedelta64(PRE_H, "h"), start + np.timedelta64(POST_H, "h")])
        near = np.nonzero(haversine_km(f.lat, f.lon, lat[j0:j1], lon[j0:j1]) <= RADIUS_KM)[0]
        if len(near):
            rows.append((f.fire_id, (t[j0 + near[0]] - start) / np.timedelta64(1, "m") / 60))
    return pd.DataFrame(rows, columns=["fire_id", "lag_h"])


def main():
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    sev = pd.read_parquet(PROC / "seviri_frp.parquet")

    fires = icnf[icnf["lat"].notna() & icnf["start_local"].dt.year.between(2004, 2019)
                 & (icnf["burned_area_ha"] >= MIN_AREA)].copy()
    # UTC offset Lisbon would have at that wall-clock time (0 winter, 1 summer)
    loc = fires["start_local"].dt.tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT")
    fires["dst"] = (loc.dt.tz_localize(None) - loc.dt.tz_convert("UTC").dt.tz_localize(None)) == pd.Timedelta("1h")
    fires = fires[loc.notna()]

    res = first_detections(fires, sev).merge(fires, on="fire_id")
    print(f"fires >= {MIN_AREA} ha: {len(fires):,}; matched (isolated, detected): {len(res):,}")

    def summary(d, label):
        med = d["lag_h"].median()
        boot = [d["lag_h"].sample(frac=1, replace=True, random_state=k).median() for k in range(500)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:<34} n={len(d):5d}  median lag {med:+.2f} h  (95% CI {lo:+.2f} .. {hi:+.2f})")
        return med

    print("Lag = first SEVIRI detection (UTC) - ICNF start (naive):")
    for area in (MIN_AREA, 100):
        d = res[res["burned_area_ha"] >= area]
        s = summary(d[d["dst"]], f">= {area} ha, summer (DST)")
        w = summary(d[~d["dst"]], f">= {area} ha, winter")
        print(f"  -> winter - summer: {w - s:+.2f} h")

    # Same comparison restricted to +-6 weeks around each DST transition (similar season)
    m = res["start_local"].dt.month
    near = res[m.isin([3, 4, 10, 11])]
    print("Months around the DST transitions only (Mar/Apr, Oct/Nov):")
    s = summary(near[near["dst"]], "summer side")
    w = summary(near[~near["dst"]], "winter side")
    print(f"  -> winter - summer: {w - s:+.2f} h")

    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(-PRE_H, 12.01, 0.25)
    for flag, label, c in [(True, "summer (DST, UTC+1)", "tab:orange"), (False, "winter (UTC+0)", "tab:blue")]:
        ax.hist(res.loc[res["dst"] == flag, "lag_h"], bins=bins, density=True, histtype="step", lw=1.8, color=c, label=label)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("first SEVIRI detection (UTC) − ICNF start (as recorded) [h]")
    ax.set_ylabel("density")
    ax.set_title(f"ICNF fires ≥ {MIN_AREA} ha, 2004–2019, first detection within {RADIUS_KM} km")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "timezone_lag.png", dpi=150)
    print(f"figure: {(FIG / 'timezone_lag.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
