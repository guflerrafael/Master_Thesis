"""Plot the largest ICNF fires against SEVIRI hotspots in lat-lon-time.

For the N_FIRES largest fires (2004-2019, valid coordinates and UTC start) all SEVIRI
detections within RADIUS_KM of the ignition point and in [start - PRE_H, end + POST_H]
are shown in two panels per fire:
  left:  map (lon/lat), colour = hours since ICNF start, size = FRP, star = ignition point
  right: distance to ignition point vs. hours since start, size = FRP
Other fires nearby are not removed on purpose: they show how clean the match is.

Run from the repository root:  .venv/bin/python src/plot_largest_fires.py
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

N_FIRES = 12
RADIUS_KM = 40
PRE_H, POST_H = 6, 6
MIN_SPAN_H = 48   # show at least this long after start (end times are often too early)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    sev = pd.read_parquet(PROC / "seviri_frp.parquet")
    t = sev["time_utc"].values

    fires = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
                 & icnf["start_utc"].dt.year.between(2004, 2019)]
    fires = fires.nlargest(N_FIRES, "burned_area_ha")

    ncol = 4
    fig, axes = plt.subplots(N_FIRES // 2, ncol, figsize=(17, 3.6 * N_FIRES // 2))
    rows = []
    for k, f in enumerate(fires.itertuples()):
        ax_map, ax_t = axes.flat[2 * k], axes.flat[2 * k + 1]
        start = f.start_utc
        end = f.end_utc if pd.notna(f.end_utc) else start
        t1 = max(end + pd.Timedelta(hours=POST_H), start + pd.Timedelta(hours=MIN_SPAN_H))
        i0, i1 = np.searchsorted(t, [np.datetime64(start.tz_localize(None) - pd.Timedelta(hours=PRE_H)),
                                     np.datetime64(t1.tz_localize(None))])
        d = sev.iloc[i0:i1].copy()
        d["dist_km"] = haversine_km(f.lat, f.lon, d["lat"].values, d["lon"].values)
        d = d[d["dist_km"] <= RADIUS_KM]
        d["h"] = (d["time_utc"] - start).dt.total_seconds() / 3600
        size = 4 + d["frp_mw"] / 25

        sc = ax_map.scatter(d["lon"], d["lat"], c=d["h"], s=size, cmap="viridis",
                            vmin=-PRE_H, vmax=d["h"].max() if len(d) else 1, alpha=0.8, lw=0)
        ax_map.plot(f.lon, f.lat, marker="*", ms=16, mfc="red", mec="k")
        dlat = RADIUS_KM / 111
        ax_map.set_xlim(f.lon - dlat / np.cos(np.radians(f.lat)), f.lon + dlat / np.cos(np.radians(f.lat)))
        ax_map.set_ylim(f.lat - dlat, f.lat + dlat)
        ax_map.set_aspect(1 / np.cos(np.radians(f.lat)))
        ax_map.set_title(f"{f.start_local:%Y-%m-%d} {f.municipality.title()} – {f.burned_area_ha:,.0f} ha",
                         fontsize=10)
        ax_map.tick_params(labelsize=7)
        fig.colorbar(sc, ax=ax_map, shrink=0.8).ax.tick_params(labelsize=7)

        ax_t.scatter(d["h"], d["dist_km"], s=size, c=d["h"], cmap="viridis",
                     vmin=-PRE_H, vmax=d["h"].max() if len(d) else 1, alpha=0.6, lw=0)
        ax_t.axvline(0, color="red", lw=1, label="ICNF start")
        if pd.notna(f.end_utc):
            ax_t.axvline((f.end_utc - start).total_seconds() / 3600, color="red", ls="--", lw=1, label="ICNF end")
        ax_t.set_xlabel("hours since ICNF start (UTC)", fontsize=8)
        ax_t.set_ylabel("distance to ignition [km]", fontsize=8)
        ax_t.set_ylim(0, RADIUS_KM)
        ax_t.tick_params(labelsize=7)
        if k == 0:
            ax_t.legend(fontsize=7)

        near = d[d["dist_km"] <= 10]
        first = near.iloc[0] if len(near) else None
        rows.append({
            "date": f"{f.start_local:%Y-%m-%d}", "municipality": f.municipality.title(),
            "area_ha": round(f.burned_area_ha), "duration_h": round(f.duration_h, 1) if pd.notna(f.duration_h) else None,
            "n_det_40km": len(d), "n_det_10km": len(near),
            "first_det_10km_h": round(first["h"], 2) if first is not None else None,
            "first_det_dist_km": round(first["dist_km"], 1) if first is not None else None,
            "det_before_start_10km": int((near["h"] < 0).sum()),
        })

    fig.suptitle(f"{N_FIRES} largest ICNF fires 2004–2019: SEVIRI hotspots within {RADIUS_KM} km "
                 f"(colour = hours since ICNF start, size = FRP, ★ = ICNF ignition point)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "largest_fires_lat_lon_time.png", dpi=110)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(f"figure: {(FIG / 'largest_fires_lat_lon_time.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
