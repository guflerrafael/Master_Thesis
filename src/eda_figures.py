"""Descriptive figures of the ICNF and SEVIRI datasets for the data chapter.

Figures (figures/eda/):
  01_icnf_annual.png              fires and burned area per year, 1980-2023
  02_icnf_size_distribution.png   share of fires / burned area above a size
  03_icnf_ignition_map.png        ignition point density, and fires >= 100 ha
  04_seviri_temporal.png          detections per year / month / hour, FRP per hour
  05_seviri_frp_confidence.png    FRP vs confidence, detections kept per threshold
  06_seviri_pixel_maps.png        detections per pixel, off-season persistence
  07_detection_probability.png    share of ICNF fires seen by SEVIRI, by size
  08_fre_vs_burned_area.png       fire radiative energy vs burned area (per fire, per year)
  09_unmatched_hotspots.png       SEVIRI detections in Portugal without an ICNF fire

ICNF <-> SEVIRI matching used here (exploratory, not the final event matching):
  radius   R(A) = 5 km + 2 * sqrt(A / pi), A = burned area in km2, capped at 40 km
  fig 07   detection in [start - 1 h, end + 3 h]           (end = start if missing)
  fig 08/09 detection in [start - 1 h, end + 24 h]; a detection matching several fires
           is assigned to the largest one

Run from the repository root:  .venv/bin/python src/eda_figures.py
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BORDERS = ROOT / "data" / "external" / "ne_10m_admin_0_countries.zip"
FIG = ROOT / "figures" / "eda"

# Palette (dataviz reference palette, light mode). Categorical slots in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fde3d6", "#f5a47f", "#eb6834", "#b8461a", "#7a2a0b"])

EXTENT = (-9.75, -5.95, 36.3, 43.65)   # lon0, lon1, lat0, lat1 (SEVIRI domain)
EXTENT_PT = (-9.75, -6.0, 36.85, 42.25)  # mainland Portugal
PERIOD = ("2004-01-20", "2019-12-30")  # ICNF fires fully inside the SEVIRI record

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def log(msg):
    print(f"  - {msg}")


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"{name}")


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def radius_km(area_ha):
    return np.minimum(5 + 2 * np.sqrt(np.nan_to_num(area_ha) / 100 / np.pi), 40)


# ---------------------------------------------------------------- maps

def load_borders():
    w = gpd.read_file(BORDERS)
    w = w[w["ADMIN"].isin(["Portugal", "Spain"])]
    parts = w[w["ADMIN"] == "Portugal"].geometry.explode(index_parts=False)
    # mainland = largest part, measured in the Portuguese metric CRS (PT-TM06, EPSG:3763)
    return w, parts.iloc[parts.to_crs(3763).area.argmax()]


def map_axes(ax, borders, title, extent=EXTENT):
    borders.plot(ax=ax, facecolor=NEUTRAL, edgecolor="none", zorder=0)
    borders.boundary.plot(ax=ax, color=MUTED, lw=0.6, zorder=3)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_xlabel("longitude [°E]"); ax.set_ylabel("latitude [°N]")
    ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.grid(False)
    ax.tick_params(labelsize=7)
    ax.set_title(title)
    ax.text(-6.45, 39.75, "SPAIN", color=MUTED, fontsize=8, ha="center", zorder=4, clip_on=True)


def in_polygon(lon, lat, poly):
    return gpd.points_from_xy(lon, lat).within(poly)


# ---------------------------------------------------------------- matching

def match_pairs(fires, sev, pre_h, post_h):
    """All (detection, fire) pairs within R(A) of the ignition point and in
    [start - pre_h, end + post_h]. Returns arrays (det_idx, fire_pos)."""
    t = sev["time_utc"].values.astype("datetime64[ns]").astype("int64")
    lat, lon = sev["lat"].values, sev["lon"].values
    h = 3_600_000_000_000
    t0 = fires["start_utc"].values.astype("datetime64[ns]").astype("int64") - pre_h * h
    t1 = fires["end_eff"].values.astype("datetime64[ns]").astype("int64") + post_h * h
    i0, i1 = np.searchsorted(t, t0), np.searchsorted(t, t1)
    R = radius_km(fires["burned_area_ha"].values)
    flat, flon = fires["lat"].values, fires["lon"].values
    dets, pos = [], []
    for k in range(len(fires)):
        if i1[k] == i0[k]:
            continue
        near = np.nonzero(haversine_km(flat[k], flon[k], lat[i0[k]:i1[k]], lon[i0[k]:i1[k]]) <= R[k])[0]
        if len(near):
            dets.append(near + i0[k]); pos.append(np.full(len(near), k))
    if not dets:
        return np.array([], int), np.array([], int)
    return np.concatenate(dets), np.concatenate(pos)


# ---------------------------------------------------------------- figures

def fig_icnf_annual(icnf):
    y = icnf.groupby(icnf["start_local"].dt.year).agg(n=("fire_id", "size"), area=("burned_area_ha", "sum"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    for ax, col, scale, label in [(a1, "n", 1e3, "fires (thousands)"), (a2, "area", 1e3, "burned area (1000 ha)")]:
        ax.axvspan(2003.5, 2019.5, color=NEUTRAL, zorder=0)
        ax.bar(y.index, y[col] / scale, width=0.8, color=BLUE)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    a1.text(2011.5, a1.get_ylim()[1] * 0.95, "SEVIRI record 2004–2019", ha="center", va="top", color=INK2, fontsize=8)
    for yr in y["area"].nlargest(3).index:
        a2.annotate(str(yr), (yr, y.loc[yr, "area"] / 1e3), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)
    a1.set_title("ICNF fire records per year")
    a2.set_title("ICNF burned area per year")
    fig.tight_layout()
    save(fig, "01_icnf_annual.png")
    log(f"years with most burned area: {', '.join(f'{k} ({v/1e3:,.0f} kha)' for k, v in y['area'].nlargest(3).items())}")


def fig_size_distribution(icnf, fires):
    sets = [("all fires 1980–2023", icnf, BLUE, "-"),
            ("2004–2019 with coordinates", fires, ORANGE, "--")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, d, c, ls in sets:
        a = np.sort(d["burned_area_ha"].dropna().values)
        a = a[a > 0]
        n_above = 1 - np.arange(len(a)) / len(a)
        area_above = np.cumsum(a[::-1])[::-1] / a.sum()
        a1.loglog(a, n_above, color=c, ls=ls, lw=2, label=label)
        a2.semilogx(a, area_above * 100, color=c, ls=ls, lw=2, label=label)
    for th in (1, 10, 100, 1000):
        n = (fires["burned_area_ha"] >= th).sum()
        for ax in (a1, a2):
            ax.axvline(th, color=MUTED, lw=0.8, ls=":")
        a1.text(th * 0.85, 2e-6, f"≥{th:,} ha: {n:,}", fontsize=7, color=INK2, rotation=90, va="bottom", ha="right")
    a1.set_xlabel("burned area x [ha]"); a1.set_ylabel("share of fires with area ≥ x")
    a2.set_xlabel("burned area x [ha]"); a2.set_ylabel("share of total burned area from fires ≥ x [%]")
    a1.set_title("Fire size distribution"); a2.set_title("Burned area is concentrated in few large fires")
    a1.set_ylim(1e-6, 1.5); a2.set_ylim(0, 102)
    for ax in (a1, a2):
        ax.set_xlim(1e-3, 1e5)   # 17,693 fires with placeholder areas < 0.001 ha (down to 1e-12) are off-axis
    a1.text(1.3e-3, 2e-4, "vertical labels: fires\n2004–2019 with coordinates", fontsize=7, color=INK2)
    a1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "02_icnf_size_distribution.png")
    for th in (100, 1000):
        sh_n = (fires["burned_area_ha"] >= th).mean() * 100
        sh_a = fires.loc[fires["burned_area_ha"] >= th, "burned_area_ha"].sum() / fires["burned_area_ha"].sum() * 100
        log(f"2004–2019 with coords: fires ≥ {th} ha are {sh_n:.2f} % of fires and {sh_a:.1f} % of burned area")


def fig_ignition_map(icnf, borders):
    d = icnf[icnf["lat"].notna()]
    big = d[d["burned_area_ha"] >= 100].sort_values("burned_area_ha", ascending=False)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7))
    map_axes(a1, borders, f"All ignition points 2001–2023 (n = {len(d):,})", EXTENT_PT)
    hb = a1.hexbin(d["lon"], d["lat"], gridsize=(70, 95), extent=(-9.6, -6.1, 36.8, 42.2),
                   bins="log", cmap=SEQ_BLUE, mincnt=1, lw=0, zorder=2)
    fig.colorbar(hb, ax=a1, shrink=0.6, label="fires per hexagon (log)")
    map_axes(a2, borders, f"Fires ≥ 100 ha, 2001–2023 (n = {len(big):,})", EXTENT_PT)
    size = lambda a: 3 + a / 150
    a2.scatter(big["lon"], big["lat"], s=size(big["burned_area_ha"]), color=BLUE, alpha=0.55,
               edgecolor="white", lw=0.4, zorder=2)
    for a in (100, 1000, 10000, 50000):
        a2.scatter([], [], s=size(a), color=BLUE, alpha=0.55, edgecolor="white", label=f"{a:,} ha")
    a2.legend(title="burned area", loc="lower right", fontsize=7, title_fontsize=7, labelspacing=1.2)
    fig.tight_layout()
    save(fig, "03_icnf_ignition_map.png")
    north = d["lat"] > 40.5
    log(f"ignitions north of 40.5°N: {north.mean()*100:.0f} % of all, "
        f"{(big['lat'] > 40.5).mean()*100:.0f} % of fires ≥ 100 ha")


def fig_seviri_temporal(sev):
    t = sev["time_utc"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, key, title, xl in [(axes[0, 0], t.dt.year, "Detections per year", "year"),
                               (axes[0, 1], t.dt.month, "Detections per month (2004–2019)", "month"),
                               (axes[1, 0], t.dt.hour, "Detections per hour of day", "hour (UTC)")]:
        c = key.value_counts().sort_index()
        ax.bar(c.index, c.values / 1e3, width=0.8, color=BLUE)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel("detections (thousands)")
        ax.grid(axis="x", visible=False)
    axes[0, 1].set_xticks(range(1, 13))
    q = sev.groupby(t.dt.hour)["frp_mw"].quantile([0.25, 0.5, 0.75]).unstack()
    ax = axes[1, 1]
    ax.fill_between(q.index, q[0.25], q[0.75], color=BLUE, alpha=0.2, lw=0, label="25–75 %")
    ax.plot(q.index, q[0.5], color=BLUE, lw=2, marker="o", ms=4, label="median")
    ax.set_title("FRP per detection by hour of day"); ax.set_xlabel("hour (UTC)"); ax.set_ylabel("FRP [MW]")
    ax.legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_xticks(range(0, 24, 3))
    fig.tight_layout()
    save(fig, "04_seviri_temporal.png")
    jaso = t.dt.month.isin([7, 8, 9, 10]).mean() * 100
    peak = t.dt.hour.value_counts().idxmax()
    log(f"{jaso:.0f} % of detections in Jul–Oct; peak hour {peak}:00 UTC; "
        f"median FRP {q[0.5].min():.0f}–{q[0.5].max():.0f} MW over the day")


def fig_frp_confidence(sev):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    xb = np.logspace(np.log10(8), np.log10(1100), 60)
    h = a1.hist2d(sev["frp_mw"], sev["confidence"], bins=[xb, np.arange(0, 102, 2)],
                  norm=LogNorm(), cmap=SEQ_BLUE, cmin=1)
    a1.set_xscale("log"); a1.grid(False)
    fig.colorbar(h[3], ax=a1, label="detections (log)")
    a1.set_xlabel("FRP [MW]"); a1.set_ylabel("fire confidence [%]")
    a1.set_title("Confidence rises with FRP")
    xb2 = np.logspace(np.log10(8), np.log10(1100), 25)
    mid = np.sqrt(xb2[:-1] * xb2[1:])
    binned = pd.cut(sev["frp_mw"], xb2)
    for th, c, ls in [(50, BLUE, "-"), (70, ORANGE, "--"), (90, AQUA, "-.")]:
        kept = (sev["confidence"] >= th).groupby(binned, observed=False).mean() * 100
        a2.semilogx(mid, kept.values, color=c, ls=ls, lw=2,
                    label=f"confidence ≥ {th} ({(sev['confidence'] >= th).mean()*100:.0f} % of all kept)")
    a2.legend(loc="lower right", fontsize=8)
    a2.set_xlabel("FRP [MW]"); a2.set_ylabel("detections kept [%]")
    a2.set_title("A confidence threshold removes the weak detections")
    a2.set_ylim(-2, 102)
    fig.tight_layout()
    save(fig, "05_seviri_frp_confidence.png")


def pixel_stats(sev):
    """Per SEVIRI pixel: detections, active days, active days outside Jun-Oct, and the
    'persistent off-season' flag (>= 20 active days, >= 50 % of them outside Jun-Oct)."""
    s = sev.assign(day=sev["time_utc"].dt.floor("D"),
                   off=~sev["time_utc"].dt.month.isin([6, 7, 8, 9, 10]))
    days = s.groupby(["lat", "lon"])["day"].nunique()
    off_days = s[s["off"]].groupby(["lat", "lon"])["day"].nunique()
    px = pd.DataFrame({"n": s.groupby(["lat", "lon"]).size(), "days": days,
                       "off_days": off_days}).fillna({"off_days": 0}).reset_index()
    px["off_share"] = px["off_days"] / px["days"]
    px["persistent"] = (px["days"] >= 20) & (px["off_share"] >= 0.5)
    return px


def fig_pixel_maps(sev, borders):
    px = pixel_stats(sev)
    persistent = px[px["persistent"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 7.2))
    map_axes(a1, borders, f"Detections per SEVIRI pixel ({len(px):,} pixels)")
    p = px.sort_values("n")
    sc = a1.scatter(p["lon"], p["lat"], c=p["n"], s=4, marker="s", cmap=SEQ_BLUE, norm=LogNorm(), lw=0, zorder=2)
    fig.colorbar(sc, ax=a1, shrink=0.6, label="detections 2004–2019 (log)")
    map_axes(a2, borders, "Share of active days outside Jun–Oct")
    q = px[px["days"] >= 5].sort_values("off_share")
    sc = a2.scatter(q["lon"], q["lat"], c=q["off_share"] * 100, s=4, marker="s", cmap=SEQ_ORANGE,
                    vmin=0, vmax=100, lw=0, zorder=2)
    a2.scatter(persistent["lon"], persistent["lat"], s=40, facecolor="none", edgecolor=INK, lw=0.8, zorder=5,
               label=f"≥ 20 active days, ≥ 50 % off-season (n = {len(persistent)})")
    a2.legend(loc="lower right", fontsize=7)
    fig.colorbar(sc, ax=a2, shrink=0.6, label="off-season days [% of active days] (pixels with ≥ 5 days)")
    fig.tight_layout()
    save(fig, "06_seviri_pixel_maps.png")
    log(f"{len(persistent)} persistent off-season pixels "
        f"({persistent['n'].sum():,} detections); top 5 by off-season days:")
    print(persistent.sort_values("off_days", ascending=False).head(5)
          [["lat", "lon", "n", "days", "off_days"]].to_string(index=False))
    return persistent


def fig_detection_probability(fires, sev):
    bins = [0.01, 0.1, 1, 5, 10, 50, 100, 500, 1000, 5000, 1e6]
    labels = ["0.01–0.1", "0.1–1", "1–5", "5–10", "10–50", "50–100", "100–500", "500–1k", "1k–5k", "≥5k"]
    f = fires[fires["burned_area_ha"] >= 0.01].reset_index(drop=True)
    f["size_bin"] = pd.cut(f["burned_area_ha"], bins, labels=labels, right=False)

    def detected(shift_days):
        g = f.copy()
        g["start_utc"] += pd.Timedelta(days=shift_days); g["end_eff"] += pd.Timedelta(days=shift_days)
        _, pos = match_pairs(g, sev, pre_h=1, post_h=3)
        hit = np.zeros(len(g), bool); hit[pos] = True
        return hit

    f["hit"] = detected(0)
    f["bg"] = (detected(-7).astype(int) + detected(7)) / 2   # same place and hour, a week before / after
    r = f.groupby("size_bin", observed=False).agg(n=("hit", "size"), k=("hit", "sum"), bg=("bg", "mean"))
    p = r["k"] / r["n"]
    z = 1.96   # Wilson interval
    centre = (p + z**2 / (2 * r["n"])) / (1 + z**2 / r["n"])
    half = z * np.sqrt(p * (1 - p) / r["n"] + z**2 / (4 * r["n"]**2)) / (1 + z**2 / r["n"])

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(r))
    ax.errorbar(x, p * 100, yerr=[(p - (centre - half)) * 100, ((centre + half) - p) * 100],
                fmt="o", color=BLUE, ms=7, lw=1.5, capsize=3)
    ax.plot(x, p * 100, color=BLUE, lw=2)
    ax.plot(x, r["bg"] * 100, color=ORANGE, lw=2, ls="--", marker="s", ms=5)
    ax.text(x[-1] + 0.15, p.iloc[-1] * 100, "detected", color=INK2, fontsize=8, va="center")
    ax.text(x[-1] + 0.15, r["bg"].iloc[-1] * 100, "chance match\n(same place, ±7 days)", color=INK2, fontsize=8, va="center")
    ax.set_xticks(x, [f"{l}\nn={n:,}" for l, n in zip(labels, r["n"])], fontsize=7)
    ax.set_xlim(-0.5, len(r) + 1.3)
    ax.set_xlabel("burned area [ha]"); ax.set_ylabel("fires with a SEVIRI detection [%]")
    ax.set_title("Share of ICNF fires (2004–2019) detected by SEVIRI, by size")
    ax.set_ylim(0, 105); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "07_detection_probability.png")
    print((pd.DataFrame({"n": r["n"], "detected_%": (p * 100).round(1), "chance_%": (r["bg"] * 100).round(1)})).to_string())


def fig_fre_vs_area(fires, sev, assigned, mainland):
    # FRE per detection: FRP [MW] * 900 s = MJ; summed and given in TJ
    fre = (sev["frp_mw"].values.astype(float) * 900 / 1e6)
    big = fires[fires["burned_area_ha"] >= 100].copy()
    per_fire = pd.Series(fre[assigned.index], index=assigned.values).groupby(level=0).sum()
    big["fre_tj"] = per_fire.reindex(big["fire_id"]).values
    seen = big[big["fre_tj"] > 0]

    sev_pt = in_polygon(sev["lon"], sev["lat"], mainland)
    y_fre = pd.Series(fre[sev_pt]).groupby(sev["time_utc"].dt.year.values[sev_pt]).sum()
    y_area = fires.groupby(fires["start_utc"].dt.year)["burned_area_ha"].sum()
    yy = pd.DataFrame({"fre": y_fre, "area": y_area}).dropna()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3))
    lx, ly = np.log10(seen["burned_area_ha"]), np.log10(seen["fre_tj"])
    b, a = np.polyfit(lx, ly, 1)
    r = np.corrcoef(lx, ly)[0, 1]
    a1.scatter(seen["burned_area_ha"], seen["fre_tj"], s=10, color=BLUE, alpha=0.45, lw=0)
    xs = np.logspace(2, np.log10(seen["burned_area_ha"].max()), 50)
    a1.plot(xs, 10**a * xs**b, color=INK2, lw=1.5)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.text(0.03, 0.95, f"n = {len(seen):,} fires ≥ 100 ha\nlog-log r = {r:.2f}\nFRE ∝ area$^{{{b:.2f}}}$",
            transform=a1.transAxes, va="top", fontsize=8, color=INK2)
    a1.set_xlabel("ICNF burned area [ha]"); a1.set_ylabel("SEVIRI fire radiative energy [TJ]")
    a1.set_title("Per fire (≥ 100 ha, 2004–2019)")
    ry = np.corrcoef(np.log10(yy["area"]), np.log10(yy["fre"]))[0, 1]
    a2.scatter(yy["area"] / 1e3, yy["fre"], s=40, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    for yr, row in yy.iterrows():
        a2.annotate(str(yr), (row["area"] / 1e3, row["fre"]), xytext=(4, -9 if yr == 2016 else 2), textcoords="offset points",
                    fontsize=7, color=INK2)
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.text(0.03, 0.95, f"log-log r = {ry:.2f}", transform=a2.transAxes, va="top", fontsize=8, color=INK2)
    a2.set_xlabel("ICNF burned area [1000 ha]"); a2.set_ylabel("SEVIRI FRE in mainland Portugal [TJ]")
    a2.set_title("Per year")
    fig.tight_layout()
    save(fig, "08_fre_vs_burned_area.png")
    log(f"fires ≥ 100 ha with assigned detections: {len(seen):,} of {len(big):,}; "
        f"per-fire log-log r = {r:.2f}, slope {b:.2f}; per-year r = {ry:.2f}")


def fig_unmatched(sev, matched, mainland, borders, persistent):
    pt = in_polygon(sev["lon"], sev["lat"], mainland)
    s = sev[pt].assign(matched=matched[pt])
    un = s[~s["matched"]]
    fig = plt.figure(figsize=(11, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15])
    a1 = fig.add_subplot(gs[:, 0])
    map_axes(a1, borders, f"Unmatched detections in mainland Portugal (n = {len(un):,})", EXTENT_PT)
    c = un.groupby(["lat", "lon"]).size().sort_values().reset_index(name="n")
    sc = a1.scatter(c["lon"], c["lat"], c=c["n"], s=5, marker="s", cmap=SEQ_BLUE, norm=LogNorm(), lw=0, zorder=2)
    a1.scatter(persistent["lon"], persistent["lat"], s=40, facecolor="none", edgecolor=INK, lw=0.8, zorder=5,
               label="persistent off-season pixel (fig. 06)")
    a1.legend(loc="lower right", fontsize=7)
    fig.colorbar(sc, ax=a1, shrink=0.5, label="unmatched detections per pixel (log)")

    a2 = fig.add_subplot(gs[0, 1])
    cb = pd.cut(s["confidence"], [0, 30, 50, 60, 70, 80, 90, 101], right=False)
    sh = (~s["matched"]).groupby(cb, observed=False).mean() * 100
    a2.bar(range(len(sh)), sh.values, color=BLUE, width=0.75)
    a2.set_xticks(range(len(sh)), ["<30", "30–49", "50–59", "60–69", "70–79", "80–89", "90–100"], fontsize=7)
    a2.set_xlabel("fire confidence [%]"); a2.set_ylabel("unmatched [%]")
    a2.set_title("Unmatched share by confidence"); a2.grid(axis="x", visible=False)

    a3 = fig.add_subplot(gs[1, 1])
    m = s["time_utc"].dt.month
    sh = (~s["matched"]).groupby(m).mean() * 100
    a3.bar(sh.index, sh.values, color=BLUE, width=0.75)
    a3.set_xticks(range(1, 13)); a3.set_xlabel("month"); a3.set_ylabel("unmatched [%]")
    a3.set_title("Unmatched share by month"); a3.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "09_unmatched_hotspots.png")
    log(f"detections in mainland Portugal: {len(s):,} of {len(sev):,} ({len(s)/len(sev)*100:.0f} %); "
        f"unmatched: {len(un):,} ({len(un)/len(s)*100:.1f} %)")
    log(f"unmatched share: confidence < 50: {(~s.loc[s['confidence'] < 50, 'matched']).mean()*100:.0f} %, "
        f"≥ 80: {(~s.loc[s['confidence'] >= 80, 'matched']).mean()*100:.0f} %; "
        f"Jul–Sep: {(~s.loc[m.isin([7, 8, 9]), 'matched']).mean()*100:.0f} %, "
        f"Nov–Apr: {(~s.loc[m.isin([11, 12, 1, 2, 3, 4]), 'matched']).mean()*100:.0f} %")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    sev = pd.read_parquet(PROC / "seviri_frp.parquet")
    borders, mainland = load_borders()

    fires = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
                 & icnf["start_utc"].between(pd.Timestamp(PERIOD[0], tz="UTC"), pd.Timestamp(PERIOD[1], tz="UTC"))].copy()
    fires["end_eff"] = fires["end_utc"].where(fires["end_utc"].notna(), fires["start_utc"])
    fires = fires.reset_index(drop=True)

    fig_icnf_annual(icnf)
    fig_size_distribution(icnf, fires)
    fig_ignition_map(icnf, borders)
    fig_seviri_temporal(sev)
    fig_frp_confidence(sev)
    persistent = fig_pixel_maps(sev, borders)
    fig_detection_probability(fires, sev)

    # generous matching for figs 08/09: each detection goes to the largest candidate fire
    det, pos = match_pairs(fires, sev, pre_h=1, post_h=24)
    pairs = pd.DataFrame({"det": det, "area": fires["burned_area_ha"].values[pos], "fire_id": fires["fire_id"].values[pos]})
    assigned = pairs.sort_values("area").drop_duplicates("det", keep="last").set_index("det")["fire_id"]
    matched = np.zeros(len(sev), bool); matched[assigned.index] = True
    fig_fre_vs_area(fires, sev, assigned, mainland)
    fig_unmatched(sev, matched, mainland, borders, persistent)


if __name__ == "__main__":
    main()
