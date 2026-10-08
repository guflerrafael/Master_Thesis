"""Group SEVIRI hotspot detections into fire events (satellite data only, no ICNF).

Rule: two detections belong to the same event if their pixel centres are at most D_KM apart
and their times at most GAP_H apart; events are the connected groups of this link graph
(single-linkage in space and time). D_KM = 6 km links a pixel to its 8 neighbours (pixel
spacing ~3.2 km E-W, ~4.4 km N-S); see notes.md 3.1 for how D_KM and GAP_H were chosen.

Nuclei: within an event, every group of detections that appears with no earlier detection
within D_KM and GAP_H starts a new nucleus (its "birth"), even if it later merges with other
nuclei. Later detections inherit the nucleus of the nearest linked earlier detection (or, in
the same time step, of the nearest neighbour that already has one). Nuclei are the candidate
ignitions; an isolated fire is one event with one nucleus, merged fires keep separate nuclei.

ICNF is used only in the diagnostics (how many large fires end up in a shared event or
nucleus), never to form events or nuclei, so they can later define the model input without
label leakage.

Outputs:
  data/processed/seviri_frp_events.parquet   detections + event_id + nucleus_id + flags
  data/processed/seviri_events.parquet       one row per event
  data/processed/seviri_nuclei.parquet       one row per nucleus
  figures/clustering/                        case study 2017-10-15, nuclei vs events

Run from the repository root:
  .venv/bin/python src/cluster_hotspots.py > outputs/cluster_hotspots_log.txt
  .venv/bin/python src/cluster_hotspots.py --sensitivity     (parameter grid, slower)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial import QhullError

import eda_figures as E   # shared helpers and palette
from proximity_analysis import count_neighbours

PROC = E.PROC
FIG = E.ROOT / "figures" / "clustering"

D_KM = 6
GAP_H = 6
CASE_DAY = "2017-10-15"
MIN_NUCLEUS_DET = 3        # nuclei drawn in the case figure
FOCUS_HA = 100
ISOLATION = (10, 20, 72)   # isolated fire: no other fire >= 10 ha within 20 km and +-72 h


# ---------------------------------------------------------------- clustering

def xy_km(lat, lon):
    """Local metric coordinates (good enough for distances of a few km)."""
    return np.c_[lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57]


def cluster(sev, d_km=D_KM, gap_h=GAP_H, nuclei=False):
    """Event label for every detection of `sev` (must be sorted by time); with nuclei=True
    also the nucleus label (ids in order of birth)."""
    px = sev[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    pix = (pd.Series(np.arange(len(px)), index=pd.MultiIndex.from_frame(px))
           .reindex(pd.MultiIndex.from_frame(sev[["lat", "lon"]])).values)
    P = xy_km(px["lat"].values, px["lon"].values)
    nb = cKDTree(P).query_ball_point(P, r=d_km)   # includes the pixel itself
    # (neighbour pixel, distance) pairs
    nbrs = [list(zip(q, np.hypot(*(P[q] - P[p]).T))) for p, q in enumerate(map(np.asarray, nb))]

    t = sev["time_utc"].values.astype("datetime64[m]").astype(np.int64)
    gap = int(gap_h * 60)
    parent = list(range(len(sev)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    nuc = [-1] * len(sev)
    n_nuc = 0
    last_det = np.full(len(px), -1)
    last_t = np.full(len(px), -10**12, dtype=np.int64)
    starts = np.r_[0, np.nonzero(np.diff(t))[0] + 1, len(t)]
    for a, b in zip(starts[:-1], starts[1:]):
        ii, tt = range(a, b), t[a]
        current = {pix[i]: i for i in ii}
        for i in ii:
            best, best_d = -1, np.inf
            for q, dq in nbrs[pix[i]]:
                if tt - last_t[q] <= gap:          # earlier detection nearby, recent enough
                    union(i, last_det[q])
                    if dq < best_d:
                        best, best_d = last_det[q], dq
                j = current.get(q)                  # same time step
                if j is not None and j != i:
                    union(i, j)
            if nuclei and best >= 0:
                nuc[i] = nuc[best]                  # inherit from the nearest earlier detection
        if nuclei:
            pending = [i for i in ii if nuc[i] < 0]
            while pending:                          # same step: grow from assigned neighbours
                rest = []
                for i in pending:
                    best, best_d = -1, np.inf
                    for q, dq in nbrs[pix[i]]:
                        j = current.get(q)
                        if j is not None and nuc[j] >= 0 and dq < best_d:
                            best, best_d = j, dq
                    if best >= 0:
                        nuc[i] = nuc[best]
                    else:
                        rest.append(i)
                if len(rest) == len(pending):
                    break
                pending = rest
            for i in pending:                       # births: adjacent new detections together
                if nuc[i] >= 0:
                    continue
                nuc[i], stack = n_nuc, [i]
                while stack:
                    k = stack.pop()
                    for q, _ in nbrs[pix[k]]:
                        j = current.get(q)
                        if j is not None and nuc[j] < 0:
                            nuc[j] = n_nuc
                            stack.append(j)
                n_nuc += 1
        for i in ii:
            last_det[pix[i]], last_t[pix[i]] = i, tt
    roots = np.array([find(i) for i in range(len(sev))])
    # ids 0..K-1 in order of first detection (roots are the earliest index of each event)
    events = pd.factorize(roots, sort=True)[0]
    return (events, np.array(nuc)) if nuclei else events


def build_events(sev):
    """One row per event. `sev` has event_id, persistent and in_pt columns."""
    s = sev.assign(fre_tj=sev["frp_mw"].astype(float) * 900 / 1e6,
                   wlat=sev["lat"] * sev["frp_mw"], wlon=sev["lon"] * sev["frp_mw"])
    g = s.groupby("event_id")
    ev = pd.DataFrame({
        "t_first": g["time_utc"].min(), "t_last": g["time_utc"].max(),
        "n_det": g.size(), "n_steps": g["time_utc"].nunique(),
        "n_pixels": g.apply(lambda d: len(d[["lat", "lon"]].drop_duplicates()), include_groups=False),
        "fre_tj": g["fre_tj"].sum(), "frp_max": g["frp_mw"].max(),
        "conf_median": g["confidence"].median(),
        "lat_c": g["wlat"].sum() / g["frp_mw"].sum(), "lon_c": g["wlon"].sum() / g["frp_mw"].sum(),
        "share_persistent": g["persistent"].mean(), "share_in_pt": g["in_pt"].mean(),
    })
    first = s[s["time_utc"] == s["event_id"].map(ev["t_first"])].groupby("event_id")
    ev["lat_first"] = first["wlat"].sum() / first["frp_mw"].sum()   # FRP-weighted centroid of
    ev["lon_first"] = first["wlon"].sum() / first["frp_mw"].sum()   # the first time step
    ev["duration_h"] = (ev["t_last"] - ev["t_first"]).dt.total_seconds() / 3600
    if "nucleus_id" in sev:
        ev["n_nuclei"] = g["nucleus_id"].nunique()
    return ev.reset_index()


def build_nuclei(sev, d_km=D_KM, gap_h=GAP_H, look_back_h=48):
    """One row per nucleus, with its birth (FRP-weighted centroid of the birth time step) and
    context: distance to the nearest detection of another nucleus active in the last gap_h
    hours (searched up to 100 km), and whether the birth place had detections within d_km
    in [birth - look_back_h, birth - gap_h) (a fire reappearing after a gap, not a new one)."""
    s = sev.assign(fre_tj=sev["frp_mw"].astype(float) * 900 / 1e6,
                   wlat=sev["lat"] * sev["frp_mw"], wlon=sev["lon"] * sev["frp_mw"])
    g = s.groupby("nucleus_id")
    nu = pd.DataFrame({
        "event_id": g["event_id"].first(), "t_birth": g["time_utc"].min(), "t_last": g["time_utc"].max(),
        "n_det": g.size(), "n_pixels": g.apply(lambda d: len(d[["lat", "lon"]].drop_duplicates()), include_groups=False),
        "fre_tj": g["fre_tj"].sum(), "share_persistent": g["persistent"].mean(), "share_in_pt": g["in_pt"].mean(),
    })
    birth = s[s["time_utc"] == s["nucleus_id"].map(nu["t_birth"])].groupby("nucleus_id")
    nu["lat_birth"] = birth["wlat"].sum() / birth["frp_mw"].sum()
    nu["lon_birth"] = birth["wlon"].sum() / birth["frp_mw"].sum()
    nu["n_det_birth"] = birth.size()
    nu["duration_h"] = (nu["t_last"] - nu["t_birth"]).dt.total_seconds() / 3600
    nu = nu.reset_index()
    nu["order_in_event"] = nu.groupby("event_id").cumcount() + 1   # ids are in birth order

    t = s["time_utc"].values.astype("datetime64[m]").astype(np.int64)
    lat, lon, nid = s["lat"].values, s["lon"].values, s["nucleus_id"].values
    tb = nu["t_birth"].values.astype("datetime64[m]").astype(np.int64)
    i0, i1 = np.searchsorted(t, tb - look_back_h * 60), np.searchsorted(t, tb, side="right")
    dist_active, reappeared = np.full(len(nu), np.nan), np.zeros(len(nu), bool)
    for k in range(len(nu)):
        sl = slice(i0[k], i1[k])
        other = nid[sl] != nu["nucleus_id"].iat[k]
        d = E.haversine_km(nu["lat_birth"].iat[k], nu["lon_birth"].iat[k], lat[sl], lon[sl])
        recent = other & (t[sl] >= tb[k] - gap_h * 60)
        if recent.any() and d[recent].min() <= 100:
            dist_active[k] = d[recent].min()
        reappeared[k] = ((t[sl] < tb[k] - gap_h * 60) & (d <= d_km)).any()
    nu["dist_active_km"], nu["reappeared"] = dist_active, reappeared
    return nu


def load():
    sev = pd.read_parquet(PROC / "seviri_frp.parquet").sort_values(["time_utc", "lat", "lon"]).reset_index(drop=True)
    icnf = pd.read_parquet(PROC / "icnf_fires.parquet")
    pool = icnf[icnf["lat"].notna() & icnf["start_utc"].notna()
                & icnf["start_utc"].between(pd.Timestamp(E.PERIOD[0], tz="UTC"), pd.Timestamp(E.PERIOD[1], tz="UTC"))]
    fires = pool[pool["burned_area_ha"] >= FOCUS_HA].reset_index(drop=True)
    fires["end_eff"] = fires["end_utc"].where(fires["end_utc"].notna(), fires["start_utc"])
    ha, km, h = ISOLATION
    fires["isolated"] = count_neighbours(fires, pool[pool["burned_area_ha"] >= ha], [km], h)[:, 0] == 0
    return sev, fires


# ---------------------------------------------------------------- diagnostics (use ICNF)

def fire_group(fires, sev, col="event_id", link_km=5, pre_h=1, post_h=12):
    """Event (or nucleus, col="nucleus_id") of each ICNF fire: that of its earliest detection
    within link_km of the ignition point in [start - pre_h, start + post_h]; -1 if none.
    Diagnostic only."""
    t = sev["time_utc"].values.astype("datetime64[ns]").astype("int64")
    h = 3_600_000_000_000
    s0 = fires["start_utc"].values.astype("datetime64[ns]").astype("int64")
    i0, i1 = np.searchsorted(t, s0 - pre_h * h), np.searchsorted(t, s0 + post_h * h)
    lat, lon, eid = sev["lat"].values, sev["lon"].values, sev[col].values
    out = np.full(len(fires), -1)
    for k in range(len(fires)):
        if i1[k] > i0[k]:
            near = np.nonzero(E.haversine_km(fires["lat"].iat[k], fires["lon"].iat[k],
                                             lat[i0[k]:i1[k]], lon[i0[k]:i1[k]]) <= link_km)[0]
            if len(near):
                out[k] = eid[i0[k] + near[0]]
    return out


def diagnostics(sev, fires, col="event_id"):
    unit = col.split("_")[0]
    units = {"event": "events", "nucleus": "nuclei"}[unit]
    ev = fire_group(fires, sev, col)
    found = ev >= 0
    shared = pd.Series(ev[found]).map(pd.Series(ev[found]).value_counts()) > 1
    size = sev.groupby(col).size()
    # fragmentation, isolated fires only: share of the hotspots matched to the fire
    # (radius R(A), [start - 1 h, end + 24 h], as in eda_figures) that lie in its own group
    iso = fires["isolated"].values & found
    det, pos = E.match_pairs(fires[iso].reset_index(drop=True), sev, pre_h=1, post_h=24)
    own = sev[col].values[det] == ev[iso][pos]
    cov = pd.Series(own).groupby(pos).mean()
    return {
        f"{units}": len(size), f"{units} ≥ 10 det": int((size >= 10).sum()),
        f"det in {units} ≥ 10 [%]": round(size[size >= 10].sum() / len(sev) * 100, 1),
        f"largest {unit} [det]": int(size.max()),
        f"fires ≥100 ha with {unit} [%]": round(found.mean() * 100, 1),
        f"… sharing {unit} with another fire [%]": round(shared.mean() * 100, 1),
        f"isolated: median own-{unit} share [%]": round(cov.median() * 100, 1),
        f"isolated: own-{unit} share ≥ 90 % [% fires]": round((cov >= 0.9).mean() * 100, 1),
    }


def fire_positions(fires, sev, events, nuclei):
    """Every ICNF fire with an event: its position in the event (alone / first / later, by
    ICNF start), and the distance and lag from its ignition to the event start (centroid of
    the event's first time step) and to the birth of its nucleus. Diagnostic only."""
    f = fires.assign(event_id=fire_group(fires, sev, "event_id"), nucleus_id=fire_group(fires, sev, "nucleus_id"))
    f = f[f["event_id"] >= 0].copy()
    n_in = f["event_id"].map(f["event_id"].value_counts())
    first = f.groupby("event_id")["start_utc"].rank(method="first") == 1
    f["position"] = np.where(n_in == 1, "alone in event", np.where(first, "first in shared event", "later in shared event"))
    e = events.set_index("event_id").loc[f["event_id"]].set_index(f.index)
    n = nuclei.set_index("nucleus_id").loc[f["nucleus_id"]].set_index(f.index)
    f["dist_event_km"] = E.haversine_km(f["lat"], f["lon"], e["lat_first"], e["lon_first"])
    f["lag_event_h"] = (e["t_first"] - f["start_utc"]).dt.total_seconds() / 3600
    f["dist_nucleus_km"] = E.haversine_km(f["lat"], f["lon"], n["lat_birth"], n["lon_birth"])
    f["lag_nucleus_h"] = (n["t_birth"] - f["start_utc"]).dt.total_seconds() / 3600
    f["nucleus_shared"] = f["nucleus_id"].map(f["nucleus_id"].value_counts()) > 1
    f["nucleus_reappeared"] = n["reappeared"].values
    return f


def position_summary(f):
    g = f.groupby("position")
    return pd.DataFrame({
        "fires": g.size(),
        "dist event [km]": g["dist_event_km"].median(), "dist nucleus [km]": g["dist_nucleus_km"].median(),
        "≤ 5 km event [%]": g["dist_event_km"].apply(lambda x: (x <= 5).mean() * 100),
        "≤ 5 km nucleus [%]": g["dist_nucleus_km"].apply(lambda x: (x <= 5).mean() * 100),
        "lag event [h]": g["lag_event_h"].median(), "lag nucleus [h]": g["lag_nucleus_h"].median(),
        "nucleus shared [%]": g["nucleus_shared"].mean() * 100,
    }).round(1)


def sensitivity(sev, fires):
    rows = []
    for d in (4.5, 6, 9):
        for g in (3, 6, 12, 24):
            s = sev.assign(event_id=cluster(sev, d, g))
            rows.append({"D [km]": d, "gap [h]": g, **diagnostics(s, fires)})
            print(rows[-1], flush=True)
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- case study figures

def hull(ax, lon, lat, **kw):
    pts = np.c_[lon, lat]
    if len(np.unique(pts, axis=0)) < 3:
        return
    try:
        h = ConvexHull(pts)
    except QhullError:
        return
    v = np.r_[h.vertices, h.vertices[0]]
    ax.plot(pts[v, 0], pts[v, 1], **kw)


def fig_case(sev, events, nuclei, fires, borders, day=CASE_DAY):
    d0 = pd.Timestamp(day, tz="UTC")
    day_fires = fires[fires["start_utc"].between(d0, d0 + pd.Timedelta(days=1))]
    lat0, lat1 = day_fires["lat"].min() - 0.3, day_fires["lat"].max() + 0.3
    lon0, lon1 = day_fires["lon"].min() - 0.3, day_fires["lon"].max() + 0.3
    active = events[(events["t_first"] <= d0 + pd.Timedelta(days=1, hours=6))
                    & (events["t_last"] >= d0 - pd.Timedelta(hours=12))
                    & events["lat_c"].between(lat0, lat1) & events["lon_c"].between(lon0, lon1)]
    top = active.sort_values("fre_tj", ascending=False).head(8).reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1)
    det = sev[sev["event_id"].isin(active["event_id"])]
    fe = fire_group(day_fires, sev)
    day_fires = day_fires.assign(event_id=fe, nucleus_id=fire_group(day_fires, sev, "nucleus_id"))
    nu = nuclei[nuclei["event_id"].isin(top["event_id"]) & (nuclei["n_det"] >= MIN_NUCLEUS_DET)]

    print(f"\nCase {day}: {len(day_fires)} ICNF fires ≥ {FOCUS_HA} ha start that day (UTC); "
          f"{len(active)} SEVIRI events active in the area ({(active['n_det'] >= 10).sum()} with ≥ 10 detections)")
    per_event = day_fires[day_fires["event_id"] >= 0].groupby("event_id").agg(
        fires=("fire_id", "size"), area_ha=("burned_area_ha", "sum"),
        names=("municipality", lambda m: ", ".join(sorted(set(str(x).title() for x in m)))))
    print(f"  fires linked to an event: {(fe >= 0).sum()} of {len(fe)}; distinct events: {len(per_event)}; "
          f"events holding ≥ 2 fires: {(per_event['fires'] >= 2).sum()} (with {per_event.loc[per_event['fires'] >= 2, 'fires'].sum()} fires)")
    tab = top.merge(per_event, on="event_id", how="left").fillna({"fires": 0})
    print(tab[["rank", "event_id", "t_first", "duration_h", "n_det", "n_pixels", "fre_tj", "fires", "area_ha", "names"]]
          .round(1).to_string(index=False))

    # overview
    fig, ax = plt.subplots(figsize=(9, 9 * (lat1 - lat0) / ((lon1 - lon0) * np.cos(np.radians(40)))))
    E.map_axes(ax, borders, f"SEVIRI events and ICNF fires ≥ {FOCUS_HA} ha, {day}", (lon0, lon1, lat0, lat1))
    ax.scatter(det["lon"], det["lat"], s=2 + det["frp_mw"] / 60, color=E.BLUE, alpha=0.25, lw=0, zorder=2)
    for r in top.itertuples():
        d = det[det["event_id"] == r.event_id]
        hull(ax, d["lon"], d["lat"], color=E.INK, lw=1.0, zorder=4)
        pt = d.loc[d["lat"].idxmax()]
        y, va = pt["lat"] + 0.03, "bottom"
        if y > lat1 - 0.1:                          # too close to the top edge: label below the hull
            pt = d.loc[d["lat"].idxmin()]
            y, va = pt["lat"] - 0.03, "top"
        ax.text(pt["lon"], y, str(r.rank), fontsize=10, fontweight="bold", color=E.INK, ha="center", va=va, zorder=6)
    ax.scatter(day_fires["lon"], day_fires["lat"], marker="*", s=40 + day_fires["burned_area_ha"] / 150,
               color=E.ORANGE, edgecolor=E.INK, lw=0.5, zorder=5)
    ax.scatter([], [], marker="*", s=80, color=E.ORANGE, edgecolor=E.INK, lw=0.5, label="ICNF ignition (≥ 100 ha)")
    ax.scatter([], [], s=10, color=E.BLUE, alpha=0.5, label="SEVIRI detection (size = FRP)")
    ax.plot([], [], color=E.INK, lw=1, label="hull of the 8 largest events (by FRE)")
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / f"events_{day}_overview.png"); plt.close(fig)
    print(f"figure: figures/clustering/events_{day}_overview.png")

    # one panel per large event: detections coloured by hours since event start
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, r in zip(axes.flat, top.itertuples()):
        d = det[det["event_id"] == r.event_id]
        h = (d["time_utc"] - r.t_first).dt.total_seconds() / 3600
        m = 0.08
        ex = (d["lon"].min() - m, d["lon"].max() + m, d["lat"].min() - m, d["lat"].max() + m)
        E.map_axes(ax, borders, "", ex)
        ax.texts[-1].remove()                       # no country label in zoomed panels
        sc = ax.scatter(d["lon"], d["lat"], c=h, cmap=E.SEQ_BLUE, s=8 + d["frp_mw"] / 40, lw=0, zorder=2)
        f = day_fires[day_fires["lon"].between(ex[0], ex[1]) & day_fires["lat"].between(ex[2], ex[3])]
        b = nu[nu["event_id"] == r.event_id]
        for x in f[f["nucleus_id"] >= 0].itertuples():   # ICNF ignition -> birth of its nucleus
            n = nuclei.loc[nuclei["nucleus_id"] == x.nucleus_id].iloc[0]
            ax.plot([x.lon, n.lon_birth], [x.lat, n.lat_birth], color=E.INK2, lw=0.8, zorder=4)
        ax.scatter(f["lon"], f["lat"], marker="*", s=90, color=E.ORANGE, edgecolor=E.INK, lw=0.6, zorder=5)
        ax.scatter(b["lon_birth"], b["lat_birth"], marker="X", s=60, color=E.INK, zorder=6)
        for x in b.itertuples():
            ax.annotate(str(x.order_in_event), (x.lon_birth, x.lat_birth), xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=E.INK, zorder=7)
        n_in = (day_fires["event_id"] == r.event_id).sum()
        ax.set_title(f"#{r.rank}: {r.n_det:,} det., {r.duration_h:.0f} h, {n_in} ICNF fire{'s' if n_in != 1 else ''}, "
                     f"{len(b)} nuclei", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel(""); ax.tick_params(labelsize=6)
        fig.colorbar(sc, ax=ax, shrink=0.75).ax.tick_params(labelsize=6)
    fig.suptitle(f"{day}: 8 largest SEVIRI events — colour = hours since event's first detection, "
                 f"★ = ICNF ignition (≥ 100 ha, started that day), ✕ = nucleus birth (≥ {MIN_NUCLEUS_DET} det., numbered "
                 "in birth order), line = ICNF fire → its nucleus",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"events_{day}_panels.png"); plt.close(fig)
    print(f"figure: figures/clustering/events_{day}_panels.png")


def fig_nuclei_vs_events(f):
    """Distance and lag from the ICNF ignition to the event start and to the nucleus birth."""
    later = f[f["position"] == "later in shared event"]
    rest = f[f["position"] != "later in shared event"]
    lines = [(later, "event", E.ORANGE, "--", f"joined a burning event, event start (n = {len(later):,})"),
             (later, "nucleus", E.BLUE, "-", f"joined a burning event, own nucleus birth (n = {len(later):,})"),
             (rest, "nucleus", E.AQUA, "-.", f"alone / first in event, nucleus birth (n = {len(rest):,})")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for d, kind, c, ls, label in lines:
        x = np.sort(d[f"dist_{kind}_km"].values)
        a1.plot(x, np.arange(1, len(x) + 1) / len(x) * 100, color=c, ls=ls, lw=2, label=label)
        y = np.sort(d[f"lag_{kind}_h"].clip(-24, 24).values)
        a2.plot(y, np.arange(1, len(y) + 1) / len(y) * 100, color=c, ls=ls, lw=2)
    a1.set_xlim(0, 40); a1.set_xlabel("distance ICNF ignition → start location [km]")
    a1.set_ylabel("fires [%] (cumulative)"); a1.set_title("Distance")
    a2.set_xlim(-24, 24); a2.axvline(0, color=E.MUTED, lw=0.8)
    a2.set_xlabel("start time − ICNF start [h] (clipped to ±24 h)"); a2.set_title("Lag")
    a1.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"ICNF fires ≥ {FOCUS_HA} ha 2004–2019: event start vs. nucleus birth", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "nuclei_vs_events.png"); plt.close(fig)
    print("figure: figures/clustering/nuclei_vs_events.png")


# ---------------------------------------------------------------- main

def main():
    sev, fires = load()
    if "--sensitivity" in sys.argv:
        sensitivity(sev, fires)
        return
    FIG.mkdir(parents=True, exist_ok=True)
    borders, mainland = E.load_borders()
    px = E.pixel_stats(sev)
    sev["event_id"], sev["nucleus_id"] = cluster(sev, nuclei=True)
    sev["persistent"] = sev.merge(px[["lat", "lon", "persistent"]], on=["lat", "lon"], how="left")["persistent"].values
    sev["in_pt"] = np.asarray(E.in_polygon(sev["lon"], sev["lat"], mainland))
    events = build_events(sev)
    nuclei = build_nuclei(sev)

    print(f"Clustering D = {D_KM} km, gap = {GAP_H} h")
    for col in ("event_id", "nucleus_id"):
        for k, v in diagnostics(sev, fires, col).items():
            print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
    big = events[events["n_det"] >= 10]
    print(f"  events ≥ 10 det: median {big['n_det'].median():.0f} det., {big['duration_h'].median():.1f} h; "
          f"mostly persistent pixels (≥ 50 %): {(big['share_persistent'] >= 0.5).sum()}")
    print(f"  events ≥ 10 det with ≥ 2 nuclei: {(big['n_nuclei'] >= 2).mean() * 100:.1f} %; "
          f"nuclei per event ≥ 10 det: median {big['n_nuclei'].median():.0f}, max {big['n_nuclei'].max()}")
    nb = nuclei[nuclei["n_det"] >= MIN_NUCLEUS_DET]
    later_born = nb[nb["order_in_event"] > 1]
    print(f"  nuclei ≥ {MIN_NUCLEUS_DET} det: {len(nb):,}; born into an event that already burned (order > 1): {len(later_born):,}; "
          f"reappeared after a gap: {nb['reappeared'].mean() * 100:.1f} %; "
          f"distance to nearest active detection at birth (order > 1): median {later_born['dist_active_km'].median():.1f} km")

    f = fire_positions(fires, sev, events, nuclei)
    print("\nICNF fires ≥ 100 ha (2004–2019) by position in their event:")
    print(position_summary(f).to_string())
    d0 = pd.Timestamp(CASE_DAY, tz="UTC")
    print(f"\nSame, fires starting on {CASE_DAY}:")
    print(position_summary(f[f["start_utc"].between(d0, d0 + pd.Timedelta(days=1))]).to_string())
    fig_nuclei_vs_events(f)

    sev.to_parquet(PROC / "seviri_frp_events.parquet", index=False)
    events.to_parquet(PROC / "seviri_events.parquet", index=False)
    nuclei.to_parquet(PROC / "seviri_nuclei.parquet", index=False)
    print("written: data/processed/seviri_frp_events.parquet, seviri_events.parquet, seviri_nuclei.parquet")
    fig_case(sev, events, nuclei, fires, borders)


if __name__ == "__main__":
    main()
