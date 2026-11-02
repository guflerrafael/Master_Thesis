"""Water-edge mask of the FRP product and why SEVIRI misses fires (notes.md 6.20).

Flag WATEREDGE (11) of the Quality Product means no fire detection took place. Uses the 2015-2016
files of src/download_msg_quality.py, the checkpoint predictions kept by src/cloud_diagnostic.py and
the validation split only for everything that involves the model. Blind cells = flag 11 on every
day of the last 30 days of 2016.

Parts of the report:
  1 mask over time    share of Portuguese cells with flag 11 per stable stretch of days
  2 flags             what these cells and the other cells report
  3 detections        share of the detections on blind cells per year, 2004-2025
  4 ICNF fires        share seen by SEVIRI, fires >= 50 ha, blind cells vs. others, 2004-2019
  5 unseen fires      validation fires >= 50 ha by cloud and blind cell
  6 model             recall and false alarms on blind cells, validation

Run from the repository root:
  .venv/bin/python src/water_edge_check.py | tee outputs/cloud/water_edge.txt
Figure: figures/cloud/water_edge_map.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cloud_diagnostic as C
import eda_figures as E
import grid as G
from dataset import FireData
from download_msg_quality import DAYS_DIR, INVENTORY, MARGIN, MISSING, day_path
from evaluate import average_precision, match

FIG = E.ROOT / "figures" / "cloud"
WATER_EDGE = 11
BIG_HA = 50
LAST_DAYS = 30
SEEDS = (0, 1, 2)


def day_flags(day):
    return np.load(day_path(day))["flags"][:, MARGIN:-MARGIN, MARGIN:-MARGIN]


def flag_shares(values):
    return ", ".join(f"{a}: {b / values.size:.1%}" for a, b in zip(*np.unique(values, return_counts=True)))


def mask_over_time(pt):
    """Blind cells and the share of flag 11 per stable stretch of days."""
    inv = pd.read_csv(INVENTORY, keep_default_na=False).set_index("date")
    days = pd.date_range("2015-01-01", "2016-12-31")
    rows, any_edge = [], []
    for d in days:
        f = day_flags(d.date())
        e = f == WATER_EDGE
        rows.append({"date": d.date().isoformat(), "edge": e[:, pt].sum() / max((f != MISSING)[:, pt].sum(), 1),
                     "versions": inv.loc[d.date().isoformat(), "versions"]})
        any_edge.append(e.any(0))
    t = pd.DataFrame(rows)
    t["stretch"] = (t["edge"].round(3).diff().abs() > 0.01).cumsum()
    print("\n1. Share of Portuguese cells with flag 11")
    print(t.groupby("stretch").agg(start=("date", "first"), end=("date", "last"), days=("date", "size"), share=("edge", "mean"),
                                   versions=("versions", lambda v: sorted(set(v)))).round(3).to_string(index=False))
    any_edge = np.array(any_edge)
    return any_edge[-LAST_DAYS:].all(0), any_edge[:LAST_DAYS].mean(0) > 0.5


def report_flags(blind, pt):
    f = np.concatenate([day_flags(pd.Timestamp(d).date()) for d in ("2016-10-20", "2016-12-20")])
    print("\n2. Flags on two days of late 2016")
    print(f"  blind cells: {flag_shares(f[:, blind & pt])}")
    print(f"  other Portuguese cells: {flag_shares(f[:, ~blind & pt])}")


def report_detections(blind, pt):
    d = pd.concat([pd.read_parquet(E.PROC / n) for n in ("seviri_frp.parquet", "seviri_frp_2020_2025.parquet")], ignore_index=True)
    r, c = G.native_rc(d["lat"].values, d["lon"].values)
    i, j = np.round(G.rc_to_ij(np.round(r), np.round(c))).astype(int)
    on = (i >= 0) & (i < G.H) & (j >= 0) & (j < G.W)
    d = d[on].assign(i=i[on], j=j[on])
    d = d[pt[d["i"], d["j"]]]
    d = d.assign(blind=blind[d["i"], d["j"]])
    y = d.groupby([d["time_utc"].dt.year, "blind"]).size().unstack(fill_value=0)
    y["share on blind"] = y[True] / (y[True] + y[False])
    print(f"\n3. Detections on Portuguese cells (blind cells are {blind[pt].mean():.1%} of the cells)")
    print(y.round(3).to_string())
    return d


def report_fires(data, blind, pt):
    f = data.fires[data.fires["burned_area_ha"] >= BIG_HA]
    i, j = C.sensor_cells(f["lat"].values, f["lon"].values, f["start_utc"].values)
    f = f.assign(blind=blind[i, j], on_pt=pt[i, j])
    f = f[f["on_pt"]]
    print(f"\n4. Fires >= {BIG_HA} ha on Portuguese cells, share seen")
    print(f.groupby(["split", "blind"])["seen"].agg(["size", "mean"]).unstack().round(3).to_string())
    print(f.groupby(["year", "blind"])["seen"].agg(["size", "mean"]).unstack().round(2).to_string())


def report_unseen(data, cloud, blind, pt):
    f = data.fires[(data.fires["split"] == C.SPLIT) & (data.fires["burned_area_ha"] >= BIG_HA)]
    i, j = C.sensor_cells(f["lat"].values, f["lon"].values, f["start_utc"].values)
    keep = pt[i, j]
    f, i, j = f[keep], i[keep], j[keep]
    s0 = np.array([C.slot_of(t) for t in f["start_utc"]])
    s1 = np.maximum(np.array([C.slot_of(t) for t in f["end_utc"].fillna(f["start_utc"])]), s0 + 24)
    cover = np.array([cloud.unobs(a, b, x, y) for a, b, x, y in zip(s0, s1, i, j)])
    f = f.assign(blind=blind[i, j], cloudy=cover > C.HIGH_CLOUD)
    t = f.groupby(["blind", "cloudy"])["seen"].agg(fires="size", seen="mean", unseen=lambda s: (~s).sum())
    ref = t.loc[(False, False), "seen"]
    excess = (t["fires"] * (ref - t["seen"])).sum()
    print(f"\n5. Validation fires >= {BIG_HA} ha (cloudy = over {C.HIGH_CLOUD:.0%} not observable from start to end, at least 6 h)")
    print(t.round(3).to_string())
    print(f"  unseen in a cloudy or blind group: {t['unseen'].sum() - t.loc[(False, False), 'unseen']} of {t['unseen'].sum()}; "
          f"more than at the seen rate {ref:.3f} of the other fires: {excess:.0f}")


def report_model(data, blind):
    periods = set(data.periods.loc[data.periods["split"] == C.SPLIT, "period"])
    lab = data.labels[data.labels["period"].isin(periods) & (data.labels["role"] == "target")]
    on_blind = blind[np.round(lab["i_f"].values).astype(int), np.round(lab["j_f"].values).astype(int)]
    rows = []
    for seed in SEEDS:
        p = pd.read_parquet(C.OUT / f"val_preds_s{seed}.parquet")
        for k in (5, 10):
            s, targets = match(p, data, periods, k)
            i, j = C.sensor_cells(s["lat"].values, s["lon"].values, s["period"].values)
            s = s.assign(blind=blind[i, j])
            hit = lab["fire_id"].isin(s.loc[s["status"] == "tp", "fire_id"]).values
            rows.append({"k [km]": k, "AP": average_precision(s, len(targets))[0],
                         "AP without false alarms on blind": average_precision(s[~((s["status"] == "fp") & s["blind"])], len(targets))[0],
                         "recall on blind": hit[on_blind].mean(), "recall elsewhere": hit[~on_blind].mean()})
    print(f"\n6. Model, validation, mean over seeds ({on_blind.sum()} of {len(lab)} targets on blind cells)")
    print(pd.DataFrame(rows).groupby("k [km]").mean().round(3).T.to_string())


def fig_map(blind, early, pt, det):
    h = np.zeros((G.H, G.W))
    e = det[det["time_utc"].dt.year.between(2017, 2019)]
    np.add.at(h, (e["i"].values, e["j"].values), 1)
    h[~pt] = np.nan
    fig, axes = plt.subplots(1, 3, figsize=(11, 6))
    for ax, a, title in ((axes[0], early, "Water edge, January 2015"), (axes[1], blind, "Water edge, December 2016")):
        ax.imshow(np.where(pt, a.astype(float), np.nan), cmap="coolwarm", vmin=0, vmax=1)
        ax.set_title(title)
    axes[2].imshow(np.log1p(h), cmap="magma")
    axes[2].set_title("Detections 2017–2019 (log)")
    for ax in axes:
        ax.axis("off")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "water_edge_map.png", dpi=150)
    plt.close(fig)
    print("\nfigure: figures/cloud/water_edge_map.png")


def main():
    pd.set_option("display.width", 200)
    data = FireData()
    pt = data.static["pt_loss"].astype(bool)
    blind, early = mask_over_time(pt)
    print(f"  blind cells: {(blind & pt).sum():,} of {pt.sum():,}; also in January 2015: {(blind & early & pt).sum():,}")
    report_flags(blind, pt)
    det = report_detections(blind, pt)
    report_fires(data, blind, pt)
    cloud = C.Cloud(DAYS_DIR, (C.FIRST_DAY - pd.Timedelta(days=C.BUFFER_DAYS)).date(), pd.Timestamp("2017-01-02").date())
    report_unseen(data, cloud, blind, pt)
    report_model(data, blind)
    fig_map(blind, early, pt, det)


if __name__ == "__main__":
    main()
