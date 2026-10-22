"""Non-learned baselines from the SEVIRI nuclei (notes.md 3.2, 5.3), scored with src/evaluate.py.

Every nucleus born inside a 12 h target period is one predicted ignition of that period.
Score = fire radiative energy of the nucleus within the sample's input window (up to the end
of the 6 h tail), i.e. only information a model would also see. Variants:
  birth                         location = FRP-weighted centroid of the birth time step
  birth, no reappeared          without nuclei born where the same place burned 6-48 h before
  birth, no reappeared/persist. also without nuclei mostly on persistent off-season pixels
  first-hour centroid           as 'no reappeared/persist.', location = FRP-weighted centroid
                                of the nucleus' detections in its first hour

The test split (2017-2019) is not evaluated here: it stays locked until the final comparison.

Run from the repository root:
  .venv/bin/python src/baselines.py > outputs/baselines_log.txt
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from build_dataset import period_of
from dataset import FireData
from evaluate import evaluate
from tiling_design import PERIOD_H, TAIL_H

FIG = E.ROOT / "figures" / "baselines"
SPLITS = ("train", "val")
KS = (5, 10)


def predictions(data):
    nu = pd.read_parquet(E.PROC / "seviri_nuclei.parquet")
    det = data.det
    nu["period"] = period_of(nu["t_birth"])
    nu = nu[nu["period"].isin(set(data.periods["period"]))]
    end = (nu.set_index("nucleus_id")["period"] + pd.Timedelta(hours=PERIOD_H + TAIL_H))
    birth = nu.set_index("nucleus_id")["t_birth"]

    d = det[det["nucleus_id"].isin(nu["nucleus_id"])].copy()
    d["end"] = d["nucleus_id"].map(end)
    d = d[d["time_utc"] < d["end"]]
    d["fre"] = d["frp_mw"].astype(float) * 900 / 1e6
    score = d.groupby("nucleus_id")["fre"].sum()
    first = d[d["time_utc"] < d["nucleus_id"].map(birth) + pd.Timedelta(hours=1)]
    w = first.assign(wl=first["lat"] * first["frp_mw"], wo=first["lon"] * first["frp_mw"]).groupby("nucleus_id")
    lat1h, lon1h = w["wl"].sum() / w["frp_mw"].sum(), w["wo"].sum() / w["frp_mw"].sum()

    nu = nu.reset_index(drop=True)
    if data.georef_fix:     # sensor frame -> true frame for births before the change (notes.md 6.11)
        nu["lat_birth"], nu["lon_birth"] = G.sensor_to_true_latlon(nu["lat_birth"].values, nu["lon_birth"].values, nu["t_birth"])
        tb = birth.reindex(lat1h.index)
        lat1h[:], lon1h[:] = G.sensor_to_true_latlon(lat1h.values, lon1h.values, tb)
    base = pd.DataFrame({"period": nu["period"], "lat": nu["lat_birth"], "lon": nu["lon_birth"],
                         "score": nu["nucleus_id"].map(score), "reappeared": nu["reappeared"],
                         "persistent": nu["share_persistent"] >= 0.5, "nucleus_id": nu["nucleus_id"]})
    clean = base[~base["reappeared"] & ~base["persistent"]]
    return {
        "birth": base,
        "birth, no reappeared": base[~base["reappeared"]],
        "birth, no reappeared/persist.": clean,
        "first-hour centroid": clean.assign(lat=clean["nucleus_id"].map(lat1h).values,
                                            lon=clean["nucleus_id"].map(lon1h).values),
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = FireData()
    preds = predictions(data)
    print("Predictions per variant (all periods): " + ", ".join(f"{k}: {len(v):,}" for k, v in preds.items()))

    rows, curves = [], {}
    for name, p in preds.items():
        for split in SPLITS:
            for k in KS:
                res, curve, _ = evaluate(p, data, split, k)
                rows.append({"baseline": name, **res})
                curves[(name, split, k)] = curve
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    main_cols = ["baseline", "split", "k [km]", "targets", "AP", "recall (all preds)", "precision (all preds)", "best F1",
                 "P @ best F1", "R @ best F1", "false alarms / day @ best F1", "median error TP [km]",
                 "nearest pred ≤ 5 km", "nearest pred median [km]"]
    print("\nBaselines (test split locked):")
    print(tab[main_cols].round(3).to_string(index=False))
    grp_cols = ["baseline", "split", "k [km]"] + [c for c in tab.columns if c.startswith("recall |")]
    print("\nRecall with all predictions, by subgroup:")
    print(tab[grp_cols].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    styles = [(E.BLUE, "-"), (E.ORANGE, "--"), (E.AQUA, "-."), (E.INK2, ":")]
    for ax, k in zip(axes, KS):
        for (name, _), (c, ls) in zip(preds.items(), styles):
            cv = curves[(name, "val", k)]
            ap = tab.loc[(tab["baseline"] == name) & (tab["split"] == "val") & (tab["k [km]"] == k), "AP"].iat[0]
            ax.plot(cv["recall"], cv["precision"], color=c, ls=ls, lw=2, label=f"{name} (AP {ap:.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (share of target ignitions found)")
        ax.set_title(f"Validation 2015–2016, match within {k} km")
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves_val.png"); plt.close(fig)
    print("\nfigure: figures/baselines/pr_curves_val.png")


if __name__ == "__main__":
    main()
