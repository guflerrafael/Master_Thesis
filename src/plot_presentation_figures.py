"""Summary figures for the supervisor-meeting slides (docs/presentation/).

  figures/presentation/georef_offset_by_year.png  ICNF ignition − first SEVIRI detection, median per year
  figures/presentation/ap_progression.png         AP @ 5 km over the project (numbers from notes §5.3–6.17)
  figures/presentation/test_summary.png           final test metrics, baseline vs. both models
  figures/presentation/recall_by_context.png      validation recall @ 10 km by context (model_review §3.3)
  figures/presentation/focal_loss.png             BCE vs. focal loss for an ignition and a background pixel

Run from the repository root:
  .venv/bin/python src/plot_presentation_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E

FIG = E.ROOT / "figures" / "presentation"
MODEL = E.PROC / "model"
KM_EW, KM_NS = 3.2, 4.5          # native cell size over Portugal [km]


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(name)


def georef_offset_by_year():
    """Per target: ICNF position (true frame) minus the FRP-weighted centroid of the first detection
    slot within 2 cells in [start − 1 h, start + 6 h] (raw sensor positions, no correction)."""
    fires = pd.read_parquet(MODEL / "fires.parquet")
    fires = fires[fires["is_target"]]
    det = pd.read_parquet(MODEL / "detections.parquet", columns=["time_utc", "i", "j", "frp_mw"])
    det = det.sort_values("time_utc").reset_index(drop=True)
    t = det["time_utc"].dt.tz_convert(None).values
    rows = []
    for f in fires.itertuples():
        s0 = f.start_utc.tz_convert(None)
        a, b = np.searchsorted(t, [(s0 - pd.Timedelta("1h")).to_datetime64(), (s0 + pd.Timedelta("6h")).to_datetime64()])
        d = det.iloc[a:b]
        d = d[(d["i"] - f.i_f) ** 2 + (d["j"] - f.j_f) ** 2 <= 4]
        if d.empty:
            continue
        d = d[d["time_utc"] == d["time_utc"].min()]
        w = d["frp_mw"]
        rows.append((f.start_utc, f.year, f.i_f - np.average(d["i"], weights=w), f.j_f - np.average(d["j"], weights=w)))
    off = pd.DataFrame(rows, columns=["start", "year", "di", "dj"])
    off["east_km"] = off["dj"] * KM_EW
    off["north_km"] = -off["di"] * KM_NS
    med = off.groupby("year")[["east_km", "north_km"]].median()
    n = off.groupby("year").size()
    print("  targets with a first detection:", len(off))
    print(med.round(2).to_string())

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axhline(0, color=E.BASE, lw=1)
    ax.axvspan(2017.9, 2019.5, color=E.NEUTRAL, zorder=0)
    ax.text(2018.7, -1.9, "after the\nEUMETSAT\ncorrection\n(Dec 2017)", ha="center", va="center", fontsize=8, color=E.INK2)
    ax.plot(med.index, med["east_km"], "-o", color=E.BLUE, lw=2, ms=5, label="east")
    ax.plot(med.index, med["north_km"], "-s", color=E.ORANGE, lw=2, ms=5, label="north")
    ax.text(med.index[0] - 0.3, med["east_km"].iloc[0], "east", color=E.INK2, ha="right", va="center")
    ax.text(med.index[0] - 0.3, med["north_km"].iloc[0], "north", color=E.INK2, ha="right", va="center")
    ax.set_xlim(2002.6, 2019.5)
    ax.set_xticks(range(2004, 2020, 2))
    ax.set_ylabel("median offset [km]")
    ax.set_title("ICNF ignition − first SEVIRI detection, per year (uncorrected positions)")
    ax.text(2019.45, -4.2, f"{len(off):,} target fires; years with few fires (2008: n = {n.get(2008, 0)}) are noisy",
            ha="right", fontsize=7.5, color=E.MUTED)
    ax.set_ylim(-4.5, 2.2)
    save(fig, "georef_offset_by_year.png")


def ap_progression():
    stages = [  # (label, AP @ 5 km, kind)
        ("baseline (nucleus birth)", 0.237, "base"),
        ("ConvLSTM v1, 1 seed", 0.302, "model"),
        ("reference, 3 seeds, last 5", 0.309, "model"),
        ("baseline, corrected", 0.448, "base"),
        ("reference, corrected", 0.597, "model"),
        ("3D CNN final (validation)", 0.621, "model"),
        ("3D CNN final (test 2017–19)", 0.623, "model"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    y = np.arange(len(stages))[::-1]
    for yi, (lab, v, k) in zip(y, stages):
        ax.barh(yi, v, height=0.62, color=E.BLUE if k == "model" else E.BASE)
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=8.5, color=E.INK)
    ax.set_yticks(y, [s[0] for s in stages])
    ax.axhline(3.5, color=E.ORANGE, lw=1.4, ls="--")
    ax.text(0.70, 3.62, "geo-referencing correction", color=E.ORANGE, fontsize=8.5, ha="right", va="bottom")
    ax.set_xlim(0, 0.72)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AP @ 5 km (validation 2015–16 unless noted)")
    ax.set_title("Localisation quality over the project")
    save(fig, "ap_progression.png")


def test_summary():
    metrics = ["AP @ 5 km", "AP @ 10 km", "recall @ 10 km", "recall @ 5 km\nwith neighbours", "recall @ 5 km\n≥ 1,000 ha"]
    series = [  # (name, values, std, colour)
        ("baseline: nucleus birth", [0.501, 0.593, 0.856, 0.734, 0.733], [0] * 5, E.BASE),
        ("3D CNN", [0.623, 0.711, 0.938, 0.833, 0.839], [0.012, 0.004, 0.004, 0.003, 0.010], E.BLUE),
        ("ConvLSTM", [0.620, 0.714, 0.933, 0.826, 0.817], [0.013, 0.013, 0.008, 0.003, 0.000], E.AQUA),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(metrics))
    w = 0.26
    for k, (name, v, s, c) in enumerate(series):
        xs = x + (k - 1) * (w + 0.01)
        ax.bar(xs, v, width=w, color=c, label=name, yerr=s, error_kw=dict(lw=0.8, ecolor=E.INK2))
    for xi, v in zip(x, series[1][1]):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", fontsize=8, color=E.INK)
    for xi, v in zip(x, series[0][1]):
        ax.text(xi - w - 0.01, v + 0.025, f"{v:.2f}", ha="center", fontsize=8, color=E.INK2)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=8.5)
    ax.set_title("Test 2017–2019 (3 seeds, mean ± std)", pad=24)
    save(fig, "test_summary.png")


def recall_by_context():
    groups = [  # (label, n, model, baseline) — recall @ 10 km, validation, cnn3d_final ensemble
        ("another target ≤ 2 cells,\nsame period", 52, 0.44, 0.37),
        ("ignition next to an\nactive fire (6 h history)", 72, 0.75, 0.31),
        ("same place burned\n6–78 h before", 89, 0.89, 0.87),
        ("fresh (nothing nearby\nin the 78 h before)", 242, 0.95, 0.93),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(groups))[::-1]
    h = 0.36
    ax.barh(y + h / 2 + 0.01, [g[3] for g in groups], height=h, color=E.BASE, label="baseline")
    ax.barh(y - h / 2 - 0.01, [g[2] for g in groups], height=h, color=E.BLUE, label="3D CNN")
    for yi, g in zip(y, groups):
        ax.text(g[2] + 0.01, yi - h / 2 - 0.01, f"{g[2]:.2f}", va="center", fontsize=8)
        ax.text(g[3] + 0.01, yi + h / 2 + 0.01, f"{g[3]:.2f}", va="center", fontsize=8, color=E.INK2)
    ax.set_yticks(y, [f"{g[0]}  (n = {g[1]})" for g in groups], fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("recall @ 10 km (validation 2015–16)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Where the model wins, and where it still fails")
    save(fig, "recall_by_context.png")


def focal_loss():
    p = np.linspace(0.005, 0.995, 400)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=True)
    ax = axes[0]
    ax.plot(p, -np.log(p), color=E.BASE, lw=2, label="cross-entropy")
    ax.plot(p, -(1 - p) ** 2 * np.log(p), color=E.BLUE, lw=2, label="focal: (1 − p)² weight")
    ax.set_title("ignition pixel (y = 1)")
    ax.set_xlabel("predicted probability p")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(p, -np.log(1 - p), color=E.BASE, lw=2, label="cross-entropy")
    ax.plot(p, -p ** 2 * np.log(1 - p), color=E.BLUE, lw=2, label="focal, far away (y = 0)")
    ax.plot(p, -(1 - 0.6) ** 4 * p ** 2 * np.log(1 - p), color=E.AQUA, lw=2, ls="--",
            label="focal, next to ignition (y = 0.6)")
    ax.set_title("background pixel")
    ax.set_xlabel("predicted probability p")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_ylim(0, 4)
        ax.set_xlim(0, 1)
    save(fig, "focal_loss.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    georef_offset_by_year()
    ap_progression()
    test_summary()
    recall_by_context()
    focal_loss()
