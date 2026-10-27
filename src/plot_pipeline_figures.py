"""Figures that explain the data pipeline (labels, loss weights, time aggregation, frames).

  figures/pipeline/sample_labels.png   one validation period: input, target heatmap, loss weight
  figures/pipeline/time_aggregation.png  15-min detections of one target pixel vs. hourly maxima
  figures/pipeline/georef_frames.png   true vs. sensor-frame ignition and the first detections

Validation data only (2015-2016). Run from the repository root:
  .venv/bin/python src/plot_pipeline_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import eda_figures as E
import grid as G
from dataset import FireData, T
from train import DEFAULTS, features, static_crop

FIG = E.ROOT / "figures" / "pipeline"
PERIOD = pd.Timestamp("2016-08-06 06:00", tz="UTC")
WIN = (33, 70, 26, 70)          # i0, i1, j0, j1 of the zoom window (cells)
ROLE_STYLE = {"target": dict(marker="+", s=70, lw=1.6, color=E.INK),
              "ignore": dict(marker="x", s=36, lw=1.2, color=E.ORANGE)}


def _border(ax, mainland, i0, j0):
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    ax.plot(bj - j0, bi - i0, color=E.MUTED, lw=0.8)


def _labels(ax, lab, i0, j0, fires):
    for role, st in ROLE_STYLE.items():
        L = lab[lab["role"] == role]
        ax.scatter(L["j_f"] - j0, L["i_f"] - i0, zorder=5, **st)
    for q in lab.itertuples():
        f = fires.loc[q.fire_id]
        ax.annotate(f"{f.burned_area_ha:.0f} ha", (q.j_f - j0, q.i_f - i0), xytext=(4, 3),
                    textcoords="offset points", fontsize=6.5, color=E.INK2)


def fig_sample(data, mainland):
    x, y, w, lab = data.sample(PERIOD)
    i0, i1, j0, j1 = WIN
    fires = data.fires.set_index("fire_id")
    lab = lab[lab["i_f"].between(i0, i1) & lab["j_f"].between(j0, j1)]
    sl = (slice(i0, i1), slice(j0, j1))
    det = x[:, 2] > 0
    first_h = np.where(det.any(0), det.argmax(0) / 4, np.nan)[sl]
    frp = np.where(det.any(0), np.expm1(x[:, 0].max(0) * np.log1p(1100)), np.nan)[sl]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
    ax = axes[0]
    im = ax.imshow(first_h, cmap=E.SEQ_BLUE.reversed(), vmin=0, vmax=24, interpolation="nearest")
    ax.axvspan(-0.5, -0.5, color="none")
    fig.colorbar(im, ax=ax, shrink=0.75, label="hour of first detection in the 24 h window\n(0–6 history, 6–18 period, 18–24 tail)")
    ax.set_title("(a) input: where and when SEVIRI detected fire")
    ax = axes[1]
    im = ax.imshow(y[sl], cmap=E.SEQ_ORANGE, vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.75, label="target heatmap y (Gaussian, σ = 1 px)")
    ax.set_title("(b) target: one bump per target ignition")
    ax = axes[2]
    ax.imshow(w[sl], cmap="Greys_r", vmin=-0.4, vmax=1, interpolation="nearest")
    ax.set_title("(c) loss weight w: white = counted, dark = no loss")
    for ax in axes:
        _border(ax, mainland, i0, j0)
        _labels(ax, lab, i0, j0, fires)
        ax.set_xlim(-0.5, j1 - j0 - 0.5); ax.set_ylim(i1 - i0 - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    h = [plt.Line2D([], [], ls="", **{k: v for k, v in dict(ROLE_STYLE["target"], markersize=9).items() if k not in ("s", "lw")}, mew=1.6, label="target (≥ 50 ha, seen)"),
         plt.Line2D([], [], ls="", **{k: v for k, v in dict(ROLE_STYLE["ignore"], markersize=7).items() if k not in ("s", "lw")}, mew=1.2, label="ignore (10–50 ha, unseen ≥ 50 ha, or boundary)")]
    axes[2].legend(handles=h, loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
    fig.suptitle(f"Validation period {PERIOD:%Y-%m-%d %H:%M} UTC + 12 h, zoom on central Portugal "
                 f"({i1 - i0} × {j1 - j0} cells ≈ {(j1 - j0) * 3.2:.0f} × {(i1 - i0) * 4.5:.0f} km); "
                 "positions in the sensor frame", fontsize=10, x=0.01, ha="left")
    fig.savefig(FIG / "sample_labels.png"); plt.close(fig)
    print("figure: figures/pipeline/sample_labels.png;", lab["role"].value_counts().to_dict())


def _largest_target(data, lab):
    tg = lab[lab["role"] == "target"]
    return tg.loc[tg["fire_id"].map(data._area).idxmax()]


def fig_time(data):
    """One target of the example period: its nearest detected pixel over the 96 slots vs. hourly steps."""
    lab = data._lab[PERIOD]
    q = _largest_target(data, lab)
    x, _, _, _ = data.sample(PERIOD)
    det = x[:, 2]
    ii, jj = np.nonzero(det.any(0))
    k = np.argmin(np.hypot(ii - q.i_f, jj - q.j_f))
    pi, pj = ii[k], jj[k]
    frp15 = np.where(det[:, pi, pj] > 0, np.expm1(x[:, 0, pi, pj] * np.log1p(1100)), np.nan)
    f = features(x[:, :, pi:pi + 1, pj:pj + 1], static_crop(data, (pi, pj, 1)), DEFAULTS)
    chans = DEFAULTS["channels"]
    frph = np.where(f[:, chans.index("detected"), 0, 0] > 0,
                    np.expm1(f[:, chans.index("frp"), 0, 0] * np.log1p(1100)), np.nan)
    start_h = (data.fires.set_index("fire_id").loc[q.fire_id, "start_utc"] - (PERIOD - pd.Timedelta(hours=6))).total_seconds() / 3600

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, layout="constrained")
    for ax in axes:
        ax.axvspan(0, 6, color=E.NEUTRAL, zorder=0); ax.axvspan(18, 24, color=E.NEUTRAL, zorder=0)
        ax.axvline(start_h, color=E.ORANGE, lw=1.2, ls="--")
        ax.set_yscale("log"); ax.set_ylim(5, 1500); ax.set_ylabel("FRP [MW]")
    t15 = np.arange(T) / 4
    axes[0].bar(t15, frp15, width=0.22, align="edge", color=E.BLUE)
    axes[0].set_title(f"(a) native 15-min slots: the pixel nearest a target ignition (cell {pi}, {pj}), 96 slots")
    axes[1].bar(np.arange(24), frph, width=0.9, align="edge", color=E.BLUE)
    axes[1].set_title("(b) model input after hourly aggregation (step_agg = 4): maximum of each hour, 24 steps")
    axes[1].set_xlabel("hours since window start (grey = 6 h history and 6 h tail; dashed = ICNF start)")
    axes[1].set_xlim(0, 24); axes[1].set_xticks(range(0, 25, 2))
    fig.savefig(FIG / "time_aggregation.png"); plt.close(fig)
    print(f"figure: figures/pipeline/time_aggregation.png (fire {q.fire_id}, detected slots {np.isfinite(frp15).sum()}, hours {np.isfinite(frph).sum()})")


def fig_frames(data):
    """Zoom on one pre-2017 target: true ICNF position, sensor-frame label, first-hour detections."""
    lab = data._lab[PERIOD]
    q = _largest_target(data, lab)
    f = data.fires.set_index("fire_id").loc[q.fire_id]
    x, _, _, _ = data.sample(PERIOD)
    s = int(((f.start_utc - (PERIOD - pd.Timedelta(hours=6))).total_seconds() // 900))
    early = x[max(s - 4, 0):s + 8, 2].any(0)
    ii, jj = np.nonzero(early)
    near = np.hypot(ii - q.i_f, jj - q.j_f) <= 3
    ci, cj = int(round(q.i_f)), int(round(q.j_f))
    fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
    for i in range(ci - 3, ci + 4):
        for j in range(cj - 3, cj + 4):
            on = early[i, j] if (0 <= i < G.H and 0 <= j < G.W) else False
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=E.BLUE if on else "white",
                                   alpha=0.35 if on else 1, edgecolor=E.BASE, lw=0.8))
    ax.scatter(f.j_f, f.i_f, marker="*", s=180, color=E.ORANGE, zorder=5, label="ICNF ignition, true position (fires.parquet)")
    ax.scatter(q.j_f, q.i_f, marker="+", s=150, lw=2, color=E.INK, zorder=6, label="label in the sensor frame (labels.parquet)")
    ax.annotate("", xy=(q.j_f, q.i_f), xytext=(f.j_f, f.i_f), arrowprops=dict(arrowstyle="->", color=E.INK2, lw=1))
    ax.scatter([], [], marker="s", s=80, color=E.BLUE, alpha=0.35, label="cells detected from 1 h before to 2 h after the start")
    ax.set_xlim(cj - 3.5, cj + 3.5); ax.set_ylim(ci + 3.5, ci - 3.5); ax.set_aspect(4.5 / 3.2)
    ax.set_xlabel("image column j (west → east, 1 cell ≈ 3.2 km)"); ax.set_ylabel("image row i (north → south, 1 cell ≈ 4.5 km)")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7.5)
    ax.set_title(f"Before 2017-12-01 labels move ½ cell north-west\n(fire of {f.start_utc:%Y-%m-%d %H:%M} UTC, {f.burned_area_ha:.0f} ha)")
    fig.savefig(FIG / "georef_frames.png"); plt.close(fig)
    print(f"figure: figures/pipeline/georef_frames.png ({near.sum()} detected cells within 3 px)")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = FireData()
    _, mainland = E.load_borders()
    fig_sample(data, mainland)
    fig_time(data)
    fig_frames(data)


if __name__ == "__main__":
    main()
