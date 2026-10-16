"""Predicted ignition probability vs. ICNF ignitions for validation periods (notes.md 6.2).

Periods are chosen by a fixed rule (no cherry-picking), validation split only (test locked):
  1. the validation period with the most target ignitions
  2. a random period with 2-5 targets      (seed 0)
  3. a random period with exactly 1 target  (seed 0)
  4. a random period with detections in Portugal but no target (hard negative, seed 0)
Columns: input (max FRP over the 24 h window), predicted probability with the predicted points
above the best-F1 threshold (10 km) and the ICNF ignitions, target heatmap with loss mask.

Run from the repository root:
  .venv/bin/python src/plot_predictions.py --name convlstm_v1
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import eda_figures as E
import grid as G
from dataset import FRP_NORM, FireData
from evaluate import evaluate
from model import IgnitionConvLSTM
from train import MODELS, features, heatmap_to_points, predict, static_crop

FIG = E.ROOT / "figures" / "model"
PROB_VMAX = 0.6   # colour scale of the probability panels (model output stays below ~0.6)


def choose_periods(per):
    v = per[per["split"] == "val"]
    rng = np.random.default_rng(0)
    pick = lambda d: d["period"].iloc[int(rng.integers(len(d)))]
    return [("most targets", v.loc[v["n_targets"].idxmax(), "period"]),
            ("2–5 targets (random)", pick(v[v["n_targets"].between(2, 5)])),
            ("1 target (random)", pick(v[v["n_targets"] == 1])),
            ("no target, fires burning (random)", pick(v[(v["n_targets"] == 0) & (v["n_det_pt"] > 0)]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convlstm_v1")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(MODELS / f"{args.name}.pt", map_location=device, weights_only=False)   # own checkpoint
    model = IgnitionConvLSTM(in_ch=7).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    step_agg = ck["config"]["step_agg"]
    data = FireData()

    # operating threshold: best F1 at 10 km on the whole validation split
    val_periods = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    _, curve, _ = evaluate(predict(model, data, val_periods, step_agg, device), data, "val", 10)
    f1 = 2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]).clip(lower=1e-9)
    thr = float(curve["score"].iloc[int(f1.values.argmax())])
    print(f"best-F1 threshold (val, 10 km): p ≥ {thr:.3f}")

    # view: mainland Portugal (+ margin)
    rows_pt, cols_pt = np.nonzero(data.static["pt"])
    i0, i1, j0, j1 = rows_pt.min() - 3, rows_pt.max() + 4, cols_pt.min() - 3, cols_pt.max() + 4
    _, mainland = E.load_borders()
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    st = static_crop(data)

    chosen = choose_periods(data.periods)
    zoom = {}
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 4.6 * len(chosen)), layout="constrained")
    for r, (label, period) in enumerate(chosen):
        x, y, w, lab = data.sample(period)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(features(x, st, step_agg))[None].to(device)))[0].cpu().numpy()
        fi, fj, s = heatmap_to_points(prob)
        keep = s >= thr
        tg, ig = lab[lab["role"] == "target"], lab[lab["role"] == "ignore"]
        frp = np.expm1(x[:, 0] * FRP_NORM).max(0)
        if r == 0:
            zoom = dict(period=period, label=label, x=x, y=y, w=w, prob=prob, frp=frp, tg=tg, ig=ig,
                        fi=fi[keep], fj=fj[keep])
        panels = [(np.where(frp > 0, np.log1p(frp), np.nan), E.SEQ_BLUE, (0, np.log1p(1000)), "input: max FRP, 24 h [log(1+MW)]"),
                  (prob, E.SEQ_ORANGE, (0, PROB_VMAX), f"predicted ignition probability (scale 0–{PROB_VMAX})"),
                  (y, E.SEQ_ORANGE, (0, 1), "ICNF target heatmap (grey = no loss)")]
        for c, (img, cmap, (v0, v1), title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            if c == 2:
                ax.imshow(np.where(w == 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3, alpha=0.5, interpolation="nearest")
            ax.plot(bj, bi, color=E.MUTED, lw=0.7)
            if c >= 1:
                ax.scatter(tg["j_loc"], tg["i_loc"], marker="*", s=90, facecolor="none", edgecolor=E.INK, lw=0.9,
                           label="ICNF target ignition", zorder=5)
            if c == 1:
                ax.scatter(ig["j_loc"], ig["i_loc"], marker="o", s=40, facecolor="none", edgecolor=E.MUTED, lw=0.8,
                           label="ICNF ignore fire", zorder=4)
                ax.scatter(fj[keep], fi[keep], marker="x", s=35, color=E.BLUE, lw=1.4,
                           label=f"predicted ignition (p ≥ {thr:.2f})", zorder=6)
            ax.set_xlim(j0, j1); ax.set_ylim(i1, i0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=9)
            if r == 0:
                fig.colorbar(im, ax=ax, shrink=0.6)
        axes[r, 0].set_ylabel(f"{period:%Y-%m-%d %H:%M} UTC + 12 h\n{label}\n"
                              f"{len(tg)} targets, {int(keep.sum())} predicted", fontsize=9)
        print(f"{period:%Y-%m-%d %H:%M}: {label}: {len(tg)} targets, {len(ig)} ignore, "
              f"{int(keep.sum())} predicted points ≥ threshold, max p {prob.max():.2f}")
    axes[0, 1].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle(f"{args.name}: predicted ignition probability vs. ICNF ignitions, validation 2015–2016",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / f"{args.name}_predictions_val.png", dpi=170); plt.close(fig)
    print(f"figure: figures/model/{args.name}_predictions_val.png")

    # zoom on the busiest period: extent of its targets + margin
    z = zoom
    m = 6
    zi0, zi1 = int(z["tg"]["i_loc"].min()) - m, int(z["tg"]["i_loc"].max()) + m
    zj0, zj1 = int(z["tg"]["j_loc"].min()) - m, int(z["tg"]["j_loc"].max()) + m
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2), layout="constrained")
    panels = [(np.where(z["frp"] > 0, np.log1p(z["frp"]), np.nan), E.SEQ_BLUE, (0, np.log1p(1000)), "input: max FRP, 24 h [log(1+MW)]"),
              (z["prob"], E.SEQ_ORANGE, (0, PROB_VMAX), f"predicted ignition probability (scale 0–{PROB_VMAX})"),
              (z["y"], E.SEQ_ORANGE, (0, 1), "ICNF target heatmap (grey = no loss)")]
    for c, (img, cmap, (v0, v1), title) in enumerate(panels):
        ax = axes[c]
        im = ax.imshow(img, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
        if c == 2:
            ax.imshow(np.where(z["w"] == 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3, alpha=0.5, interpolation="nearest")
        ax.plot(bj, bi, color=E.MUTED, lw=0.8)
        ax.scatter(z["tg"]["j_loc"], z["tg"]["i_loc"], marker="*", s=150, facecolor="none", edgecolor=E.INK, lw=1,
                   label="ICNF target ignition", zorder=5)
        if c == 1:
            ax.scatter(z["ig"]["j_loc"], z["ig"]["i_loc"], marker="o", s=60, facecolor="none", edgecolor=E.MUTED, lw=1,
                       label="ICNF ignore fire", zorder=4)
            ax.scatter(z["fj"], z["fi"], marker="x", s=60, color=E.BLUE, lw=1.8, label=f"predicted ignition (p ≥ {thr:.2f})", zorder=6)
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
        ax.set_xlim(zj0, zj1); ax.set_ylim(zi1, zi0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"{args.name}, {z['period']:%Y-%m-%d %H:%M} UTC + 12 h ({z['label']}): {len(z['tg'])} ICNF target ignitions, "
                 f"{len(z['fi'])} predicted; one pixel ≈ 3.2 × 4.5 km", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / f"{args.name}_predictions_val_zoom.png", dpi=170); plt.close(fig)
    print(f"figure: figures/model/{args.name}_predictions_val_zoom.png")


if __name__ == "__main__":
    main()
