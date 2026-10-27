"""Overview diagrams for the write-up.

  figures/pipeline/pipeline_overview.png / .pdf   raw files -> labels / inputs -> model -> scores
  figures/model/model_convlstm.png / .pdf         ConvLSTM (conv_final): input, layers, target, output
  figures/model/model_cnn3d.png / .pdf            3D CNN (cnn3d_final): input, layers, target, output

The model figures show real data: one validation period (2016-08-06 06 UTC, 32 x 32 crop around a
target), its target heatmap, and the prediction of the seed-0 checkpoint of each model
(ablations_georef/conv_final_s0, ablations_georef/cnn3d_final_s0).
Parameter counts are read from src/model.py.

Run from the repository root:
  .venv/bin/python src/plot_overview_diagrams.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import eda_figures as E
from dataset import FireData, crop_origin
from model import build_model
from train import DEFAULTS, features, get_device, load_model, static_crop

FIG_P = E.ROOT / "figures" / "pipeline"
FIG_M = E.ROOT / "figures" / "model"
SEV, SEV_EDGE = "#e6f0fc", E.BLUE          # SEVIRI / input side
ICNF, ICNF_EDGE = "#fdebe2", E.ORANGE      # ICNF / label side
JOINT, JOINT_EDGE = "white", E.INK2         # both
PERIOD = pd.Timestamp("2016-08-06 06:00", tz="UTC")


def n_params(cfg):
    return sum(p.numel() for p in build_model({**DEFAULTS, **cfg}).parameters())


def box(ax, x, y, w, h, title, body="", face=JOINT, edge=JOINT_EDGE, size=8.6, lw=1.1):
    """Rounded box, (x, y) = lower-left; bold title at the top, body lines below it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, lw=lw))
    ax.text(x + 0.14, y + h - 0.16, title, ha="left", va="top", fontsize=size + 0.8, weight="bold", color=E.INK)
    if body:
        ax.text(x + 0.14, y + h - 0.52, body, ha="left", va="top", fontsize=size, color=E.INK2, linespacing=1.45)


def arrow(ax, p0, p1, color=E.INK2, lw=1.2, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def label(ax, x, y, s, size=9, color=E.INK2, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=kw.pop("ha", "left"), va=kw.pop("va", "center"), **kw)


# ---------------------------------------------------------------- pipeline

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(18, 6.6))
    ax.set_xlim(0, 18); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(0.2, 6.45, "Data pipeline: SEVIRI forms the input, ICNF only selects the labels and scores the output",
            fontsize=14, weight="bold", va="top", color=E.INK)

    w, h, gap = 2.62, 1.85, 0.34
    xs = [0.2 + k * (w + gap) for k in range(6)]
    y_top, y_bot = 3.75, 0.7
    label(ax, 0.2, 5.85, "SATELLITE INPUT", 9.5, SEV_EDGE, weight="bold")
    label(ax, 0.2, 0.35, "GROUND TRUTH", 9.5, ICNF_EDGE, weight="bold")

    # SEVIRI lane
    box(ax, xs[0], y_top, w, h, "SEVIRI FRP extract", "569,019 fire pixels\n15-min scans, 2004–2019\nFRP [MW], confidence", SEV, SEV_EDGE)
    box(ax, xs[1], y_top, w, h, "Clean", "896 duplicates removed\nsorted, times in UTC\nno confidence filter", SEV, SEV_EDGE)
    box(ax, xs[2], y_top, w, h, "Native SEVIRI grid", "lat/lon → cell (i, j), exact\n177 × 106 cells\n≈ 3.2 × 4.5 km per cell", SEV, SEV_EDGE)
    # ICNF lane
    box(ax, xs[0], y_bot, w, h, "ICNF fire register", "813,384 fires, 1980–2023\nstart, burned area,\nignition point, local time", ICNF, ICNF_EDGE)
    box(ax, xs[1], y_bot, w, h, "Clean", "duplicates removed\nbad ends / positions flagged\nlocal time → UTC", ICNF, ICNF_EDGE)
    box(ax, xs[2], y_bot, w, h, "Candidate fires", "≥ 10 ha, 2004–2019: 10,911\nposition → sensor frame\n(½ cell NW before Dec 2017)", ICNF, ICNF_EDGE)
    for a, b in ((0, 1), (1, 2)):
        arrow(ax, (xs[a] + w, y_top + h / 2), (xs[b], y_top + h / 2), SEV_EDGE)
        arrow(ax, (xs[a] + w, y_bot + h / 2), (xs[b], y_bot + h / 2), ICNF_EDGE)

    # merge: sample index
    box(ax, xs[3], y_bot, w, y_top + h - y_bot, "Sample index",
        "one sample = one 12 h period\n(06–18 / 18–06 UTC)\n\n"
        "seen: detection ≤ 5 km,\n   start − 1 h … + 6 h\n\n"
        "target: ≥ 50 ha and seen\n   2,821 fires\nignore: 10–50 ha, unseen,\n   ± 1 h period boundary\nbackground: everything else\n\n"
        "split by year: train 2004–14,\nval 2015–16, test 2017–19", JOINT, JOINT_EDGE)
    arrow(ax, (xs[2] + w, y_top + h / 2), (xs[3], y_top + h / 2), SEV_EDGE)
    arrow(ax, (xs[2] + w, y_bot + h / 2), (xs[3], y_bot + h / 2), ICNF_EDGE)

    # sample tensors
    box(ax, xs[4], y_top, w, h, "Input x", "raw detections, 24 h window\n6 h history · 12 h · 6 h tail\n24 hourly steps × 7 channels", SEV, SEV_EDGE)
    box(ax, xs[4], y_bot, w, h, "Target y, weight w", "Gaussian σ = 1 cell / target\nw = 0 outside Portugal and\n≤ 2 cells of an ignore fire", ICNF, ICNF_EDGE)
    arrow(ax, (xs[3] + w, y_top + h / 2), (xs[4], y_top + h / 2), SEV_EDGE)
    arrow(ax, (xs[3] + w, y_bot + h / 2), (xs[4], y_bot + h / 2), ICNF_EDGE)

    # model + evaluation
    box(ax, xs[5], y_top, w, h, "Model", "train: 32 × 32 crops\n(targets + hard negatives)\npredict: full-grid heatmap", JOINT, JOINT_EDGE)
    box(ax, xs[5], y_bot, w, h, "Evaluation", "peaks → points (true frame)\nmatched to ICNF ≤ 5 / 10 km\nAP, recall, F1, error", JOINT, JOINT_EDGE)
    arrow(ax, (xs[4] + w, y_top + h / 2), (xs[5], y_top + h / 2))
    arrow(ax, (xs[4] + w, y_bot + h - 0.3), (xs[5] + 0.45, y_top), ICNF_EDGE)
    label(ax, xs[4] + w + 0.05, (y_top + y_bot + h) / 2 + 0.25, "loss", 8, ICNF_EDGE)
    arrow(ax, (xs[5] + w / 2, y_top), (xs[5] + w / 2, y_bot + h))
    label(ax, xs[5] + w / 2 + 0.08, (y_top + y_bot + h) / 2, "points", 8)

    label(ax, xs[0], (y_top + y_bot + h) / 2,
          "side branch, not in front of the model: cluster_hotspots.py → events / nuclei\n"
          "→ nucleus-birth baseline and the persistent-pixel flag", 8, E.MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_P / f"pipeline_overview.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("figure: figures/pipeline/pipeline_overview.png / .pdf")


# ---------------------------------------------------------------- models

def example_crop(ckpt):
    """32 x 32 crop around the 95 ha target of the example period: input, target, prediction of ckpt."""
    data = FireData()
    lab = data._lab[PERIOD]
    q = lab[(lab["role"] == "target")].iloc[(lab[lab["role"] == "target"]["fire_id"].map(data._area) - 95).abs().argmin()]
    crop = crop_origin(q.i_f, q.j_f, 32)
    x, y, w, labc = data.sample(PERIOD, crop)
    f = features(x, static_crop(data, crop), DEFAULTS)
    device = get_device()
    model, cfg, _ = load_model(ckpt, device)
    xf, _, _, _ = data.sample(PERIOD)
    ff = features(xf, static_crop(data), cfg)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(ff[None]).to(device))).cpu().numpy()[0]
    i0, j0, s = crop
    return f, y, w, p[i0:i0 + s, j0:j0 + s], labc


def thumb(fig, ax0, x, y, w, h, img, cmap, vmin, vmax, title, marks=None):
    """Image axes placed at data coordinates (x, y, w, h) of the main axes ax0."""
    (fx0, fy0), (fx1, fy1) = fig.transFigure.inverted().transform(ax0.transData.transform([(x, y), (x + w, y + h)]))
    ax = fig.add_axes([fx0, fy0, fx1 - fx0, fy1 - fy0])
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if marks is not None:
        ax.scatter(marks[0], marks[1], marker="+", s=45, lw=1.4, color=E.INK)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(E.BASE)
    ax.set_title(title, fontsize=8, color=E.INK2, pad=3)
    return ax


MODELS = {
    "convlstm": dict(
        ckpt="ablations_georef/conv_final_s0", cfg=dict(hidden=16, readout="max"), out="model_convlstm",
        title="ConvLSTM (conv_final): a recurrent memory per cell, read out as the maximum over 24 hours",
        name="ConvLSTM  (conv_final)",
        blocks=[("Encoder, every hour", "Conv 3×3, 7 → 16, ReLU\nConv 3×3, 16 → 16, ReLU\nsame weights for\nall 24 steps", "24 × 16 × H × W"),
                ("ConvLSTM layer 1", "16 hidden channels\n3×3 gate convolution\nhour 1 → 24,\nstate starts at 0", "24 × 16 × H × W"),
                ("ConvLSTM layer 2", "16 hidden channels\nreads layer 1's h\nper hour", "24 × 16 × H × W"),
                ("Max over time", "max of layer-2 h\nover all 24 steps", "16 × H × W")],
        loop=(1, 2),
        detail=("Inside one ConvLSTM cell, at hour t (one 3×3 convolution, then element-wise updates per cell):",
                "[i, f, o, g] = Conv3×3([xₜ, hₜ₋₁])      cₜ = σ(f) ⊙ cₜ₋₁ + σ(i) ⊙ tanh(g)      hₜ = σ(o) ⊙ tanh(cₜ)",
                "Every cell keeps its own memory c; neighbours enter only through the 3×3 convolution, so context spreads about one cell per hour and layer.\n"
                "The reference ConvLSTM (ref) uses 2 × 32 channels and reads out only the last hour; conv_final reads the maximum, which keeps early hours."),
        train="trained with 2 hard negatives per positive"),
    "cnn3d": dict(
        ckpt="ablations_georef/cnn3d_final_s0", cfg=dict(arch="cnn3d"), out="model_cnn3d",
        title="3D CNN (cnn3d_final, best model): convolutions over hours and cells, no recurrence",
        name="3D CNN  (cnn3d_final)",
        blocks=[("Conv3d 3×3×3", "7 → 16 channels, ReLU\n3 hours × 3 × 3 cells\nper output", "16 × 24 × H × W"),
                ("Conv3d, stride 2 in t", "16 → 32 channels, ReLU\ntime 24 → 12 steps", "32 × 12 × H × W"),
                ("Conv3d, stride 2 in t", "32 → 32 channels, ReLU\ntime 12 → 6 steps", "32 × 6 × H × W"),
                ("Pool over time", "max and mean of\nthe 6 steps,\nconcatenated", "64 × H × W")],
        loop=None,
        detail=("What each output step sees (receptive field of the three Conv3d layers):",
                "space: 3 → 5 → 7 cells (9 × 9 after the head)      time: 3 → 5 → 9 hours, sampled every 4 hours",
                "All hours enter together; there is no state carried from hour to hour. Max and mean over the 6 steps then\n"
                "combine the whole 24 h window: the maximum keeps the strongest event, the mean how long it lasted."),
        train="trained with 2 hard negatives per positive"),
}


def fig_model(kind):
    M = MODELS[kind]
    f, y, w, p, labc = example_crop(M["ckpt"])
    tg = labc[labc["role"] == "target"]
    marks = (tg["j_loc"].values, tg["i_loc"].values)
    n_par = n_params(M["cfg"])

    W, Y0, H = 18, 0.45, 7.3
    fig = plt.figure(figsize=(W, H - Y0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.axis("off")
    ax.text(0.2, 7.15, M["title"], fontsize=14, weight="bold", va="top", color=E.INK)

    # ---- 1 input
    box(ax, 0.2, 0.6, 3.9, 5.95, "1  Input x  (one 12 h period)",
        "24 hourly steps = max of four 15-min slots\n"
        "   6 h history · 12 h period · 6 h tail\n\n"
        "7 channels per step and cell\n"
        "   frp           log(1 + FRP) / log(1 + 1100)\n"
        "   confidence    confidence / 100\n"
        "   detected      1 if detected in the hour\n"
        "   period        1 inside the 12 h period\n"
        "   time          step / 23\n"
        "   pt            mainland Portugal mask\n"
        "   persistent    off-season pixel flag\n\n"
        "shape 24 × 7 × H × W\ntraining: H = W = 32 (crops)\ninference: 177 × 106 (full grid)", SEV, SEV_EDGE, size=8.6)
    for k, (a, b, name) in enumerate(((0, 6, "history"), (6, 18, "period"), (18, 24, "tail"))):
        m = f[a:b, 0].max(0)
        thumb(fig, ax, 0.38 + k * 1.22, 0.85, 1.12, 1.12, np.where(m > 0, m, np.nan), E.SEQ_BLUE, 0.3, 1, name)
    label(ax, 0.38, 2.5, "example crop (32 × 32 cells): max frp per part", 8, E.INK2)

    # ---- 2 network blocks
    bw, bh, bg = 1.95, 1.35, 0.38
    xb = [4.65 + k * (bw + bg) for k in range(4)]
    yb = 4.3
    label(ax, xb[0], 6.35, f"2  {M['name']}: {n_par:,} parameters", 10.5, E.INK, weight="bold")
    for k, (t, body, shape) in enumerate(M["blocks"]):
        box(ax, xb[k], yb, bw, bh, t, body, SEV, SEV_EDGE, size=8.4)
        label(ax, xb[k] + bw / 2, yb - 0.22, f"→ {shape}", 8, SEV_EDGE, ha="center")
        if k:
            arrow(ax, (xb[k - 1] + bw, yb + bh / 2), (xb[k], yb + bh / 2), SEV_EDGE)
    if M["loop"]:
        a, b = M["loop"]
        for k in (a, b):
            ax.add_patch(FancyArrowPatch((xb[k] + 0.75 * bw, yb + bh), (xb[k] + 0.25 * bw, yb + bh), arrowstyle="-|>",
                                         mutation_scale=10, color=SEV_EDGE, lw=1.1, connectionstyle="arc3,rad=0.7"))
        label(ax, (xb[a] + xb[b] + bw) / 2, yb + bh + 0.4, "h, c carried to the next hour", 8, SEV_EDGE, ha="center")
    arrow(ax, (4.1, 3.6), (xb[0], yb + bh / 2), SEV_EDGE)

    # detail strip
    t1, t2, t3 = M["detail"]
    ax.add_patch(FancyBboxPatch((xb[0], 0.6), xb[3] + bw - xb[0], 2.8, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=E.NEUTRAL, edgecolor="none"))
    label(ax, xb[0] + 0.15, 3.1, t1, 8.8, E.INK, weight="bold")
    label(ax, xb[0] + 0.15, 2.5, t2, 10, E.INK)
    label(ax, xb[0] + 0.15, 1.6, t3, 8.3, E.INK2, linespacing=1.5)
    label(ax, xb[0] + 0.15, 0.85, M["train"], 8.3, E.MUTED)

    # ---- 3 head
    xh, hw = 14.0, 1.75
    box(ax, xh, 4.25, hw, 1.45, "3  Head", "Conv 3×3 → 32,\n   ReLU\nConv 1×1 → 1 logit\nsigmoid → p", JOINT, JOINT_EDGE, size=8.4)
    arrow(ax, (xb[3] + bw, yb + bh / 2), (xh, yb + bh / 2))
    label(ax, xh, 3.95, "p = probability that an\nICNF fire ≥ 50 ha started\nin this cell during the\n12 h period", 8, E.INK2, va="top")

    # ---- 4 target, 5 output
    xt, tw = 16.1, 1.75
    box(ax, xt, 4.15, tw, 2.4, "4  Target y", "", ICNF, ICNF_EDGE)
    thumb(fig, ax, xt + 0.25, 4.65, 1.25, 1.25, y, E.SEQ_ORANGE, 0, 1, "σ = 1 cell", marks)
    label(ax, xt + 0.12, 4.38, "focal loss vs. p", 8, E.INK2)
    arrow(ax, (xh + hw, 5.35), (xt, 5.35), ICNF_EDGE, style="<|-|>")
    box(ax, xt, 0.6, tw, 3.2, "5  Output p", "", JOINT, JOINT_EDGE)
    thumb(fig, ax, xt + 0.25, 1.75, 1.25, 1.25, p, E.SEQ_ORANGE, 0, 1, "prediction", marks)
    label(ax, xt + 0.12, 1.15, "3 × 3 peaks, p ≥ 0.05\n→ sub-pixel points\n(lat, lon, score)", 7.8, E.INK2)
    arrow(ax, (xh + hw, 4.45), (xt, 3.45))
    label(ax, xh, 1.0, "example: 2016-08-06\n06–18 UTC (validation)\n+ = target ignition", 7.8, E.MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(FIG_M / f"{M['out']}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"figure: figures/model/{M['out']}.png / .pdf ({n_par:,} parameters)")


def main():
    FIG_P.mkdir(parents=True, exist_ok=True); FIG_M.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_model("convlstm")
    fig_model("cnn3d")


if __name__ == "__main__":
    main()
