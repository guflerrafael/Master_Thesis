"""Overview diagrams for the write-up.

  figures/pipeline/pipeline_overview.png / .pdf   raw files -> labels / inputs -> model -> scores
  figures/model/model_convlstm.png / .pdf         ConvLSTM (conv_final): input, layers, target, output
  figures/model/model_cnn3d.png / .pdf            3D CNN (cnn3d_final): input, layers, target, output
  figures/model/model_cnn3d_st.png / .pdf         planned space-time 3D CNN (notes 6.19): hourly output, shown with targets

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


# ---------------------------------------------------------------- planned space-time model

ST_PERIOD = pd.Timestamp("2016-08-20 06:00", tz="UTC")   # validation: two targets 0.8 cells apart, 4 h apart
ST_SIGMA_T = 1.0                                         # target width in time [h]
ST_PARAMS = "≈ 140k"                                     # prototype benchmark (notes 6.19); read from model.py once built


def time_bar(ax, x, y, w, n, a, b, h=0.17, label_txt=None):
    """n cells across width w; cells a..b-1 (the 12 h period) highlighted."""
    cw = w / n
    for k in range(n):
        ax.add_patch(Rectangle((x + k * cw, y), cw, h, facecolor=E.BLUE if a <= k < b else E.GRID,
                               edgecolor="white", lw=0.6))
    if label_txt:
        label(ax, x + w + 0.08, y + h / 2, label_txt, 7.6, E.INK2)


def st_example():
    """Crop around the close pair of ST_PERIOD: hour of first detection, hourly 3D target, current 2D target."""
    data = FireData()
    x, y2d, _, lab = data.sample(ST_PERIOD)
    tg = lab[lab["role"] == "target"].join(data.fires.set_index("fire_id")["start_utc"], on="fire_id")
    tg["t_h"] = (tg["start_utc"] - ST_PERIOD).dt.total_seconds() / 3600
    v = tg[["i_f", "j_f"]].values
    d = np.hypot(*(v[:, None] - v[None]).transpose(2, 0, 1)) + np.eye(len(v)) * 99
    a, b = np.unravel_index(d.argmin(), d.shape)
    ci, cj = (v[a] + v[b]) / 2
    i0, j0, s = int(round(ci)) - 5, int(round(cj)) - 5, 11
    det = x[:, 2] > 0
    first_h = np.where(det.any(0), det.argmax(0) / 4, np.nan)[i0:i0 + s, j0:j0 + s]
    r, c = np.mgrid[i0:i0 + s, j0:j0 + s]
    y3d = np.zeros((12, s, s), np.float32)
    for q in tg.itertuples():
        sp = np.exp(-((r - q.i_f) ** 2 + (c - q.j_f) ** 2) / 2)
        for k in range(12):
            y3d[k] = np.maximum(y3d[k], sp * np.exp(-((k + 0.5 - q.t_h) ** 2) / (2 * ST_SIGMA_T ** 2)))
    pair = tg.iloc[[a, b]].sort_values("t_h")
    return first_h, y3d, y2d[i0:i0 + s, j0:j0 + s], pair.assign(i_loc=pair["i_f"] - i0, j_loc=pair["j_f"] - j0)


def fig_model_st():
    first_h, y3d, y2d, pair = st_example()
    W, H = 18, 9.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.2, 9.45, "Planned: space–time 3D CNN (cnn3d_st): one ignition map per hour of the 12 h period",
            fontsize=14, weight="bold", va="top", color=E.INK)
    ax.text(0.2, 9.0, "Same encoder as cnn3d_final. New: a decoder brings the time axis back to hourly steps, "
                      "so the output keeps the hour of each ignition instead of pooling time away.",
            fontsize=9.5, va="top", color=E.INK2)

    # ---- 1 input
    box(ax, 0.2, 2.75, 3.5, 5.85, "1  Input x", "24 hourly steps × 7 channels\n(frp, confidence, detected, period,\n"
        "time, pt mask, persistent)\nshape 24 × 7 × H × W", SEV, SEV_EDGE, size=8.4)
    time_bar(ax, 0.35, 6.75, 3.2, 24, 6, 18)
    label(ax, 0.35, 6.55, "6 h history · 12 h period (blue) · 6 h tail", 7.6, E.INK2)
    label(ax, 0.35, 6.15, "example: 2016-08-20 06–18 UTC (validation),\n11 × 11 cells; colour = hour of first detection\n(darker = earlier), SEVIRI only", 7.6, E.MUTED, va="top")
    th = thumb(fig, ax, 0.4, 2.95, 2.45, 2.45, first_h, E.SEQ_BLUE.reversed(), 0, 24, "")
    cb = fig.colorbar(th.images[0], ax=th, fraction=0.08, pad=0.04)
    cb.set_ticks([0, 6, 12, 18, 24]); cb.ax.tick_params(labelsize=7)

    # ---- 2 network as a U (encoder down, decoder up)
    bw, bh = 2.55, 1.32
    xe, xd = 4.25, 7.65
    ye = [6.95, 5.0, 3.05]
    blocks_e = [("E1  Conv3d 3×3×3", "7 → 16 channels, ReLU", 24),
                ("E2  Conv3d, stride 2 in t", "16 → 32 channels, ReLU", 12),
                ("E3  Conv3d, stride 2 in t", "32 → 32 channels, ReLU", 6)]
    for k, (t, body, n) in enumerate(blocks_e):
        xx = xe if k < 2 else (xe + xd) / 2
        box(ax, xx, ye[k], bw, bh, t, body, SEV, SEV_EDGE, size=8.2)
        time_bar(ax, xx + 0.15, ye[k] + 0.18, bw - 0.3, n, n // 4, 3 * n // 4)
        label(ax, xx + bw / 2, ye[k] - 0.13, f"{n} steps of {24 // n} h", 7.6, SEV_EDGE, ha="center")
    blocks_d = [("D1  upsample + Conv3d", "↑ to 24 steps, joined with E1\n→ 16 channels", 24),
                ("D2  upsample + Conv3d", "↑ to 12 steps, joined with E2\n→ 32 channels", 12)]
    for k, (t, body, n) in enumerate(blocks_d):
        box(ax, xd, ye[k], bw, bh, t, body, ICNF, ICNF_EDGE, size=8.2)
        time_bar(ax, xd + 0.15, ye[k] + 0.18, bw - 0.3, n, n // 4, 3 * n // 4)
        label(ax, xd + bw / 2, ye[k] - 0.13, f"{n} steps of {24 // n} h", 7.6, ICNF_EDGE, ha="center")
    label(ax, xe, 8.55, "2  Encoder: compress time", 10, SEV_EDGE, weight="bold")
    label(ax, xd, 8.55, "3  Decoder: restore hours (new)", 10, ICNF_EDGE, weight="bold")
    xm = (xe + xd) / 2
    arrow(ax, (3.7, 6.1), (xe, ye[0] + bh / 2), SEV_EDGE)
    arrow(ax, (xe + bw / 2, ye[0] - 0.25), (xe + bw / 2, ye[1] + bh), SEV_EDGE)
    arrow(ax, (xe + bw / 2, ye[1] - 0.25), (xm + 0.4, ye[2] + bh), SEV_EDGE)
    arrow(ax, (xm + bw - 0.4, ye[2] + bh), (xd + bw / 2, ye[1] - 0.25), ICNF_EDGE)
    arrow(ax, (xd + bw / 2, ye[1] + bh), (xd + bw / 2, ye[0] - 0.25), ICNF_EDGE)
    for k in range(2):   # skip connections: fine timing from the encoder
        arrow(ax, (xe + bw, ye[k] + bh / 2), (xd, ye[k] + bh / 2), E.MUTED, lw=1.1)
        label(ax, (xe + bw + xd) / 2, ye[k] + bh / 2 + 0.17, "skip", 7.6, E.MUTED, ha="center")

    # global context
    gx, gy, gw, gh = xe, 0.75, xd + bw - xe, 1.35
    box(ax, gx, gy, gw, gh, "Global context (as in cnn3d_final)",
        "max and mean of E3 over its 6 steps → 64 channels, copied to every hour:\n"
        "each hour also 'knows' the whole 24 h window (did it keep burning? was it there before?)", JOINT, JOINT_EDGE, size=8.2)
    arrow(ax, (xm + bw / 2, ye[2] - 0.27), (xm + bw / 2, gy + gh), E.INK2)

    # ---- 4 cut to the period + head
    xh, hw = 10.75, 2.35
    box(ax, xh, ye[0], hw, bh, "Keep the period", "hours 7–18 of D1\n→ 16 × 12 × H × W", JOINT, JOINT_EDGE, size=8.2)
    time_bar(ax, xh + 0.15, ye[0] + 0.18, hw - 0.3, 24, 6, 18)
    arrow(ax, (xd + bw, ye[0] + bh / 2), (xh, ye[0] + bh / 2), ICNF_EDGE)
    box(ax, xh, 3.5, hw, 2.1, "4  Head, every hour",
        "join 16 + 64 channels\nConv3d 1×3×3 → 32, ReLU\nConv 1×1 → 1 logit\nsigmoid → p(hour, i, j)\n\n"
        f"{ST_PARAMS} parameters (now 63k)", JOINT, JOINT_EDGE, size=8.2)
    arrow(ax, (xh + hw / 2, ye[0] - 0.05), (xh + hw / 2, 5.6))
    arrow(ax, (gx + gw, gy + gh / 2), (xh + hw / 2, 3.5), E.INK2, rad=0.25)

    # ---- 5 output: 12 hourly maps
    xo, ow = 13.45, 4.35
    box(ax, xo, 3.95, ow, 4.65, "5  Output: 12 hourly maps", "", ICNF, ICNF_EDGE)
    label(ax, xo + 0.15, 8.08, f"shown: target y (Gaussian σ = 1 cell, σ_t = {ST_SIGMA_T:.0f} h)", 7.8, E.INK2)
    tw, tg_ = 0.62, 0.08
    for k in range(12):
        rr, cc = divmod(k, 6)
        x0, y0 = xo + 0.2 + cc * (tw + tg_), 6.95 - rr * 1.0
        a = thumb(fig, ax, x0, y0, tw, tw, y3d[k], E.SEQ_ORANGE, 0, 1, "")
        a.set_title(f"{6 + k:02d}–{7 + k:02d}", fontsize=7, color=E.INK2, pad=2)
    for q, name in zip(pair.itertuples(), "AB"):
        k = int(q.t_h)
        rr, cc = divmod(k, 6)
        label(ax, xo + 0.2 + cc * (tw + tg_) + tw / 2, 6.95 - rr * 1.0 - 0.15, name, 9, E.INK, ha="center", weight="bold")
    label(ax, xo + 0.15, 5.55,
          f"A starts {pair['t_h'].iat[0]:.1f} h, B {pair['t_h'].iat[1]:.1f} h into the period (UTC hours shown);\n"
          "0.8 cells apart: two separate peaks in (hour, i, j)", 7.8, E.INK2, va="top")
    label(ax, xo + 0.15, 4.85, "decode: 3D local maxima (3 h × 3 × 3 cells)\n→ points (lat, lon, hour, score)", 8.2, E.INK, va="top")
    arrow(ax, (xh + hw, 4.55), (xo, 5.2))

    # ---- comparison with the current model
    cx, cw_ = xo, ow
    box(ax, cx, 0.75, cw_, 2.85, "Compared with cnn3d_final", "", JOINT, JOINT_EDGE)
    thumb(fig, ax, cx + 0.2, 1.75, 1.1, 1.1, y2d, E.SEQ_ORANGE, 0, 1, "")
    label(ax, cx + 1.45, 2.3, "now: time is pooled away →\none 2D map: A and B merge\ninto one bump and one peak,\nso one of them is missed", 7.8, E.INK2)
    label(ax, cx + 0.2, 1.2, "new: same encoder + decoder + per-hour head;\nevery ignition also gets an hour.\n≈ 2.3× training time per step", 7.8, E.INK2)

    for ext in ("png", "pdf"):
        fig.savefig(FIG_M / f"model_cnn3d_st.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("figure: figures/model/model_cnn3d_st.png / .pdf (planned model)")


def main():
    FIG_P.mkdir(parents=True, exist_ok=True); FIG_M.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_model("convlstm")
    fig_model("cnn3d")
    fig_model_st()


if __name__ == "__main__":
    main()
