"""Architecture diagram of the ConvLSTM ignition-heatmap model (notes.md 6.1).

Top: data flow with layer sizes and parameter counts (taken from src/model.py).
Bottom: the network unrolled over the 24 hourly steps (history / target period / tail).

Second figure (detail): (a) inside one ConvLSTM cell, (b) receptive field growth through the
stacked 3x3 convolutions within one time step, (c) growth over time through the recurrence.

Run from the repository root:
  .venv/bin/python src/plot_model_diagram.py
Outputs: figures/model/convlstm_v1_diagram.png / .pdf, figures/model/convlstm_cell_detail.png / .pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import eda_figures as E
from model import IgnitionConvLSTM
from tiling_design import HISTORY_H, PERIOD_H, TAIL_H

FIG = E.ROOT / "figures" / "model"
LIGHT_BLUE, LIGHT_ORANGE, GREY = "#cde2fb", "#fde3d6", E.NEUTRAL


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def box(ax, x, y, w, h, text, face, edge=E.INK2, size=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=E.INK, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=E.INK2, lw=1.2, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=9, color=E.INK, **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)


def grid(ax, x0, y0, n, cell, highlight=None, face=LIGHT_BLUE):
    """n x n pixel grid with lower-left corner (x0, y0); highlight = list of (size, colour) squares
    centred on the middle pixel, largest first."""
    for (k, col) in highlight or []:
        o = (n - k) / 2
        ax.add_patch(Rectangle((x0 + o * cell, y0 + o * cell), k * cell, k * cell, facecolor=col, edgecolor="none"))
    for a in range(n + 1):
        ax.plot([x0, x0 + n * cell], [y0 + a * cell] * 2, color=E.BASE, lw=0.5)
        ax.plot([x0 + a * cell] * 2, [y0, y0 + n * cell], color=E.BASE, lw=0.5)
    m = (n - 1) / 2
    ax.add_patch(Rectangle((x0 + m * cell, y0 + m * cell), cell, cell, facecolor=E.INK, edgecolor="none"))


def detail_figure(p_l1, p_l2):
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
    text(ax, 0.2, 10.8, "Inside the ConvLSTM and how the receptive field grows", 13, weight="bold", ha="left", va="top")

    # ---------------- (a) one ConvLSTM cell
    text(ax, 0.2, 10.15, "(a) One ConvLSTM cell at time step t (layer 1: x = 16 encoder channels; layer 2: x = 32 channels of layer 1)",
         10, weight="bold", ha="left")
    box(ax, 0.3, 7.9, 1.6, 0.8, "input xₜ\n(16 or 32 ch)", LIGHT_BLUE, size=8.5)
    box(ax, 0.3, 6.7, 1.6, 0.8, "hidden hₜ₋₁\n(32 ch)", LIGHT_BLUE, size=8.5)
    box(ax, 2.5, 7.0, 1.3, 1.4, "concat\n(48 or 64 ch)", "white", size=8.5)
    arrow(ax, 1.9, 8.3, 2.5, 7.95); arrow(ax, 1.9, 7.1, 2.5, 7.45)
    box(ax, 4.4, 6.8, 2.2, 1.8, f"one Conv 3×3\n→ 4 × 32 = 128 ch\n\nonly spatial operation:\neach pixel sees its\n3×3 neighbourhood",
        LIGHT_ORANGE, size=8.5)
    arrow(ax, 3.8, 7.7, 4.4, 7.7)
    gates = [("i: input gate", "σ", 9.2), ("f: forget gate", "σ", 8.35), ("o: output gate", "σ", 7.5), ("g: candidate", "tanh", 6.65)]
    for name, fn, y in gates:
        box(ax, 7.3, y - 0.3, 1.9, 0.6, f"{name}\n{fn}(·)", "white", size=8)
        arrow(ax, 6.6, 7.7, 7.3, y, rad=0.0, lw=0.9)
    box(ax, 10.0, 7.95, 3.0, 1.15, "cₜ = σ(f) ⊙ cₜ₋₁ + σ(i) ⊙ tanh(g)\nkeep part of the old memory,\nwrite part of the new content",
        GREY, size=8.5)
    box(ax, 10.0, 6.35, 3.0, 1.15, "hₜ = σ(o) ⊙ tanh(cₜ)\nwhat the layer passes on", GREY, size=8.5)
    for y in (9.2, 8.35, 6.65):
        arrow(ax, 9.2, y, 10.0, 8.5, lw=0.9)
    arrow(ax, 9.2, 7.5, 10.0, 6.9, lw=0.9)
    arrow(ax, 11.5, 7.95, 11.5, 7.5, lw=0.9)
    box(ax, 10.0, 9.45, 3.0, 0.5, "memory cₜ₋₁ (32 ch, one per pixel)", LIGHT_BLUE, size=8)
    arrow(ax, 11.5, 9.45, 11.5, 9.1, lw=0.9)
    box(ax, 13.7, 7.95, 2.1, 0.8, "cₜ → step t + 1", LIGHT_BLUE, size=8.5)
    box(ax, 13.7, 6.45, 2.1, 0.95, "hₜ → step t + 1\nand → next layer / head", LIGHT_BLUE, size=8.5)
    arrow(ax, 13.0, 8.4, 13.7, 8.35); arrow(ax, 13.0, 6.9, 13.7, 6.9)
    text(ax, 11.5, 5.95, "⊙ = element-wise per pixel: no exchange between pixels", 8.5, color=E.INK2)
    text(ax, 5.5, 6.35, f"parameters: layer 1 {p_l1:,}, layer 2 {p_l2:,}\n(= weights of the gate convolution)", 8.5, color=E.INK2)

    # ---------------- (b) receptive field within one time step
    text(ax, 0.2, 5.25, "(b) Within one time step: every 3×3 convolution adds one pixel in each direction", 10, weight="bold", ha="left")
    n, cell = 11, 0.33
    shades = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5"]
    steps = [("encoder conv 1", 3), ("encoder conv 2", 5), ("ConvLSTM 1 gates", 7), ("ConvLSTM 2 gates", 9), ("head conv 3×3", 11)]
    x0, y0 = 0.6, 0.9
    hl = [(k, shades[len(steps) - 1 - j]) for j, (_, k) in enumerate(reversed(steps))]
    grid(ax, x0, y0, n, cell, hl)
    for j, (name, k) in enumerate(steps):
        yy = y0 + n * cell - 0.3 - j * 0.62
        ax.add_patch(Rectangle((x0 + n * cell + 0.35, yy - 0.16), 0.32, 0.32, facecolor=shades[j], edgecolor=E.BASE))
        text(ax, x0 + n * cell + 0.8, yy, f"{name}: {k} × {k} px", 8.5, ha="left")
    text(ax, x0 + n * cell / 2, y0 - 0.3, "black = output pixel; one pixel ≈ 3.2 × 4.5 km", 8, color=E.INK2)
    text(ax, x0 + n * cell + 0.8, y0 + 0.25, "9 × 9 px ≈ 29 × 40 km after layer 2\n(only 7 × 7 with a single ConvLSTM layer)",
         8, color=E.INK2, ha="left")

    # ---------------- (c) over time
    text(ax, 8.9, 5.25, "(c) Over time: hₜ₋₁ passes the 3×3 convolution again each step", 10, weight="bold", ha="left")
    n, cell = 9, 0.24
    for j, (label, k) in enumerate([("step t", 3), ("step t + 1", 5), ("step t + 2", 7)]):
        gx = 8.95 + j * 2.35
        grid(ax, gx, 1.55, n, cell, [(k, shades[j + 1])])
        text(ax, gx + n * cell / 2, 1.2, f"{label}: {k} × {k}", 8.5)
        if j:
            arrow(ax, gx - 0.22, 1.55 + n * cell / 2, gx - 0.05, 1.55 + n * cell / 2, color=E.BLUE)
    text(ax, 12.2, 0.55, "information from a neighbour arrives one step later, from a neighbour's neighbour two steps later …\n"
         "in theory the whole crop is reachable after 24 steps; in practice the gates damp distant / old\n"
         "signals, so the effective receptive field is much smaller (motivation for attention over time)",
         8, color=E.INK2)

    fig.savefig(FIG / "convlstm_cell_detail.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "convlstm_cell_detail.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figures: figures/model/convlstm_cell_detail.png / .pdf")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = IgnitionConvLSTM(in_ch=7)
    p_enc, p_l1, p_l2, p_head = n_params(m.encoder), n_params(m.cells[0]), n_params(m.cells[1]), n_params(m.head)
    total = n_params(m)
    T = (HISTORY_H + PERIOD_H + TAIL_H)   # hourly steps

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0.2, 9.75, f"ConvLSTM ignition-heatmap model (convlstm_v1, {total:,} parameters)",
            fontsize=13, weight="bold", va="top")

    # ---------------- top: data flow
    y0, h = 6.4, 2.2
    for k in range(6):                                   # input stack
        ax.add_patch(Rectangle((0.35 + 0.12 * k, y0 + 0.15 + 0.12 * k), 1.5, 1.5, facecolor=LIGHT_BLUE,
                               edgecolor=E.BLUE, lw=0.8))
    ax.text(1.4, y0 - 0.35, f"Input sequence\n{T} hourly steps × 7 channels × H × W", ha="center", va="top", fontsize=9)
    ax.text(1.4, y0 - 1.05, "FRP · confidence · detected\nperiod flag · time · mainland · persistent",
            ha="center", va="top", fontsize=7.5, color=E.INK2)

    box(ax, 3.0, y0, 2.1, h, f"Encoder\n(each step, shared)\n\nConv 3×3, 7→16, ReLU\nConv 3×3, 16→16, ReLU\n\n{p_enc:,} params", GREY)
    box(ax, 5.8, y0, 2.1, h, f"ConvLSTM 1\n\n32 channels\nkernel 3×3\n\n{p_l1:,} params", LIGHT_BLUE, weight="normal")
    box(ax, 8.6, y0, 2.1, h, f"ConvLSTM 2\n\n32 channels\nkernel 3×3\n\n{p_l2:,} params", LIGHT_BLUE)
    box(ax, 11.4, y0, 2.1, h, f"Head\n(final state h₂₄ only)\n\nConv 3×3, 32→32, ReLU\nConv 1×1, 32→1\n\n{p_head:,} params", LIGHT_ORANGE)
    for xa, xb in [(2.6, 3.0), (5.1, 5.8), (7.9, 8.6), (10.7, 11.4), (13.5, 14.0)]:
        arrow(ax, xa, y0 + h / 2, xb, y0 + h / 2)
    for xc in (6.85, 9.65):                              # recurrence
        arrow(ax, xc - 0.45, y0 + h, xc + 0.45, y0 + h, rad=-1.2, color=E.BLUE)
        ax.text(xc, y0 + h + 0.55, "(h, c) from t − 1", ha="center", fontsize=7.5, color=E.BLUE)

    # output heatmap (schematic)
    r, c = np.mgrid[0:40, 0:30]
    hm = np.zeros((40, 30))
    for (ci, cj) in [(12, 9), (25, 20), (31, 7)]:
        hm = np.maximum(hm, np.exp(-((r - ci) ** 2 + (c - cj) ** 2) / 8))
    ax.imshow(hm, extent=(14.05, 15.75, y0, y0 + h), cmap=E.SEQ_ORANGE, vmin=0, vmax=1, zorder=2)
    ax.add_patch(Rectangle((14.05, y0), 1.7, h, fill=False, edgecolor=E.INK2, lw=1, zorder=3))
    ax.text(14.9, y0 - 0.35, "Ignition heatmap\nsigmoid, H × W\n(schematic)", ha="center", va="top", fontsize=9)

    # training / inference notes
    box(ax, 6.4, 3.95, 4.4, 1.05, "Training\n32 × 32 crops (targets + hard negatives, random flips)\n"
        "focal loss vs. Gaussian targets (σ = 1 px), ignore weights", "white", size=8)
    box(ax, 11.2, 3.95, 4.7, 1.05, "Inference\nfull grid 177 × 106 → 3×3 peaks with p ≥ 0.05, top 50\n"
        "→ sub-pixel points → matched to ICNF within 5 / 10 km", "white", size=8)
    arrow(ax, 14.9, y0 - 1.15, 13.55, 5.0, color=E.MUTED)
    arrow(ax, 14.5, y0 - 1.15, 8.6, 5.0, color=E.MUTED, rad=0.15)

    # ---------------- bottom: unrolled in time
    ax.text(0.2, 3.85, "Unrolled over the 24 hourly steps of one sample (24 h window)", fontsize=11, weight="bold", va="top")
    x_start, dx, bw = 0.9, 0.52, 0.38
    rows = [("x", 0.35, LIGHT_BLUE), ("Enc", 1.0, GREY), ("L1", 1.65, LIGHT_BLUE), ("L2", 2.3, LIGHT_BLUE)]
    phases = [(0, HISTORY_H, f"history ({HISTORY_H} h)"), (HISTORY_H, HISTORY_H + PERIOD_H, f"target period ({PERIOD_H} h)"),
              (HISTORY_H + PERIOD_H, T, f"tail ({TAIL_H} h)")]
    for k, (a, b, label) in enumerate(phases):
        xa, xb = x_start + a * dx - 0.07, x_start + b * dx - 0.07
        ax.add_patch(Rectangle((xa, 0.15), xb - xa, 2.75, facecolor=LIGHT_ORANGE if k == 1 else "#f6f5f2",
                               edgecolor="none", zorder=0))
        ax.text((xa + xb) / 2, 3.05, label, ha="center", fontsize=8.5, color=E.INK2)
    for name, y, face in rows:
        ax.text(x_start - 0.2, y + bw / 2, name, ha="right", va="center", fontsize=8)
    for t in range(T):
        x = x_start + t * dx
        for k, (name, y, face) in enumerate(rows):
            ax.add_patch(Rectangle((x, y), bw, bw, facecolor=face, edgecolor=E.INK2, lw=0.6))
            if k:
                arrow(ax, x + bw / 2, rows[k - 1][1] + bw, x + bw / 2, y, lw=0.7)
        if t:
            for name, y, _ in rows[2:]:
                arrow(ax, x - dx + bw, y + bw / 2, x, y + bw / 2, lw=0.7, color=E.BLUE)
        if t in (0, T - 1) or (t + 1) % 6 == 0:
            ax.text(x + bw / 2, 0.05, f"{t + 1}", ha="center", va="top", fontsize=7, color=E.INK2)
    x_end = x_start + (T - 1) * dx + bw
    box(ax, x_end + 0.35, 2.15, 1.0, 0.7, "Head", LIGHT_ORANGE, size=8.5)
    arrow(ax, x_end, rows[3][1] + bw / 2, x_end + 0.35, 2.5)
    ax.imshow(hm, extent=(x_end + 1.65, x_end + 2.45, 1.95, 3.0), cmap=E.SEQ_ORANGE, vmin=0, vmax=1, zorder=2)
    ax.add_patch(Rectangle((x_end + 1.65, 1.95), 0.8, 1.05, fill=False, edgecolor=E.INK2, lw=0.8, zorder=3))
    arrow(ax, x_end + 1.35, 2.5, x_end + 1.65, 2.5)
    ax.text(x_end + 0.35, 1.8, "one heatmap per sample:\nignitions in the\ntarget period",
            ha="left", va="top", fontsize=8, color=E.INK2)
    ax.text(x_start + (T * dx) / 2, -0.3, "hour of the input window", ha="center", fontsize=8, color=E.INK2)

    fig.savefig(FIG / "convlstm_v1_diagram.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "convlstm_v1_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"parameters: encoder {p_enc:,}, ConvLSTM 1 {p_l1:,}, ConvLSTM 2 {p_l2:,}, head {p_head:,}, total {total:,}")
    print("figures: figures/model/convlstm_v1_diagram.png / .pdf")
    detail_figure(p_l1, p_l2)


if __name__ == "__main__":
    main()
