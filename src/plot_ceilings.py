"""Summary figures of the ceilings and error analysis (notes.md 9.21), from the tables of src/error_analysis.py,
plus a labels-only projection to a 3x finer grid (the 1 km sampling of MTG FCI at 3.8 um vs 3 km of SEVIRI).

  ceilings_radius.png   recall (all points) and AP vs the matching radius k: pixel quantisation and the oracle
                        nearest detected cell as bounds, cnn3d_st, cnn3d_final, nucleus birth (val and test)
  ceilings_ap.png       AP @ 10 km when classes of false alarms are counted as neutral, cumulatively
  ceilings_errors.png   false alarms at the operating point by nearest ICNF evidence, misses by primary reason
  ceilings_signal.png   cnn3d_st recall at the operating point vs with all points, by signal and context (val)
Projection (printed, no model): in-pair targets with their own peak in a 2D target map at SEVIRI resolution
and on a grid 3x finer (sigma = 1 fine cell); pixel quantisation on both grids. All non-test years.

Run from the repository root (after error_analysis.py for val and test):
  .venv/bin/python src/plot_ceilings.py > outputs/error_analysis/ceilings_projection.txt
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
from dataset import FireData
from train import heatmap_to_points

OUT = E.ROOT / "outputs" / "error_analysis"
FIG = E.ROOT / "figures" / "error_analysis"
COL = {"cnn3d_st": E.BLUE, "cnn3d_final": E.ORANGE, "nucleus birth": E.AQUA}
NAME = {"val": "validation 2015–2016 (dev runs)", "test": "test 2024–2025 (final runs, descriptive)"}


def style(ax):
    ax.grid(True, color=E.GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(E.BASE)
    ax.tick_params(colors=E.INK2, labelsize=8)


def fig_radius():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True, layout="constrained")
    for c, split in enumerate(("val", "test")):
        r = pd.read_csv(OUT / f"{split}_radius.csv")
        for row, metric in enumerate(("recall", "AP")):
            ax = axes[row, c]; style(ax)
            if metric == "recall":
                for m, ls, lab in (("pixel quantisation (bound)", ":", "bound: pixel quantisation (ICNF point -> its cell centre)"),
                                   ("oracle: nearest detected cell", "--", "bound: oracle, nearest detected cell centre")):
                    g = r[r["method"] == m]
                    ax.plot(g["k"], g[metric], color=E.INK2, ls=ls, lw=1.6, label=lab)
                ax.legend(fontsize=7.5, frameon=False, loc="lower right")
            ends = []
            for m, colr in COL.items():
                g = r[r["method"] == m]
                ax.plot(g["k"], g[metric], color=colr, lw=2, marker="o", ms=4)
                ends.append([m, g[metric].iat[-1]])
            ends.sort(key=lambda e: -e[1])
            for n in range(1, len(ends)):                       # keep direct labels apart
                ends[n][1] = min(ends[n][1], ends[n - 1][1] - 0.045)
            for m, yv in ends:
                ax.annotate(m, (15, yv), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=E.INK, va="center")
            ax.axvspan(0, 1.5, color=E.NEUTRAL, zorder=0)
            ax.set_ylim(0, 1.02); ax.set_xlim(0, 17.5)
            ax.set_ylabel("recall with all points" if metric == "recall" else "average precision", fontsize=9)
            if row == 0:
                ax.set_title(NAME[split], fontsize=10, loc="left")
            if row == 1:
                ax.set_xlabel("matching radius k [km] (shaded: below the 1.5 km quantisation median)", fontsize=8.5)
    fig.suptitle("Localisation ceiling: how close predictions can get to the ICNF point at SEVIRI resolution", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "ceilings_radius.png", dpi=150); plt.close(fig)


def fig_ap():
    stages = ["actual", "duplicates neutral", "+ label definition neutral", "+ border / persistent neutral",
              "every false alarm neutral (= recall, all points)"]
    short = ["actual AP", "+ duplicates\nneutral", "+ rekindles, earlier,\nsmall, tail fires", "+ border /\npersistent", "perfect\nprecision"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, layout="constrained")
    for ax, split in zip(axes, ("val", "test")):
        style(ax)
        t = pd.read_csv(OUT / f"{split}_ap_ceiling.csv")
        t = t[t["k"] == 10]
        x = np.arange(len(stages))
        for n, m in enumerate(("cnn3d_st", "cnn3d_final")):
            v = t.loc[t["method"] == m, stages].values[0]
            ax.bar(x + (n - 0.5) * 0.38, v, width=0.36, color=COL[m], label=m, edgecolor="white", lw=2)
            for xi, vi in zip(x, v):
                ax.text(xi + (n - 0.5) * 0.38, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7, color=E.INK2)
        ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
        ax.set_ylim(0, 1.05); ax.set_title(NAME[split], fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper left", frameon=False)
    axes[0].set_ylabel("AP @ 10 km", fontsize=9)
    fig.suptitle("AP left on the table: false alarms counted as neutral, cumulatively (mean of 3 seeds)", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "ceilings_ap.png", dpi=150); plt.close(fig)


def fig_errors():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    # false alarms at the operating point
    ax = axes[0]; style(ax)
    classes = ["earlier fire >= 10 ha", "rekindle record", "fire < 10 ha", "duplicate", "fire in the tail", "border cell",
               "persistent cell", "no ICNF evidence"]
    labels = ["ICNF fire ≥ 10 ha started 1–72 h before", "ICNF rekindle record", "ICNF fire < 10 ha", "duplicate of a found fire",
              "ICNF fire starting in the 6 h tail", "border cell (Spain)", "persistent cell", "no ICNF record ≤ 10 km"]
    y = np.arange(len(classes))[::-1]
    for n, split in enumerate(("val", "test")):
        f = pd.read_csv(OUT / f"{split}_fp_classes.csv")
        f = f[(f["method"] == "cnn3d_st") & (f["k"] == 10) & (f["at"] == "best F1")].groupby("cls")["share"].mean()
        v = [f.get(c, 0) for c in classes]
        ax.barh(y + (0.5 - n) * 0.38, v, height=0.36, color=(E.BLUE, E.ORANGE)[n], label=NAME[split].split(" (")[0], edgecolor="white", lw=2)
        for yi, vi in zip(y, v):
            ax.text(vi + 0.005, yi + (0.5 - n) * 0.38, f"{vi:.0%}", va="center", fontsize=7, color=E.INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("share of the false alarms at best F1 @ 10 km (cnn3d_st, mean of 3 seeds)", fontsize=8.5)
    ax.set_title("What the false alarms are (nearest ICNF evidence, first that applies)", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_xlim(0, 0.55)
    # misses by primary reason, parsed from the text outputs
    ax = axes[1]; style(ax)
    rows = {}
    for split in ("val", "test"):
        txt = (OUT / f"{split}.txt").read_text().splitlines()
        k = next(i for i, l in enumerate(txt) if "misses at the operating point by primary reason" in l)
        for l in txt[k + 1:k + 8]:                      # "<reason>   <count>  (<pct> %)"
            left = l.rsplit("  (", 1)[0].rstrip()
            name, count = left.rsplit(None, 1)
            rows.setdefault(name.strip(), {})[split] = float(count)
    order = list(rows)
    yy = np.arange(len(order))[::-1]
    for n, split in enumerate(("val", "test")):
        tot = sum(rows[o].get(split, 0) for o in order)
        v = [rows[o].get(split, 0) / tot for o in order]
        ax.barh(yy + (0.5 - n) * 0.38, v, height=0.36, color=(E.BLUE, E.ORANGE)[n], edgecolor="white", lw=2,
                label=f"{NAME[split].split(' (')[0]} ({tot:.0f} misses / seed)")
        for yi, vi in zip(yy, v):
            ax.text(vi + 0.005, yi + (0.5 - n) * 0.38, f"{vi:.0%}", va="center", fontsize=7, color=E.INK2)
    wrap = {"weak signal (<= 5 detections in 6 h or FRE in the lowest quartile)": "weak signal (≤ 5 detections in 6 h\nor FRE in the lowest quartile)",
            "next to an active fire (already burning / seen via an older nucleus)": "next to an active fire",
            "close pair (<= 2 cells, same period)": "close pair (≤ 2 cells, same period)",
            "other (strong, fresh signal, scored low)": "other (strong, fresh signal)",
            "detected > 2 h after the ICNF start": "first detected > 2 h after the start"}
    ax.set_yticks(yy); ax.set_yticklabels([wrap.get(o, o) for o in order], fontsize=8)
    ax.set_xlabel("share of the targets missed at best F1 @ 10 km (cnn3d_st)", fontsize=8.5)
    ax.set_title("Why targets are missed at the operating point (first reason that applies)", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(FIG / "ceilings_errors.png", dpi=150); plt.close(fig)


def fig_signal():
    """Recall of cnn3d_st at the operating point and with all points, by signal / context (validation, dev runs)."""
    txt = (OUT / "val.txt").read_text().splitlines()
    k = next(i for i, l in enumerate(txt) if "recall at the operating point / with all points, by subgroup" in l)
    rows, cur = [], None
    for l in txt[k + 2:]:
        if l.startswith("   FRE quartile edges"):
            break
        parts = l.split()
        vals = [float(v) for v in parts[-4:]]
        lead = l[:41].strip()
        if lead:
            cur = lead
        rows.append(dict(group=cur, bin=" ".join(parts[len(lead.split()) if lead else 0:-4]), n=vals[0], op=vals[1], all=vals[2]))
    r = pd.DataFrame(rows)
    pick = {"FRE <= 1.5 cells in the window [GJ]": ["q1 (low)", "q2", "q3", "q4 (high)"],
            "detections <= 1.5 cells in the first 6 h": ["1-2", "3-5", "6-10", "11-20", "> 20"],
            "size_class": ["50–100", "100–500", "500–1k", "≥1k"],
            "ICNF start in the period [h]": ["[0, 3)", "[3, 6)", "[6, 9)", "[9, 12)"]}
    titles = ["FRE near the target in the window", "detections ≤ 1.5 cells in the first 6 h", "burned area [ha]", "ICNF start, hours into the period"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True, layout="constrained")
    for ax, (g, bins), title in zip(axes, pick.items(), titles):
        style(ax)
        s = r[r["group"] == g].set_index("bin").reindex(bins)
        x = np.arange(len(bins))
        ax.bar(x - 0.19, s["all"], width=0.36, color=E.NEUTRAL, edgecolor=E.BASE, label="with all points")
        ax.bar(x + 0.19, s["op"], width=0.36, color=E.BLUE, edgecolor="white", label="at best F1 @ 10 km")
        for xi, (a, b, n) in enumerate(zip(s["all"], s["op"], s["n"])):
            ax.text(xi + 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=7, color=E.INK2)
            ax.text(xi, -0.11, f"n={n:.0f}", ha="center", fontsize=6.5, color=E.MUTED, transform=ax.get_xaxis_transform())
        ax.set_xticks(x); ax.set_xticklabels(bins, fontsize=7.5)
        ax.set_title(title, fontsize=9, loc="left"); ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("recall of cnn3d_st (validation)", fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("The fires are found, but weak or busy ones score below the threshold (validation, dev runs, 3 seeds pooled)", fontsize=11, x=0.01, ha="left")
    fig.savefig(FIG / "ceilings_signal.png", dpi=150); plt.close(fig)


def own_peaks(tg, scale):
    """Share of targets of one period with their own 2D peak on a grid `scale` x finer (sigma = 1 fine cell)."""
    i, j = tg["i_f"].values * scale, tg["j_f"].values * scale
    i0, j0 = int(np.floor(i.min())) - 6, int(np.floor(j.min())) - 6
    h, w = int(np.ceil(i.max())) - i0 + 7, int(np.ceil(j.max())) - j0 + 7
    r, c = np.mgrid[0:h, 0:w]
    y = np.zeros((h, w), np.float32)
    for a, b in zip(i - i0, j - j0):
        y = np.maximum(y, np.exp(-((r - a) ** 2 + (c - b) ** 2) / 2))
    fi, fj, s = heatmap_to_points(y, thr=0.3)
    free, own = np.ones(len(i), bool), np.zeros(len(i), bool)
    for a, b in zip(fi, fj):
        d = np.hypot(i - i0 - a, j - j0 - b)
        ok = free & (d <= 1.5)
        if ok.any():
            m = int(np.where(ok, d, np.inf).argmin()); free[m] = False; own[m] = True
    return own


def projection():
    data = FireData()
    per = set(data.periods.loc[data.periods["split"].isin(["train", "val", "train2", "train3"]), "period"])
    tg = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(per)].copy()
    rows = []
    for p, g in tg.groupby("period"):
        if len(g) < 2:
            continue
        v = g[["i_f", "j_f"]].values
        d = np.hypot(v[:, None, 0] - v[None, :, 0], v[:, None, 1] - v[None, :, 1]) + np.eye(len(v)) * 1e9
        pair = (d <= 2.0).any(1)
        if not pair.any():
            continue
        for scale in (1, 3):
            own = own_peaks(g, scale)
            rows += [dict(scale=scale, own=o) for o, q in zip(own, pair) if q]
    r = pd.DataFrame(rows)
    print("Labels-only projection, all non-test years (no model):")
    for scale, name in ((1, "SEVIRI grid (3 km sampling)"), (3, "3x finer grid (1 km, MTG FCI 3.8 um)")):
        print(f"   in-pair targets (<= 2 SEVIRI cells apart) with their own peak in ONE 2D map, {name}: "
              f"{r.loc[r['scale'] == scale, 'own'].mean():.1%} of {int((r['scale'] == scale).sum())}")
    dev = pd.read_csv(OUT / "dev_targets.csv")
    print(f"   pixel quantisation, median distance ICNF point -> cell centre: SEVIRI {dev['quant_km'].median():.2f} km; "
          f"a 3x finer grid with the same geometry {dev['quant_km'].median() / 3:.2f} km (scales with the cell size)")
    one = pd.read_csv(OUT / "dev_offset_axes.csv")
    sd = one.set_index("axis")["remaining_sd_km"]
    rng = np.random.default_rng(0)
    for name, f in (("upper bound (remaining spread of A2)", 1.0), ("half of it", 0.5)):
        e = rng.normal(0, sd["east"] * f, 200000); n = rng.normal(0, sd["south"] * f, 200000)
        print(f"   if the ICNF position error had per-axis SD {sd['east'] * f:.2f} / {sd['south'] * f:.2f} km ({name}), a perfect "
              f"estimate would still be a median {np.median(np.hypot(e, n)):.2f} km from the ICNF point")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_radius(); fig_ap(); fig_errors(); fig_signal()
    projection()
    print("figures: figures/error_analysis/ceilings_{radius,ap,errors,signal}.png")
