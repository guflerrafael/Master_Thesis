"""Summarise the ablation runs of src/run_ablations.py (notes.md 6.7).

Per run (outputs/ablations/<experiment>_s<seed>_history.csv, finished runs only): metrics at the
best validation epoch (max mean of AP@5 and AP@10, the saved checkpoint) and the mean of the last
five validation evaluations (less sensitive to a lucky epoch). Per experiment: mean ± std over
seeds and the difference to 'ref'. Validation split only.

Outputs: outputs/ablations/summary.csv, outputs/ablations/summary.md, figures/ablations/summary.png

Run from the repository root:  python src/summarize_ablations.py [--folder ablations]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
from run_ablations import EXPERIMENTS

METRICS = ["AP@5", "AP@10", "recall@10", "recall_nonisolated@10", "median_err@5", "F1@10"]


def run_rows(folder):
    rows = []
    for h in sorted((E.ROOT / "outputs" / folder).glob("*_history.csv")):
        run = h.name[:-len("_history.csv")]
        if not (h.parent / f"{run}_done.json").exists():
            continue                                             # unfinished
        exp, seed = run.rsplit("_s", 1)
        d = pd.read_csv(h).dropna(subset=["AP@5"])
        best = d.loc[((d["AP@5"] + d["AP@10"]) / 2).idxmax()]
        last = d.tail(5)
        r = {"experiment": exp, "seed": int(seed), "best_epoch": int(best["epoch"]),
             "train_min_per_epoch": pd.read_csv(h)["train_min"].mean()}
        for m in METRICS:
            if m in d:
                r[f"{m} best"] = best[m]
                r[f"{m} last5"] = last[m].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="ablations")
    args = ap.parse_args()
    runs = run_rows(args.folder)
    if runs.empty:
        print(f"no finished runs in outputs/{args.folder}/"); return
    cols = [c for c in runs.columns if c.endswith(("best", "last5"))]
    g = runs.groupby("experiment")
    summ = pd.concat({"mean": g[cols].mean(), "std": g[cols].std()}, axis=1)
    summ.columns = [f"{m} {stat}" for stat, m in summ.columns]
    summ.insert(0, "seeds", g.size())
    summ.insert(1, "min/epoch", g["train_min_per_epoch"].mean().round(2))
    summ.insert(0, "tier", [EXPERIMENTS.get(e, (0, "", ""))[0] for e in summ.index])
    summ.insert(1, "question", [EXPERIMENTS.get(e, (0, "", "?"))[2] for e in summ.index])
    if "ref" in summ.index:
        for m in ("AP@5 best", "AP@10 best", "AP@5 last5", "AP@10 last5"):
            summ[f"Δ {m} vs ref"] = summ[f"{m} mean"] - summ.loc["ref", f"{m} mean"]
    order = [e for e in EXPERIMENTS if e in summ.index] + [e for e in summ.index if e not in EXPERIMENTS]
    summ = summ.loc[order]

    out = E.ROOT / "outputs" / args.folder
    summ.to_csv(out / "summary.csv")
    show = ["tier", "question", "seeds", "min/epoch"]
    for m in ("AP@5", "AP@10"):
        for kind in ("best", "last5"):
            summ[f"{m} {kind}"] = summ[f"{m} {kind} mean"].map("{:.3f}".format) + " ± " + summ[f"{m} {kind} std"].fillna(0).map("{:.3f}".format)
            show.append(f"{m} {kind}")
    show += [c for c in summ.columns if c.startswith("Δ")]
    md = summ[show].round(3)
    try:
        (out / "summary.md").write_text(md.to_markdown())
    except ImportError:                                          # 'tabulate' not installed
        (out / "summary.md").write_text("```\n" + md.to_string() + "\n```\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(md.to_string())
    print(f"\nwritten: outputs/{args.folder}/summary.csv, summary.md")

    # figure: AP@5 and AP@10 (best epoch) per experiment, one dot per seed, mean as bar
    fig_dir = E.ROOT / "figures" / args.folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * len(order) + 1.5), sharey=True)
    y = np.arange(len(order))
    for ax, m in zip(axes, ("AP@5 best", "AP@10 best")):
        ax.barh(y, summ[f"{m} mean"], color=E.BLUE, alpha=0.35, height=0.6)
        for k, e in enumerate(order):
            v = runs.loc[runs["experiment"] == e, m]
            ax.scatter(v, np.full(len(v), k), color=E.BLUE, s=14, zorder=3)
        if "ref" in summ.index:
            ax.axvline(summ.loc["ref", f"{m} mean"], color=E.ORANGE, ls="--", lw=1.2, label="ref (mean)")
        ax.set_title(f"{m.split()[0]} (validation, best epoch)"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, order, fontsize=8); axes[0].invert_yaxis()
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "summary.png", dpi=150); plt.close(fig)
    print(f"figure: figures/{args.folder}/summary.png")


if __name__ == "__main__":
    main()
