"""Evaluate a trained checkpoint on the validation split next to the reference baseline
(notes.md 6.2). The test split stays locked.

Run from the repository root:
  .venv/bin/python src/evaluate_model.py --name convlstm_v1 > outputs/convlstm_v1_eval.txt
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import eda_figures as E
from baselines import predictions as baseline_predictions
from dataset import FireData
from evaluate import evaluate
from model import IgnitionConvLSTM
from train import MODELS, get_device, predict

FIG = E.ROOT / "figures" / "model"
SPLIT = "val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convlstm_v1")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    device = get_device()
    # own checkpoint written by src/train.py (v1 stored numpy scalars in its metrics)
    ck = torch.load(MODELS / f"{args.name}.pt", map_location=device, weights_only=False)
    model = IgnitionConvLSTM(in_ch=7).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"checkpoint {args.name}: epoch {ck['epoch']}, config {ck['config']}")

    data = FireData()
    periods = data.periods.loc[(data.periods["split"] == SPLIT) & (data.periods["n_det"] > 0), "period"]
    preds = {"baseline: nucleus birth": baseline_predictions(data)["birth"],
             f"ConvLSTM ({args.name})": predict(model, data, periods, ck["config"]["step_agg"], device)}

    rows, curves = [], {}
    for name, p in preds.items():
        for k in (5, 10):
            res, curve, _ = evaluate(p, data, SPLIT, k)
            rows.append({"method": name, **res}); curves[(name, k)] = curve
    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["method", "k [km]", "targets", "AP", "recall (all preds)", "best F1", "P @ best F1", "R @ best F1",
            "false alarms / day @ best F1", "median error TP [km]", "nearest pred ≤ 5 km", "nearest pred median [km]"]
    print(f"\nValidation ({SPLIT}, 2015–2016):")
    print(tab[cols].round(3).to_string(index=False))
    print("\nRecall with all predictions, by subgroup:")
    print(tab[["method", "k [km]"] + [c for c in tab.columns if c.startswith("recall |")]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, k in zip(axes, (5, 10)):
        for (name, (c, ls)) in zip(preds, [(E.ORANGE, "--"), (E.BLUE, "-")]):
            cv = curves[(name, k)]
            a = tab.loc[(tab["method"] == name) & (tab["k [km]"] == k), "AP"].iat[0]
            ax.plot(cv["recall"], cv["precision"], color=c, ls=ls, lw=2, label=f"{name} (AP {a:.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall (share of target ignitions found)")
        ax.set_title(f"Validation 2015–2016, match within {k} km")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(FIG / f"{args.name}_pr_val.png"); plt.close(fig)
    print(f"\nfigure: figures/model/{args.name}_pr_val.png")


if __name__ == "__main__":
    main()
