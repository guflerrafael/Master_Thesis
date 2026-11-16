"""Step 2 of the space-time ignition heatmap (notes.md 9.11): validation of a trained cnn3d_st run
against cnn3d_final and the rule-based time baselines (spacetime_checks.py, part E).

For the saved checkpoint, on validation 2015-2016 (the test split is never used):
  - the three decodings D1 (3D peaks), D2 (max over hours), D0 (auxiliary 2D head), each scored
    place-only and with the space-time protocol (tau = 1, 2, 3 h);
  - recall and time error by subgroup (isolated, in_pair, already_burning, rekindle);
  - the D1 points with the first-detection rule time instead of the model's own time (raw and with
    the training lag removed), so both times are compared on the same true positives.

Run from the repository root:
  .venv/bin/python src/spacetime_eval.py ablations_st/cnn3d_st_s0 > outputs/spacetime_eval_s0.txt
"""
import sys

import numpy as np
import pandas as pd

from baselines import first_detection_times, training_lag_h
from dataset import FireData
from evaluate import evaluate
from spacetime_checks import GROUPS, score_table
from train import get_device, load_model, predict


def group_recall(name, preds, data):
    rows = []
    for k in (5, 10):
        res, _, _ = evaluate(preds, data, "val", k, groups=GROUPS)
        rows.append({"method": name, "k": k, **{c.replace("recall | ", ""): v for c, v in res.items() if c.startswith("recall | ")}})
    return rows


def main(name):
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
    data = FireData()
    device = get_device()
    model, cfg, ck = load_model(name, device)
    print(f"{name}: checkpoint of epoch {ck.get('epoch', '?')}; stored val "
          + ", ".join(f"{k} {v:.4f}" for k, v in ck["val"].items() if k in ("AP@5", "AP@10", "AP@5_max", "AP@10_max", "AP@5_2d", "AP@10_2d")))
    per = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    pts = predict(model, data, per, cfg, device, decodes=("3d", "max", "2d"))
    lag = training_lag_h(data, "train")
    labels = {"3d": "D1 3D peaks", "max": "D2 max over hours", "2d": "D0 2D head"}
    rows, rec = [], []
    for m, p in pts.items():
        rows += score_table(labels[m], p, data)
        rec += group_recall(labels[m], p, data)
        print(f"   {labels[m]}: {len(p):,} points", flush=True)
    pf = first_detection_times(pts["3d"].drop(columns="t_h"), data)
    rows += score_table("D1 + first detection", pf, data)
    rows += score_table("D1 + first detection, lag removed", pf.assign(t_h=np.clip(pf["t_h"] - lag, 0, 12)), data)
    print(f"   D1 points without a detection within 1.5 cells: {pf['t_h'].isna().mean():.1%}; training lag {lag:+.2f} h")

    tab = pd.DataFrame(rows).set_index(["method", "k"]).round(3)
    main_cols = ["AP", "recall", "median |dt| [h]", "median dt [h]", "|dt| <= 1 h", "|dt| <= 2 h", "AP tau=1 h", "AP tau=2 h", "AP tau=3 h"]
    print("\nValidation 2015-2016:")
    print(tab[[c for c in main_cols if c in tab]].to_string())
    print("\nmedian |dt| of the hits by subgroup [h]:")
    print(tab[[c for c in tab.columns if c.startswith("|dt| ") and "<=" not in c]].to_string())
    print("\nrecall (all predictions) by subgroup:")
    print(pd.DataFrame(rec).set_index(["method", "k"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ablations_st/cnn3d_st_s0")
