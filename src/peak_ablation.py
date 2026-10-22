"""Peak-extraction ablation without retraining (notes.md 6.7, ablation 17).

Predicts the validation probability maps of one checkpoint once, then turns them into points
with different peak-extraction settings (neighbourhood window, probability threshold, sub-pixel
refinement) and scores each with src/evaluate.py at 5 and 10 km. Validation split only.

Run from the repository root:
  python src/peak_ablation.py --name convlstm_v1 > outputs/convlstm_v1_peak_ablation.txt
"""
import argparse
import itertools

import pandas as pd

from dataset import FireData
from evaluate import evaluate
from train import get_device, load_model, maps_to_points, predict_maps

WINDOWS = (3, 5)
THRESHOLDS = (0.02, 0.05, 0.1)
SUBPIXEL = (True, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convlstm_v1")
    args = ap.parse_args()
    device = get_device()
    model, cfg, ck = load_model(args.name, device)
    data = FireData()
    periods = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    maps = predict_maps(model, data, periods, cfg, device)
    print(f"checkpoint {args.name} (epoch {ck['epoch']}); {len(maps)} validation maps")
    rows = []
    for window, thr, sub in itertools.product(WINDOWS, THRESHOLDS, SUBPIXEL):
        pts = maps_to_points(maps, data.georef_fix, window=window, thr=thr, subpixel=sub)
        r = {"window": window, "threshold": thr, "subpixel": sub, "points": len(pts)}
        for k in (5, 10):
            res, _, _ = evaluate(pts, data, "val", k)
            r.update({f"AP@{k}": res["AP"], f"recall@{k}": res["recall (all preds)"], f"F1@{k}": res["best F1"],
                      f"median_err@{k}": res["median error TP [km]"]})
        rows.append(r)
        print(r, flush=True)
    tab = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 250)
    print("\n(v1/v2 default: window 3, threshold 0.05, subpixel True)")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
