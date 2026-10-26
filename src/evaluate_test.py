"""Single final evaluation of the final models on the TEST split 2017-2019 (notes.md 6.16).

Evaluates every checkpoint models/<folder>/<experiment>_s<seed>.pt of the given experiments and
the nucleus-birth baseline with src/evaluate.py (k = 5 and 10 km): per seed, mean ± std over
seeds, recall by isolation / size class / year, and AP per year (2017 = extreme year, still
before the SEVIRI geo-referencing change; 2018-2019 after it, 6.11-6.12).

Safeguard: the test split is only used with --confirm-test. Without it the script evaluates the
VALIDATION split instead, so the code path can be checked without looking at test numbers.

Final models are trained first (train + val, no validation, last epoch):
  python src/run_ablations.py --only cnn3d_final conv_final --seeds 0 1 2 --folder final \\
      --extra="--train-splits train,val --no-val"
Then, once:
  python src/evaluate_test.py --confirm-test > outputs/final/test_log.txt
Outputs: outputs/final/<split>_results_{per_seed,summary}.csv, <split>_results.md,
figures/final/<split>_pr.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
from baselines import predictions as baseline_predictions
from dataset import FireData
from evaluate import evaluate
from train import get_device, load_model, predict

KEYS = ["AP", "recall (all preds)", "best F1", "P @ best F1", "R @ best F1", "false alarms / day @ best F1",
        "median error TP [km]", "nearest pred ≤ 5 km", "nearest pred median [km]"]
GROUPS = ("isolated", "size_class", "year")


def score(name, seed, preds, data, split, periods_by_year):
    rows, curves = [], {}
    for k in (5, 10):
        res, curve, _ = evaluate(preds, data, split, k, groups=GROUPS)
        r = {"method": name, "seed": seed, "k [km]": k, **{c: res[c] for c in KEYS},
             **{c: v for c, v in res.items() if c.startswith("recall |")}}
        for y, per in periods_by_year.items():
            r[f"AP | year={y}"] = evaluate(preds, data, split, k, groups=(), periods=per)[0]["AP"]
        rows.append(r); curves[k] = curve
    return rows, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=["cnn3d_final", "conv_final"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--folder", default="final")
    ap.add_argument("--confirm-test", action="store_true", help="really evaluate on the test split (do this once)")
    args = ap.parse_args()
    split = "test" if args.confirm_test else "val"
    if split == "val":
        print("NOTE: --confirm-test not given -> evaluating on the VALIDATION split (code check, no test numbers).")
    device = get_device()
    data = FireData()
    assert data.georef_fix, "the final evaluation expects the geo-referencing-corrected build (6.12)"
    periods = data.periods.loc[(data.periods["split"] == split) & (data.periods["n_det"] > 0), "period"]
    years = sorted(periods.dt.year.unique())
    periods_by_year = {y: periods[periods.dt.year == y] for y in years}

    rows, curves = [], {}
    r, c = score("baseline: nucleus birth", -1, baseline_predictions(data)["birth"], data, split, periods_by_year)
    rows += r; curves[("baseline: nucleus birth", -1)] = c
    for exp in args.experiments:
        for seed in args.seeds:
            name = f"{args.folder}/{exp}_s{seed}"
            try:
                model, cfg, ck = load_model(name, device)
            except FileNotFoundError:
                print(f"missing checkpoint models/{name}.pt, skipped"); continue
            print(f"{name}: epoch {ck['epoch']}, trained on {ck['config'].get('train_splits', 'train')}", flush=True)
            r, c = score(exp, seed, predict(model, data, periods, cfg, device), data, split, periods_by_year)
            rows += r; curves[(exp, seed)] = c

    per_seed = pd.DataFrame(rows)
    num = [c for c in per_seed.columns if c not in ("method", "seed", "k [km]")]
    g = per_seed.groupby(["method", "k [km]"], sort=False)[num]
    summary = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    summary.insert(0, "seeds", per_seed.groupby(["method", "k [km]"], sort=False).size())

    out = E.ROOT / "outputs" / args.folder
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / f"{split}_results_per_seed.csv", index=False)
    summary.to_csv(out / f"{split}_results_summary.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    main_cols = ["seeds"] + KEYS
    text = (f"Final evaluation on the {split.upper()} split ({years[0]}–{years[-1]}), mean ± std over seeds\n\n"
            + summary[main_cols].to_string() + "\n\nRecall (all predictions) by subgroup and AP by year:\n"
            + summary[[c for c in summary.columns if "|" in c]].to_string() + "\n")
    (out / f"{split}_results.md").write_text("```\n" + text + "```\n")
    print(text)

    fig_dir = E.ROOT / "figures" / args.folder
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {"baseline: nucleus birth": E.ORANGE}
    palette = iter([E.BLUE, E.AQUA, E.INK2])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, k in zip(axes, (5, 10)):
        for (name, seed), cv in curves.items():
            if name not in colours:
                colours[name] = next(palette)
            ax.plot(cv[k]["recall"], cv[k]["precision"], color=colours[name], lw=1.6 if seed < 1 else 1.0,
                    ls="--" if name.startswith("baseline") else "-", alpha=1 if seed < 1 else 0.6,
                    label=name if seed < 1 else None)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("recall"); ax.set_title(f"{split} {years[0]}–{years[-1]}, match within {k} km")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{split}_pr.png", dpi=150); plt.close(fig)
    print(f"written: outputs/{args.folder}/{split}_results*.csv/.md, figures/{args.folder}/{split}_pr.png (one curve per seed)")


if __name__ == "__main__":
    main()
