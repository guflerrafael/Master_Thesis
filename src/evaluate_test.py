"""Single final evaluation of the final models on the TEST split (notes.md 6.16; split in tiling_design.SPLIT).

Evaluates every checkpoint models/<folder>/<experiment>_s<seed>.pt of the given experiments, for every
given folder, and the nucleus-birth baseline with src/evaluate.py (k = 5 and 10 km): per seed, mean ±
std over seeds, recall by subgroup (isolation, size class, year; with the SGIF fields also rekindle,
cause type, FWI quartile, land cover) and AP per year, for the default protocol and for the second
protocol with rekindle records neutral (notes.md 8.4).

Ignition time (notes.md 9.9-9.16): for the true positives @ 5 / 10 km, the time error against the ICNF
start (median |dt|, share within 1 / 2 h, median |dt| by subgroup) and the space-time AP (tau = 1, 2,
3 h). Time variants per method:
  model      - the model's own time: cnn3d_st (D1, 3D peaks); the nucleus-birth baseline (birth time)
  rule       - first-detection rule on the method's points, training lag of 'train' removed; for every
               model, so the 2D models get a time too and cnn3d_st is compared on the same points
  combined   - space-time models only: rule time where the place was quiet >= COMBINED_QUIET_H before
               its first detection and rule and model agree within COMBINED_AGREE_H, model time
               otherwise (baselines.combined_times; fixed on validation, notes.md 9.18)
Recall also by in_pair / already_burning (target context, notes.md 9.2).

The test of notes.md 6.17 (2017-2019) is now the training block 'train2'; its results stay in
outputs/final/ and are never overwritten (--out must not be 'final').

Safeguard: the test split is only used with --confirm-test. Without it the script evaluates the
VALIDATION split instead, so the code path can be checked without looking at test numbers.

Code check on validation with the all-years runs (notes.md 9.17), then the single test (E4):
  .venv/bin/python src/evaluate_test.py --folders ablations_all --experiments cnn3d_st_all cnn3d_final_all --out final_smoke
  .venv/bin/python src/evaluate_test.py --folders final final_2023 --out final_2023 --confirm-test
Outputs: outputs/<out>/<split>_results_{per_seed,summary}.csv, <split>_time_{per_seed,summary}.csv,
<split>_results.md, figures/<out>/<split>_pr.png
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
from baselines import combined_times, first_detection_times, time_evidence, training_lag_h
from baselines import predictions as baseline_predictions
from dataset import FireData
from evaluate import CONTEXT_GROUPS, REPORT_GROUPS, evaluate
from train import get_device, load_model, predict

KEYS = ["AP", "recall (all preds)", "best F1", "P @ best F1", "R @ best F1", "false alarms / day @ best F1",
        "median error TP [km]", "nearest pred ≤ 5 km", "nearest pred median [km]"]
GROUPS = ("isolated", "size_class", "year") + CONTEXT_GROUPS
TIME_GROUPS = ("isolated", "in_pair", "already_burning", "rekindle")
COMBINED_QUIET_H, COMBINED_AGREE_H = 12, 1.5      # chosen on validation 2015-2016 (notes.md 9.18)
TIME_KEYS = ["median |dt| TP [h]", "median dt TP [h]", "|dt| <= 1 h", "|dt| <= 2 h"]


def score(name, seed, preds, data, split, periods_by_year):
    groups = GROUPS + (REPORT_GROUPS if "rekindle" in data.fires else ())
    rows, curves = [], {}
    for k in (5, 10):
        res, curve, _ = evaluate(preds, data, split, k, groups=groups)
        r = {"method": name, "seed": seed, "k [km]": k, **{c: res[c] for c in KEYS},
             **{c: v for c, v in res.items() if c.startswith("recall |")}}
        for y, per in periods_by_year.items():
            r[f"AP | year={y}"] = evaluate(preds, data, split, k, groups=(), periods=per)[0]["AP"]
        if len(data.rekindles):
            r["AP | rekindles neutral"] = evaluate(preds, data, split, k, groups=(), neutral_rekindles=True)[0]["AP"]
        rows.append(r); curves[k] = curve
    return rows, curves


def time_score(name, seed, variant, preds, data, split):
    """Time error of the true positives and space-time AP of points with a t_h column."""
    groups = tuple(g for g in TIME_GROUPS if g in CONTEXT_GROUPS or g in data.fires or g == "isolated")
    rows = []
    for k in (5, 10):
        res, _, _ = evaluate(preds, data, split, k, groups=groups)
        r = {"method": name, "seed": seed, "time": variant, "k [km]": k, "TPs": res["recall (all preds)"] * res["targets"],
             **{c: res.get(c, np.nan) for c in TIME_KEYS},
             **{c.replace("median |dt| | ", "|dt| "): v for c, v in res.items() if c.startswith("median |dt| | ")}}
        for tau in (1, 2, 3):
            r[f"AP tau={tau} h"] = evaluate(preds, data, split, k, groups=(), tau_h=tau)[0]["AP"]
        rows.append(r)
    return rows


def time_variants(preds, data, lag, combine=False):
    """{variant: points with t_h}: the method's own time (if any), the first-detection rule and, for
    space-time models (combine), the combined time."""
    out = {"model": preds} if "t_h" in preds else {}
    if combine:
        ev = time_evidence(preds[["period", "lat", "lon", "score", "t_h"]], data, lag)
        out["combined"] = combined_times(ev, COMBINED_QUIET_H, COMBINED_AGREE_H).drop(columns="rule_used")
    rule = first_detection_times(preds.drop(columns=["t_h"], errors="ignore")[["period", "lat", "lon", "score"]], data)
    out["rule"] = rule.assign(t_h=np.clip(rule["t_h"] - lag, 0, 12))
    return out


def summarise(per_seed, keys):
    num = [c for c in per_seed.columns if c not in keys + ["seed"]]
    g = per_seed.groupby(keys, sort=False)[num]
    summary = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    summary.insert(0, "seeds", per_seed.groupby(keys, sort=False).size())
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="*", default=["cnn3d_final", "conv_final"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--folders", nargs="*", default=["final"], help="checkpoint folders under models/")
    ap.add_argument("--out", required=True, help="output folder under outputs/ and figures/ (never 'final')")
    ap.add_argument("--confirm-test", action="store_true", help="really evaluate on the test split (do this once)")
    args = ap.parse_args()
    assert args.out != "final", "outputs/final holds the test results of notes.md 6.17; choose another --out"
    split = "test" if args.confirm_test else "val"
    if split == "val":
        print("NOTE: --confirm-test not given -> evaluating on the VALIDATION split (code check, no test numbers).")
    device = get_device()
    data = FireData()
    assert data.georef_fix, "the final evaluation expects the geo-referencing-corrected build (6.12)"
    periods = data.periods.loc[(data.periods["split"] == split) & (data.periods["n_det"] > 0), "period"]
    years = sorted(periods.dt.year.unique())
    periods_by_year = {y: periods[periods.dt.year == y] for y in years}

    lag = training_lag_h(data, "train")
    print(f"rule time: first detection within the rule radius, training lag {lag:+.2f} h removed", flush=True)
    rows, trows, curves = [], [], {}

    def add(name, seed, preds, combine=False):
        r, c = score(name, seed, preds, data, split, periods_by_year)
        rows.extend(r); curves[(name, seed)] = c
        for variant, p in time_variants(preds, data, lag, combine).items():
            trows.extend(time_score(name, seed, variant, p, data, split))

    add("baseline: nucleus birth", -1, baseline_predictions(data)["birth"])
    for folder in args.folders:
        for exp in args.experiments:
            for seed in args.seeds:
                name = f"{folder}/{exp}_s{seed}"
                try:
                    model, cfg, ck = load_model(name, device)
                except FileNotFoundError:
                    print(f"missing checkpoint models/{name}.pt, skipped"); continue
                tb = cfg.get("time_bins", 0)
                print(f"{name}: epoch {ck.get('epoch', '?')}, trained on {ck['config'].get('train_splits', 'train')}"
                      + (f", space-time ({tb} bins, D1)" if tb else ""), flush=True)
                add(f"{folder}/{exp}", seed, predict(model, data, periods, cfg, device), combine=bool(tb))

    per_seed = pd.DataFrame(rows)
    summary = summarise(per_seed, ["method", "k [km]"])
    tper_seed = pd.DataFrame(trows)
    tsummary = summarise(tper_seed, ["method", "time", "k [km]"])

    out = E.ROOT / "outputs" / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / f"{split}_results_per_seed.csv", index=False)
    summary.to_csv(out / f"{split}_results_summary.csv")
    tper_seed.to_csv(out / f"{split}_time_per_seed.csv", index=False)
    tsummary.to_csv(out / f"{split}_time_summary.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    main_cols = ["seeds"] + KEYS
    text = (f"Final evaluation on the {split.upper()} split ({years[0]}–{years[-1]}), mean ± std over seeds\n\n"
            + summary[main_cols].to_string() + "\n\nRecall (all predictions) by subgroup and AP by year:\n"
            + summary[[c for c in summary.columns if "|" in c]].to_string() + "\n"
            + "\nIgnition time of the true positives (model = own time, rule = first detection, lag removed, combined = 9.18):\n"
            + tsummary[["seeds", "TPs"] + TIME_KEYS + [f"AP tau={t} h" for t in (1, 2, 3)]].to_string()
            + "\n\nmedian |dt| of the true positives by subgroup [h]:\n"
            + tsummary[[c for c in tsummary.columns if c.startswith("|dt| ") and "<=" not in c]].to_string() + "\n")
    (out / f"{split}_results.md").write_text("```\n" + text + "```\n")
    print(text)

    fig_dir = E.ROOT / "figures" / args.out
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {"baseline: nucleus birth": E.ORANGE}
    palette = itertools.cycle([E.BLUE, E.AQUA, E.INK2, "#8e5bd0", "#c23b5a", E.MUTED])
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
    print(f"written: outputs/{args.out}/{split}_results*.csv/.md, figures/{args.out}/{split}_pr.png (one curve per seed)")


if __name__ == "__main__":
    main()
