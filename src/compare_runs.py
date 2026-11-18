"""Compare configurations (several seeds each) on validation 2015-2016 with the current evaluation
(notes.md 9.15, 9.16): does training on more years help, for the 2D and the space-time model?

Every checkpoint is re-scored here, so runs trained with older code are scored exactly like new ones
(predictions cached in outputs/compare_runs/, refreshed when the checkpoint is newer). Per run: AP and
recall @ 5 / 10 km, recall in pair / already burning @ 10 km, best epoch, mean AP of epochs 16-20 from
the history. Per configuration: mean and range over seeds. Per difference A - B: the difference of the
seed means with a 95 % period bootstrap (validation periods resampled, the same resample for every run),
so the interval covers the choice of periods, not the seeds; the seed range is printed next to it.
Space-time runs are scored with their default decoding (D1, 3D peaks), place only.

Run from the repository root (groups or differences without finished runs are skipped):
  .venv/bin/python src/compare_runs.py > outputs/compare_runs.txt
  .venv/bin/python src/compare_runs.py --group a=ablations_x/run_s* --group b=... --diff a-b
Intervals on the test split, after the single test of evaluate_test.py (notes.md 9.20; no choices are
made from it): --split test --confirm-test.
"""
import argparse
import glob

import numpy as np
import pandas as pd

import eda_figures as E
from dataset import FireData
from evaluate import evaluate, match
from spacetime_checks import GROUPS
from train import get_device, load_model, predict

ROOT = E.ROOT
CACHE = ROOT / "outputs" / "compare_runs"
GROUPS_DEFAULT = ["final_0414=ablations_georef/cnn3d_final_s*", "final_all=ablations_all/cnn3d_final_all_s*",
                  "st_0414=ablations_st/cnn3d_st_s*", "st_all=ablations_all/cnn3d_st_all_s*"]
DIFFS_DEFAULT = ["final_all-final_0414", "st_all-st_0414", "st_0414-final_0414", "st_all-final_all", "st_all-final_0414"]
B = 2000


def runs_of(pattern):
    """Finished runs (done.json present) whose checkpoint matches models/<pattern>.pt."""
    out = []
    for p in sorted(glob.glob(str(ROOT / "models" / f"{pattern}.pt"))):
        name = p[len(str(ROOT / "models")) + 1:-3]
        if not name.endswith("_last") and (ROOT / "outputs" / f"{name}_done.json").exists():
            out.append(name)
    return out


def points(name, data, device, periods, split="val"):
    ckpt = ROOT / "models" / f"{name}.pt"
    f = CACHE / f"{name.replace('/', '__')}{'' if split == 'val' else '_' + split}.parquet"
    if f.exists() and f.stat().st_mtime > ckpt.stat().st_mtime:
        return pd.read_parquet(f)
    model, cfg, _ = load_model(name, device)
    pts = predict(model, data, periods, cfg, device)
    CACHE.mkdir(parents=True, exist_ok=True)
    pts.to_parquet(f)
    return pts


def history_stats(name):
    h = pd.read_csv(ROOT / "outputs" / f"{name}_history.csv")
    if "AP@5" not in h or h["AP@5"].isna().all():          # final runs (--no-val): last epoch, no history
        return {"best epoch": int(h["epoch"].max()), "late AP@5": np.nan, "late AP@10": np.nan}
    h = h.dropna(subset=["AP@5"])
    m = (h["AP@5"] + h["AP@10"]) / 2
    late = h[h["epoch"] >= 16]
    return {"best epoch": int(h.loc[m.idxmax(), "epoch"]), "late AP@5": late["AP@5"].mean(), "late AP@10": late["AP@10"].mean()}


class Boot:
    """AP of one run on resampled periods: period weights w (resample counts) -> AP, as in
    evaluate.average_precision (same ordering of the scored points)."""

    def __init__(self, scored, targets, periods):
        pos = {p: i for i, p in enumerate(periods)}
        s = scored[scored["status"] != "ignored"].sort_values("score", ascending=False)
        self.p = s["period"].map(pos).values
        self.tp = (s["status"] == "tp").values
        self.n = np.bincount(targets["period"].map(pos).values, minlength=len(periods))

    def ap(self, w):
        wr = w[self.p]
        tp = np.cumsum(wr * self.tp)
        fp = np.cumsum(wr * ~self.tp)
        n = max((w * self.n).sum(), 1)
        prec = tp / np.maximum(tp + fp, 1)
        env = np.maximum.accumulate(prec[::-1])[::-1]
        return np.sum(np.diff(np.r_[0, tp / n]) * env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", help="label=pattern under models/ (glob), repeatable")
    ap.add_argument("--diff", action="append", help="A-B of two group labels, repeatable")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true", help="needed with --split test")
    args = ap.parse_args()
    assert args.split == "val" or args.confirm_test, "--split test needs --confirm-test (after the single test, 9.20)"
    split = args.split
    groups = dict(g.split("=", 1) for g in (args.group or GROUPS_DEFAULT))
    diffs = args.diff or DIFFS_DEFAULT
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

    data = FireData()
    device = get_device()
    per_all = data.periods.loc[data.periods["split"] == split, "period"]
    per_pred = data.periods.loc[(data.periods["split"] == split) & (data.periods["n_det"] > 0), "period"]
    periods = sorted(set(pd.to_datetime(per_all, utc=True)))
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(periods), len(periods)), minlength=len(periods)) for _ in range(B)])

    rows, boots = [], {}
    for label, pattern in groups.items():
        names = runs_of(pattern)
        print(f"{label}: {len(names)} finished runs ({pattern})", flush=True)
        for name in names:
            pts = points(name, data, device, per_pred, split)
            row = {"config": label, "run": name.split("/")[-1], **history_stats(name)}
            for k in (5, 10):
                res, _, _ = evaluate(pts, data, split, k, groups=GROUPS)
                row[f"AP@{k}"], row[f"recall@{k}"] = res["AP"], res["recall (all preds)"]
                if k == 10:
                    row["in pair@10"] = res["recall | in_pair=True"]
                    row["already burning@10"] = res["recall | already_burning=True"]
                scored, targets = match(pts, data, set(periods), k)
                b = Boot(scored, targets, periods)
                assert abs(b.ap(np.ones(len(periods), int)) - res["AP"]) < 1e-9, (name, k)
                boots.setdefault((label, k), []).append(np.array([b.ap(w) for w in W]))
            rows.append(row)
    if not rows:
        return
    tab = pd.DataFrame(rows).set_index(["config", "run"])
    print(f"\nPer run ({split}, current evaluation; late = mean of the validated epochs 16-20 in the history):")
    print(tab.round(3).to_string())
    num = tab.drop(columns="best epoch")
    agg = num.groupby(level=0, sort=False).agg(["mean", "min", "max"])
    print("\nPer configuration (mean, min, max over seeds):")
    for label in agg.index:
        n = len(tab.loc[label])
        cells = [f"{c} {agg.loc[label, (c, 'mean')]:.3f} [{agg.loc[label, (c, 'min')]:.3f}-{agg.loc[label, (c, 'max')]:.3f}]"
                 for c in num.columns]
        print(f"  {label} ({n} seeds): " + "; ".join(cells))

    print(f"\nDifferences of the seed means, 95 % period bootstrap ({B} resamples of the {len(periods)} {split} periods):")
    for d in diffs:
        a, b_ = d.split("-")
        if (a, 5) not in boots or (b_, 5) not in boots:
            print(f"  {d}: skipped (no finished runs for {a if (a, 5) not in boots else b_})")
            continue
        parts = []
        for k in (5, 10):
            point = tab.loc[a, f"AP@{k}"].mean() - tab.loc[b_, f"AP@{k}"].mean()
            dist = np.mean(boots[(a, k)], axis=0) - np.mean(boots[(b_, k)], axis=0)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            parts.append(f"AP@{k} {point:+.3f} [{lo:+.3f}, {hi:+.3f}], P(<=0) {np.mean(dist <= 0):.3f}")
        na, nb = len(tab.loc[a]), len(tab.loc[b_])
        print(f"  {d} ({na} vs {nb} seeds): " + "; ".join(parts))


if __name__ == "__main__":
    main()
