"""Choose the combined ignition time of cnn3d_st on validation 2015-2016 (notes.md 9.18), before the test.

Rule (baselines.combined_times): use the first-detection time (lag removed) where the place was quiet
for at least quiet_min_h before that detection (a fresh fire) and the rule agrees with the model within
agree_h hours; the model's own time (D1) otherwise. quiet_min_h = 0, agree_h = inf: rule wherever there
is a detection (= the 'rule' variant of evaluate_test.py with the model as fallback).

Points: the D1 predictions of ablations_all/cnn3d_st_all_s{0,1,2} cached by compare_runs.py. Only the
times change, so the place AP is the same for every candidate. Objective = time-aware AP, mean over
tau = 1, 2 h and k = 5, 10 km, mean over seeds. Fixed before the run: each candidate is chosen on one
validation year and scored on the other (both directions); the final choice is the best candidate on
the mean of the two years, adopted only if the cross-year choices beat the model time in both
directions; otherwise the model time stays.

Run from the repository root (after compare_runs.py):
  .venv/bin/python -u src/time_combination.py | tee outputs/time_combination.txt
"""
import itertools

import numpy as np
import pandas as pd

import eda_figures as E
from baselines import combined_times, time_evidence, training_lag_h
from dataset import FireData
from evaluate import evaluate
from spacetime_checks import GROUPS

ROOT = E.ROOT
RUNS = [f"ablations_all/cnn3d_st_all_s{s}" for s in (0, 1, 2)]
QUIET = (0, 1, 2, 3, 6, 12, 24)          # 24 = no detection in the 24 h before (time_evidence back_h)
AGREE = (np.inf, 3, 2, 1.5, 1, 0.5)
OUT = ROOT / "outputs" / "time_combination"


def evidence(name, data, lag):
    f = OUT / f"{name.replace('/', '__')}_evidence.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pts = pd.read_parquet(ROOT / "outputs" / "compare_runs" / f"{name.replace('/', '__')}.parquet")
    ev = time_evidence(pts, data, lag)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(f)
    return ev


def metrics(pts, data, periods):
    out = {}
    for k in (5, 10):
        res = evaluate(pts, data, "val", k, groups=(), periods=periods)[0]
        out[f"median |dt| @{k}"] = res["median |dt| TP [h]"]
        out[f"|dt|<=1h @{k}"] = res["|dt| <= 1 h"]
        for tau in (1, 2):
            out[f"AP tau={tau} @{k}"] = evaluate(pts, data, "val", k, groups=(), periods=periods, tau_h=tau)[0]["AP"]
    out["objective"] = np.mean([out[f"AP tau={t} @{k}"] for t in (1, 2) for k in (5, 10)])
    return out


def label(q, a):
    return "model" if q is None else f"quiet>={q}h, agree<={'inf' if np.isinf(a) else a}h"


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    data = FireData()
    lag = training_lag_h(data, "train")
    per = data.periods.loc[data.periods["split"] == "val", "period"]
    years = {y: per[per.dt.year == y] for y in sorted(per.dt.year.unique())}
    evs = {n: evidence(n, data, lag) for n in RUNS}
    for n, ev in evs.items():
        print(f"{n}: {len(ev):,} points, rule time for {ev['t_rule'].notna().mean():.1%}, "
              f"quiet >= 6 h for {(ev['quiet_h'] >= 6).mean():.1%}", flush=True)

    cands = [(None, None)] + list(itertools.product(QUIET, AGREE))
    rows = []
    for q, a in cands:
        for n, ev in evs.items():
            pts = ev.drop(columns=["t_rule", "quiet_h"]) if q is None else combined_times(ev, q, a)
            share = 0.0 if q is None else pts["rule_used"].mean()
            for y, p in years.items():
                rows.append({"candidate": label(q, a), "q": q, "a": a, "run": n, "year": y, "rule share": share,
                             **metrics(pts.drop(columns="rule_used", errors="ignore"), data, p)})
        print(f"  {label(q, a)} done", flush=True)
    df = pd.DataFrame(rows)
    by = df.groupby(["candidate", "year"], sort=False).mean(numeric_only=True).drop(columns=["q", "a"])
    obj = by["objective"].unstack("year")
    y0, y1 = list(years)
    base = obj.loc["model"]

    print("\nObjective (time-aware AP, mean of tau 1/2 h and 5/10 km, mean of 3 seeds) by year, best 15 on the mean:")
    t = obj.assign(mean=obj.mean(axis=1), gain_vs_model=obj.mean(axis=1) - base.mean(),
                   rule_share=by["rule share"].unstack("year").mean(axis=1)).sort_values("mean", ascending=False)
    print(t.head(15).round(4).to_string())
    print("\nmodel time:", base.round(4).to_dict())

    ok = True
    for a_, b_ in ((y0, y1), (y1, y0)):
        c = obj[a_].drop("model").idxmax()
        g = obj.loc[c, b_] - base[b_]
        ok &= g > 0
        print(f"chosen on {a_}: {c}; on {b_}: {obj.loc[c, b_]:.4f} vs model {base[b_]:.4f} ({g:+.4f})")
    best = t.drop("model").index[0]
    choice = best if ok else "model"
    print(f"\nFINAL CHOICE: {choice}" + ("" if ok else " (cross-year check failed, model time kept)"))

    # pooled report of model, rule-everywhere and the choice, incl. tau = 3 h and subgroups
    print("\nPooled validation 2015-2016, mean of 3 seeds:")
    report = {"model": (None, None), "rule wherever detected": (0, np.inf)}
    if choice != "model":
        report[f"combined: {choice}"] = tuple(df.loc[df["candidate"] == choice, ["q", "a"]].iloc[0])
    rep = []
    for lab_, (q, a) in report.items():
        for n, ev in evs.items():
            pts = ev.drop(columns=["t_rule", "quiet_h"]) if q is None else combined_times(ev, q, a).drop(columns="rule_used")
            for k in (5, 10):
                res = evaluate(pts, data, "val", k, groups=GROUPS)[0]
                r = {"time": lab_, "run": n, "k": k, "median |dt|": res["median |dt| TP [h]"], "median dt": res["median dt TP [h]"],
                     "<=1 h": res["|dt| <= 1 h"], "<=2 h": res["|dt| <= 2 h"],
                     **{c.replace("median |dt| | ", ""): v for c, v in res.items()
                        if c.startswith("median |dt| | ") and c.endswith("=True")}}
                for tau in (1, 2, 3):
                    r[f"AP tau={tau}"] = evaluate(pts, data, "val", k, groups=(), tau_h=tau)[0]["AP"]
                rep.append(r)
    print(pd.DataFrame(rep).groupby(["time", "k"], sort=False).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
