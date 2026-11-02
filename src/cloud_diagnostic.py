"""Is cloud linked to the errors of the model? Go/no-go test before adding a cloud input channel.

Uses the validation years 2015-2016 only and the checkpoints trained on 2004-2014
(ablations_georef/cnn3d_final_s0-2). Cloud comes from src/download_msg_quality.py. Not observable =
flags CLOUD (3), SUNG (4), SUNGRATIO (5), BADINPUT (9) or no file. WATER (10) and WATEREDGE (11) are
fixed masks that hide the cloud state (except WATEREDGE in Jan-Apr 2016, algorithm 2.0), so these
cells are left out of the shares.
Every point gets the not-observable share of a 5 x 5 window:
  unobs_history, unobs_period   the 6 h before the period, the 12 h of the period
  unobs_before                  the 3 h before the first detection near the point
  reappear, reappear_cloud      detection near the point in the 48 h before, with a gap of at
                                least 1 h; and at least half of that gap not observable

Parts of the report:
  1 alignment   the detections must carry flag 1/2 in the same slot and cell
  2 predictions true vs. false positives, also within score bins (period bootstrap)
  3 targets     recall by cloud cover at the target
  4 re-scoring  AP of logit(score) + reappear + has_det + cloud features, fitted on the other year,
                against the same without the cloud features: a simple estimate of what cloud adds
                to the score (matches kept fixed). Next to it the AP if every reappear_cloud false
                positive vanished (upper bound)
  5 unseen      cloud after the ICNF start, ICNF fires seen vs. not seen by SEVIRI

Decision, fixed before seeing the data, at 10 km:
  GO       re-scoring gain >= 0.02 AP (the validation noise) with a confidence interval above 0
  WEAK     no such gain, but false positives have >= 0.05 more unobs_before or reappear_cloud
           than true positives within score bins, or recall is >= 0.10 lower in the cloudiest
           third of targets than in the clearest third (interval excluding 0 in each case)
  NO-GO    neither

Run from the repository root (needs no login; predictions are kept in outputs/cloud/):
  .venv/bin/python src/cloud_diagnostic.py | tee outputs/cloud/diagnostic.txt
"""
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from dataset import FireData
from download_msg_quality import DAYS_DIR, MARGIN, MISSING
from evaluate import average_precision, match

OUT = E.ROOT / "outputs" / "cloud"
SPLIT = "val"
FIRST_DAY = pd.Timestamp("2015-01-01", tz="UTC")
BUFFER_DAYS = 3             # slots before a period start that the features look at
SLOTS_DAY = 96
UNOBS_FLAGS = (3, 4, 5, 9, MISSING)
CANDIDATE_FLAGS = (6, 7)
R_WINDOW = 2                # 5 x 5 cells
R_DET = 3                   # a detection is "near" a point within this many cells
H, P = 24, 48               # slots of the history and of the target period
BEFORE = 12                 # 3 h
PRIOR = 192                 # 48 h
MIN_GAP = 4                 # 1 h
BOOT = 500
GO_GAIN = 0.02              # validation noise (notes.md)
WEAK_DIFF, WEAK_RECALL = 0.05, 0.10
HIGH_CLOUD = 0.5
SCORE_BINS = 5
NOT_CLOUD = ["reappear", "has_det"]
CLOUD_FEATURES = ["unobs_history", "unobs_period", "unobs_before", "reappear_cloud"]

UNOBS_LUT = np.isin(np.arange(256), UNOBS_FLAGS)
CAND_LUT = np.isin(np.arange(256), CANDIDATE_FLAGS)
WATER_LUT = np.isin(np.arange(256), (10, 11))


def unobs_share(window):
    """Not-observable share of the cells that can report cloud; NaN if there are none."""
    usable = (~WATER_LUT[window]).sum()
    return UNOBS_LUT[window].sum() / usable if usable else np.nan


def slot_of(period):
    return pd.Timestamp(period).value // (15 * 60 * 10**9)


class Cloud:
    """Flags of all days around the validation years, indexed by absolute 15-min slot and cell."""

    def __init__(self, quality_dir, start, end):
        self.s0 = slot_of(start)
        days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
        self.flags = None
        self.absent = 0
        for k, d in enumerate(days):
            f = quality_dir / f"{d.year}" / f"{d:%Y%m%d}.npz"
            if not f.exists():
                self.absent += 1
                continue
            a = np.load(f)["flags"]
            if self.flags is None:
                self.flags = np.full((len(days) * SLOTS_DAY, *a.shape[1:]), MISSING, np.uint8)
            self.flags[k * SLOTS_DAY:(k + 1) * SLOTS_DAY] = a
        if self.flags is None:
            raise SystemExit(f"No Quality files in {quality_dir}: run src/download_msg_quality.py first.")

    def window(self, s, e, i, j, r=R_WINDOW):
        s, e = max(s - self.s0, 0), max(e - self.s0, 0)
        i, j = i + MARGIN, j + MARGIN
        return self.flags[s:e, max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]

    def unobs(self, s, e, i, j):
        w = self.window(s, e, i, j)
        return unobs_share(w)


def check_alignment(cloud, data):
    """Share of detections with flag 1/2 and share of slots present; stops the run if misaligned."""
    t = data.det["time_utc"]
    d = data.det[(t >= FIRST_DAY) & (t < FIRST_DAY + pd.DateOffset(years=2))]
    f = cloud.flags[d["slot"].values - cloud.s0, d["i"].values.astype(int) + MARGIN, d["j"].values.astype(int) + MARGIN]
    have = f != MISSING
    share = np.isin(f[have], (1, 2)).mean()
    values, counts = np.unique(cloud.flags[slot_of(FIRST_DAY) - cloud.s0::8], return_counts=True)
    print("\n1. Alignment")
    print(f"  days absent {cloud.absent}; detections in present slots {have.sum():,} of {len(d):,}; "
          f"with flag 1/2 {share:.1%}")
    print("  flag shares: " + ", ".join(f"{v}: {c / counts.sum():.1%}" for v, c in zip(values, counts)))
    m = MARGIN
    pt = cloud.flags[slot_of(FIRST_DAY) - cloud.s0::8, m:-m, m:-m][:, data.static["pt_loss"].astype(bool)]
    always = WATER_LUT[pt].all(axis=0)
    print(f"  Portugal cells always water or water edge: {always.mean():.1%}; ever both water/edge and "
          f"another flag: {(WATER_LUT[pt].any(axis=0) & ~always).mean():.1%}")
    return share


def det_lookup(data):
    return data.det["slot"].values, data.det["i"].values.astype(int), data.det["j"].values.astype(int)


def point_features(cloud, det, period, i, j):
    """Cloud features of one point at cell (i, j) in the sensor frame."""
    slot, di, dj = det
    s0 = slot_of(period) - H
    a, b = np.searchsorted(slot, [s0 - PRIOR, s0 + 96])
    near = (np.abs(di[a:b] - i) <= R_DET) & (np.abs(dj[a:b] - j) <= R_DET)
    times = slot[a:b][near]
    inside = times[times >= s0]
    has_det = len(inside) > 0
    first = int(inside[0]) if has_det else s0 + H
    prior = times[(times < first) & (times >= first - PRIOR)]
    gap = first - int(prior[-1]) if len(prior) else 0
    reappear = bool(has_det and gap >= MIN_GAP)
    unobs_gap = cloud.unobs(first - gap + 1, first, i, j) if reappear else 0.0
    before = cloud.window(first - BEFORE, first, i, j)
    return {
        "unobs_history": cloud.unobs(s0, s0 + H, i, j),
        "unobs_period": cloud.unobs(s0 + H, s0 + H + P, i, j),
        "unobs_before": unobs_share(before),
        "candidate_before": float(CAND_LUT[before].any()) if before.size else np.nan,
        "reappear": float(reappear),
        "reappear_cloud": float(reappear and unobs_gap >= 0.5),
        "has_det": float(has_det),
    }


def sensor_cells(lat, lon, periods):
    i, j = G.latlon_to_ij(lat, lon)
    i, j = G.true_to_sensor_ij(i, j, periods)
    return np.clip(np.round(i).astype(int), 0, G.H - 1), np.clip(np.round(j).astype(int), 0, G.W - 1)


def seed_predictions(data, seeds, folder):
    """Raw predictions of every seed, computed once."""
    from train import get_device, load_model, predict
    periods = data.periods.loc[(data.periods["split"] == SPLIT) & (data.periods["n_det"] > 0), "period"]
    out = {}
    for seed in seeds:
        f = OUT / f"val_preds_s{seed}.parquet"
        if not f.exists():
            device = get_device()
            model, cfg, _ = load_model(f"{folder}/cnn3d_final_s{seed}", device)
            predict(model, data, periods, cfg, device).to_parquet(f, index=False)
            print(f"  predictions of seed {seed} written", flush=True)
        out[seed] = pd.read_parquet(f)
    return out


def scored_table(cloud, data, preds, k_km):
    """All predictions of all seeds with status, score bin and cloud features."""
    periods = set(data.periods.loc[data.periods["split"] == SPLIT, "period"])
    det = det_lookup(data)
    rows = []
    n_targets = None
    for seed, p in preds.items():
        s, targets = match(p, data, periods, k_km)
        n_targets = len(targets)
        ap, rec, prec, thr = average_precision(s, n_targets)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        s = s[s["status"] != "ignored"].copy()
        s["seed"] = seed
        s["operating"] = s["score"] >= thr[int(np.argmax(f1))]
        # equal numbers of true positives per bin: quantiles of all points would put nearly all of
        # them in the top bin, and the low bins would compare a handful with thousands
        edges = np.quantile(s.loc[s["status"] == "tp", "score"], np.linspace(0, 1, SCORE_BINS + 1)[1:-1])
        s["score_bin"] = np.searchsorted(edges, s["score"].values)
        i, j = sensor_cells(s["lat"].values, s["lon"].values, s["period"].values)
        feats = [point_features(cloud, det, q, a, b) for q, a, b in zip(s["period"], i, j)]
        rows.append(pd.concat([s.reset_index(drop=True), pd.DataFrame(feats)], axis=1))
    return pd.concat(rows, ignore_index=True), n_targets


def boot_periods(table, stat, rng, n=BOOT):
    """Mean of `stat` and a 95 % interval, resampling whole periods."""
    groups = list(table.groupby("period").indices.values())
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(table.iloc[np.concatenate([groups[k] for k in pick])]))
    return stat(table), *np.nanpercentile(vals, [2.5, 97.5])


def adjusted_diff(table, feature):
    """False minus true positives within seed and score bin. Mantel-Haenszel weights: a bin counts
    by the smaller of its two groups, so a few true positives among many points weigh little."""
    g = table.groupby(["seed", "score_bin", "status"])[feature].agg(["mean", "size"]).unstack("status")
    d = (g["mean"]["fp"] - g["mean"]["tp"]).dropna()
    fp, tp = g["size"]["fp"].reindex(d.index), g["size"]["tp"].reindex(d.index)
    return float(np.average(d, weights=fp * tp / (fp + tp)))


def report_predictions(table, rng):
    print("\n2. True vs. false positives")
    rows = []
    for name, sub in (("all predictions", table), ("at best F1", table[table["operating"]])):
        for f in CLOUD_FEATURES + ["reappear", "candidate_before"]:
            m = sub.groupby("status")[f].mean()
            est, lo, hi = boot_periods(sub, lambda t: adjusted_diff(t, f), rng, n=BOOT // 2)
            rows.append({"predictions": name, "feature": f, "TP": m.get("tp"), "FP": m.get("fp"),
                         "FP - TP, same score": est, "2.5 %": lo, "97.5 %": hi})
    tab = pd.DataFrame(rows)
    print(f"  counts per seed: {table.groupby(['seed', 'status']).size().unstack().round().to_dict('index')}")
    print(tab.round(3).to_string(index=False))
    return tab


def report_targets(cloud, data, preds, k_km, rng):
    print("\n3. Recall by cloud cover at the target (history and period, 5 x 5 window)")
    periods = set(data.periods.loc[data.periods["split"] == SPLIT, "period"])
    lab = data.labels[(data.labels["period"].isin(periods)) & (data.labels["role"] == "target")]
    i, j = np.round(lab["i_f"].values).astype(int), np.round(lab["j_f"].values).astype(int)
    cover = [cloud.unobs(slot_of(p) - H, slot_of(p) + P, a, b) for p, a, b in zip(lab["period"], i, j)]
    t = lab[["period", "fire_id"]].assign(cover=cover)
    for seed, p in preds.items():
        s, _ = match(p, data, periods, k_km)
        t[f"hit{seed}"] = t["fire_id"].isin(s.loc[s["status"] == "tp", "fire_id"]).astype(float)
    hits = [c for c in t.columns if c.startswith("hit")]
    t["hit"] = t[hits].mean(axis=1)
    t["bin"] = pd.cut(t["cover"], [-0.01, 0.1, HIGH_CLOUD, 1.0], labels=["<= 10 %", "10-50 %", "> 50 %"])
    print(t.groupby("bin", observed=False)["hit"].agg(targets="size", recall="mean").round(3).to_string())

    low, high = t["cover"].quantile([1 / 3, 2 / 3])
    n_clear, n_cloudy = (t["cover"] <= low).sum(), (t["cover"] > high).sum()

    def gap(x):
        return x.loc[x["cover"] <= low, "hit"].mean() - x.loc[x["cover"] > high, "hit"].mean()
    est, lo, hi = boot_periods(t, gap, rng)
    print(f"  recall in the clearest third ({n_clear} targets, cover <= {low:.2f}) minus the cloudiest "
          f"third ({n_cloudy} targets, cover > {high:.2f}): {est:.3f} [{lo:.3f}, {hi:.3f}]")
    return est, lo, hi


def fit_logistic(x, y, ridge=1.0, steps=25):
    x = np.c_[np.ones(len(x)), x]
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-x @ w))
        grad = x.T @ (p - y) + ridge * w
        hess = (x.T * (p * (1 - p) + 1e-6)) @ x + ridge * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w


def rescore(table, columns):
    """Out-of-year score: fitted on the other validation year, per seed."""
    out = pd.Series(np.nan, index=table.index)
    logit = np.log(table["score"].clip(1e-5, 1 - 1e-5) / (1 - table["score"].clip(1e-5, 1 - 1e-5))).values
    x = np.c_[logit, table[columns].fillna(0).values] if columns else logit[:, None]
    y = (table["status"] == "tp").values.astype(float)
    year = table["period"].dt.year.values
    seed = table["seed"].values
    mu, sd = x.mean(0), x.std(0) + 1e-9
    x = (x - mu) / sd
    for sd_seed in np.unique(seed):
        for yr in np.unique(year):
            test, train = (seed == sd_seed) & (year == yr), (seed == sd_seed) & (year != yr)
            w = fit_logistic(x[train], y[train])
            out[test] = np.c_[np.ones(test.sum()), x[test]] @ w
    return out


def mean_ap(table, column, n_targets):
    return np.mean([average_precision(g.assign(score=g[column]), n_targets)[0] for _, g in table.groupby("seed")])


def report_rescoring(table, n_targets, rng, k_km):
    print(f"\n4. Re-scoring, {k_km} km (fitted on the other year, status fixed)")
    t = table.assign(base=rescore(table, NOT_CLOUD), with_cloud=rescore(table, NOT_CLOUD + CLOUD_FEATURES))
    ap_raw, ap0, ap1 = (mean_ap(t, c, n_targets) for c in ("score", "base", "with_cloud"))
    oracle = mean_ap(t[~((t["status"] == "fp") & (t["reappear_cloud"] > 0))], "score", n_targets)
    est, lo, hi = boot_periods(t, lambda x: mean_ap(x, "with_cloud", n_targets) - mean_ap(x, "base", n_targets), rng, n=BOOT // 2)
    print(f"  AP model {ap_raw:.3f}; refit without cloud {ap0:.3f}; with cloud {ap1:.3f}")
    print(f"  gain {est:+.3f} [{lo:+.3f}, {hi:+.3f}]; if all reappear_cloud false positives vanished: AP {oracle:.3f} ({oracle - ap_raw:+.3f})")
    for seed, g in t.groupby("seed"):
        print(f"    seed {seed}: gain {average_precision(g.assign(score=g['with_cloud']), n_targets)[0] - average_precision(g.assign(score=g['base']), n_targets)[0]:+.3f}")
    return est, lo, hi


def report_unseen(cloud, data):
    print("\n5. ICNF fires seen vs. not seen by SEVIRI: not observable in the 6 h after the start")
    f = data.fires[data.fires["split"] == SPLIT]
    i, j = sensor_cells(f["lat"].values, f["lon"].values, f["start_utc"].values)
    cover = [cloud.unobs(slot_of(t), slot_of(t) + 24, a, b) for t, a, b in zip(f["start_utc"], i, j)]
    t = f.assign(cover=cover, big=np.where(f["burned_area_ha"] >= 50, ">= 50 ha", "< 50 ha"))
    print(t.groupby(["big", "seen"])["cover"].agg(fires="size", mean="mean", high=lambda c: (c > HIGH_CLOUD).mean()).round(3).to_string())


def verdict(gain, prediction_tab, recall_gap):
    print("\nDecision (10 km)")
    all_preds = prediction_tab[prediction_tab["predictions"] == "all predictions"].set_index("feature")
    enriched = [f for f in ("unobs_before", "reappear_cloud")
                if all_preds.loc[f, "FP - TP, same score"] >= WEAK_DIFF and all_preds.loc[f, "2.5 %"] > 0]
    low_recall = recall_gap[0] >= WEAK_RECALL and recall_gap[1] > 0
    if gain[0] >= GO_GAIN and gain[1] > 0:
        result = "GO: cloud raises AP beyond the validation noise; train with a cloud channel."
    elif enriched or low_recall:
        result = (f"WEAK: linked to errors (false positives: {enriched or 'no'}; recall drop: {low_recall}), "
                  f"but re-scoring gain {gain[0]:+.3f} is below {GO_GAIN}.")
    else:
        result = "NO-GO: no clear link to the errors; do not train with a cloud channel."
    print("  " + result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-dir", type=Path, default=DAYS_DIR)
    ap.add_argument("--folder", default="ablations_georef")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--no-alignment-stop", action="store_true", help="for tests with made-up flags")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 200)

    data = FireData()
    assert data.georef_fix, "expects the geo-referencing-corrected build"
    cloud = Cloud(args.quality_dir, (FIRST_DAY - pd.Timedelta(days=BUFFER_DAYS)).date(), pd.Timestamp("2017-01-02").date())
    share = check_alignment(cloud, data)
    if share < 0.9 and not args.no_alignment_stop:
        raise SystemExit("Stopped: the flags do not line up with the detections.")

    preds = seed_predictions(data, args.seeds, args.folder)
    gains, tabs, gaps = {}, {}, {}
    for k_km in (10, 5):
        print(f"\n===== match within {k_km} km =====")
        table, n_targets = scored_table(cloud, data, preds, k_km)
        tabs[k_km] = report_predictions(table, rng)
        gaps[k_km] = report_targets(cloud, data, preds, k_km, rng)
        gains[k_km] = report_rescoring(table, n_targets, rng, k_km)
        table.drop(columns=["lat", "lon"]).to_parquet(OUT / f"val_scored_{k_km}km.parquet", index=False)
    report_unseen(cloud, data)
    verdict(gains[10], tabs[10], gaps[10])


if __name__ == "__main__":
    main()
