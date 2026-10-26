"""Evaluation protocol for ignition predictions (notes.md 5.2).

Predictions are points (period, lat, lon, score). A heatmap model is turned into points by
peak extraction; the baselines produce points directly.

Matching, per 12 h target period, predictions in descending score order:
  - nearest still unmatched *target* within k km   -> true positive (error = distance)
  - otherwise within k km of an *ignore* fire       -> ignored (neither TP nor FP)
  - otherwise                                       -> false positive (also duplicates)
Predictions outside the loss mask (mainland Portugal + 1 px) are dropped.

Metrics per split: average precision (area under the precision-recall curve over all score
thresholds), recall and precision with all predictions, the best F1 with its precision,
recall and false alarms per day, the median error of true positives, recall by subgroup, and
the distance from every target to the nearest prediction of its period (no matching).
"""
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G


def _haversine(lat, lon, lats, lons):
    return E.haversine_km(lat, lon, np.asarray(lats), np.asarray(lons))


def match(preds, data, periods, k_km):
    """Status of every prediction in `periods` and the matched target of every true positive."""
    preds = preds.assign(period=pd.to_datetime(preds["period"], utc=True))   # all times are UTC
    preds = preds[preds["period"].isin(periods)].copy()
    i, j = G.latlon_to_ij(preds["lat"].values, preds["lon"].values)
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    on = (i >= 0) & (i < G.H) & (j >= 0) & (j < G.W)
    keep = np.zeros(len(preds), bool)
    keep[on] = data.static["pt_loss"][i[on], j[on]]
    preds = preds[keep]

    lab = data.labels[data.labels["period"].isin(periods)].merge(data.fires[["fire_id", "lat", "lon"]], on="fire_id")
    lab_by = {p: g for p, g in lab.groupby("period")}
    status, fire, err = [], [], []
    preds = preds.sort_values(["period", "score"], ascending=[True, False])
    for p, g in preds.groupby("period", sort=False):
        L = lab_by.get(p)
        tg = L[L["role"] == "target"] if L is not None else None
        ig = L[L["role"] == "ignore"] if L is not None else None
        free = np.ones(0 if tg is None else len(tg), bool)
        for q in g.itertuples():
            if tg is not None and len(tg):
                d = _haversine(q.lat, q.lon, tg["lat"], tg["lon"])
                d = np.where(free, d, np.inf)
                if d.min() <= k_km:
                    m = int(d.argmin()); free[m] = False
                    status.append("tp"); fire.append(tg["fire_id"].iat[m]); err.append(d[m])
                    continue
            if ig is not None and len(ig) and _haversine(q.lat, q.lon, ig["lat"], ig["lon"]).min() <= k_km:
                status.append("ignored")
            else:
                status.append("fp")
            fire.append(-1); err.append(np.nan)
    preds["status"], preds["fire_id"], preds["err_km"] = status, fire, err
    return preds, lab[lab["role"] == "target"]


def average_precision(scored, n_targets):
    s = scored[scored["status"] != "ignored"].sort_values("score", ascending=False)
    tp = (s["status"] == "tp").cumsum().values
    fp = (s["status"] == "fp").cumsum().values
    recall = tp / max(n_targets, 1)
    precision = tp / np.maximum(tp + fp, 1)
    env = np.maximum.accumulate(precision[::-1])[::-1]          # interpolated precision
    ap = np.sum(np.diff(np.r_[0, recall]) * env)
    return ap, recall, precision, s["score"].values


def evaluate(preds, data, split, k_km, groups=("isolated", "size_class"), periods=None):
    """periods: optional subset of the split's periods (e.g. one year); default all periods."""
    all_p = data.periods.loc[data.periods["split"] == split, "period"]
    periods = all_p if periods is None else all_p[all_p.isin(set(periods))]
    scored, targets = match(preds, data, set(periods), k_km)
    n = len(targets)
    ap, rec, prec, thr = average_precision(scored, n)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    b = int(np.argmax(f1)) if len(f1) else 0
    n_days = len(periods) / 2
    fp_best = int(((scored["status"] == "fp") & (scored["score"] >= (thr[b] if len(thr) else np.inf))).sum())
    tp = scored[scored["status"] == "tp"]

    # distance from every target to the nearest prediction of its period (no matching)
    by = {p: g for p, g in scored.groupby("period")}
    nearest = np.array([_haversine(t.lat, t.lon, by[t.period]["lat"], by[t.period]["lon"]).min() if t.period in by else np.inf
                        for t in targets.itertuples()])
    res = {
        "split": split, "k [km]": k_km, "targets": n, "AP": ap,
        "recall (all preds)": rec[-1] if len(rec) else 0, "precision (all preds)": prec[-1] if len(prec) else 0,
        "best F1": f1[b] if len(f1) else 0, "P @ best F1": prec[b] if len(prec) else 0, "R @ best F1": rec[b] if len(rec) else 0,
        "false alarms / day @ best F1": fp_best / n_days,
        "median error TP [km]": tp["err_km"].median(),
        "nearest pred ≤ 5 km": np.mean(nearest <= 5), "nearest pred median [km]": np.median(nearest),
    }
    attrs = data.fires.set_index("fire_id").loc[targets["fire_id"]]
    hit = targets["fire_id"].isin(tp["fire_id"]).values
    for gname in groups:
        for val, h in pd.Series(hit, index=attrs[gname].astype(str).values).groupby(level=0):
            res[f"recall | {gname}={val}"] = h.mean()
    curve = pd.DataFrame({"recall": rec, "precision": prec, "score": thr})
    return res, curve, scored
