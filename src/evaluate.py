"""Evaluation protocol for ignition predictions (notes.md 5.2).

Predictions are points (period, lat, lon, score). A heatmap model is turned into points by
peak extraction; the baselines produce points directly.

Matching, per 12 h target period, predictions in descending score order:
  - nearest still unmatched *target* within k km   -> true positive (error = distance)
  - otherwise within k km of an *ignore* fire       -> ignored (neither TP nor FP)
  - otherwise                                       -> false positive (also duplicates)
Predictions outside the loss mask (mainland Portugal + 1 px) are dropped.
Second protocol (neutral_rekindles=True, notes.md 8.4): a prediction that matches no target and lies
within k km of an ICNF rekindle record of its period (any size) is also 'ignored'. The default
protocol is unchanged.
Space-time protocol (tau_h given, notes.md 9.4 item 12): predictions carry t_h (hours since the period
start) and a true positive also needs |t_h - ICNF start| <= tau_h; ignore and rekindle rules stay
place-only. Whenever predictions carry t_h, the time error of the true positives is reported (median
|dt|, share within 1 h / 2 h), overall and per subgroup.

Metrics per split: average precision (area under the precision-recall curve over all score
thresholds), recall and precision with all predictions, the best F1 with its precision,
recall and false alarms per day, the median error of true positives, recall by subgroup (isolation, size class; with the SGIF fields also rekindle, cause type,
fire-weather quartile and dominant land cover, see report_attrs), and the distance from every target
to the nearest prediction of its period (no matching). Subgroups from target_context(): 'in_pair'
(another target <= 2 cells away in the same period) and 'already_burning' (a detection <= 1.5 cells
away in [start - 6 h, start - 30 min]), as in notes.md 9.2.
"""
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G


# FWI quartiles of the ICNF fires >= 50 ha 2004-2023 (notes.md 8.4), fixed so that every split uses the same bins
FWI_BINS = [0, 23.7, 32.8, 40.5, np.inf]
FWI_LABELS = ["q1 (<23.7)", "q2 (23.7-32.8)", "q3 (32.8-40.5)", "q4 (>=40.5)"]
REPORT_GROUPS = ("rekindle", "cause_type", "fwi_quartile", "cover")
CONTEXT_GROUPS = ("in_pair", "already_burning")
PAIR_PX, BURN_PX = 2.0, 1.5          # close pair / already burning, in native cells (notes.md 6.19, 9.2)


def target_context(data):
    """Per target fire (index fire_id): t_h = ICNF start in hours since its period start, in_pair,
    already_burning (module docstring). Positions in the sensor frame (labels.parquet), detections on
    their cells. Cached on the FireData object."""
    if getattr(data, "_target_context", None) is not None:
        return data._target_context
    tg = data.labels.loc[data.labels["role"] == "target", ["period", "fire_id", "i_f", "j_f"]].copy()
    start = tg["fire_id"].map(data.fires.set_index("fire_id")["start_utc"])
    tg["t_h"] = (start - tg["period"]).dt.total_seconds() / 3600
    tg["in_pair"] = False
    for _, g in tg.groupby("period"):
        if len(g) > 1:
            v = g[["i_f", "j_f"]].values
            d = np.hypot(v[:, None, 0] - v[None, :, 0], v[:, None, 1] - v[None, :, 1]) + np.eye(len(v)) * 1e9
            tg.loc[g.index, "in_pair"] = (d <= PAIR_PX).any(1)
    t = data.det["time_utc"].values                      # sorted (FireData sorts by slot)
    a = np.searchsorted(t, (start - pd.Timedelta(hours=6)).dt.tz_convert(None).values.astype("datetime64[ns]"))
    b = np.searchsorted(t, (start - pd.Timedelta(minutes=30)).dt.tz_convert(None).values.astype("datetime64[ns]"), side="right")
    tg["already_burning"] = [bool((np.hypot(data._i[x:y] - fi, data._j[x:y] - fj) <= BURN_PX).any())
                             for x, y, fi, fj in zip(a, b, tg["i_f"], tg["j_f"])]
    data._target_context = tg.set_index("fire_id")[["t_h", "in_pair", "already_burning"]]
    return data._target_context


def _haversine(lat, lon, lats, lons):
    return E.haversine_km(lat, lon, np.asarray(lats), np.asarray(lons))


def report_attrs(fires):
    """fires.parquet plus the reporting subgroups that need a derivation: rekindle (True / False /
    unknown, the cause is often missing before 2008), FWI quartile, dominant land cover of the burned
    area (>= 50 % forest / shrub / agricultural, else mixed). Builds without the SGIF fields are returned
    unchanged."""
    if "rekindle" not in fires:
        return fires
    a = fires.copy()
    a["rekindle"] = a["rekindle"].astype(object).where(a["rekindle"].notna(), "unknown")
    a["fwi_quartile"] = pd.cut(a["fwi"], FWI_BINS, labels=FWI_LABELS, right=False)
    tot = a["burned_area_ha"]
    a["cover"] = np.select([a["area_forest_ha"] >= 0.5 * tot, a["area_shrub_ha"] >= 0.5 * tot,
                            a["area_agric_ha"] >= 0.5 * tot], ["forest", "shrub", "agric"], "mixed")
    a.loc[a["area_forest_ha"].isna(), "cover"] = np.nan
    return a


def match(preds, data, periods, k_km, neutral_rekindles=False, tau_h=None):
    """Status of every prediction in `periods` and the matched target of every true positive.
    tau_h: space-time protocol (module docstring); needs a t_h column in preds."""
    timed = "t_h" in preds
    assert tau_h is None or timed, "tau_h needs predictions with a t_h column"
    preds = preds.assign(period=pd.to_datetime(preds["period"], utc=True))   # all times are UTC
    preds = preds[preds["period"].isin(periods)].copy()
    i, j = G.latlon_to_ij(preds["lat"].values, preds["lon"].values)
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    on = (i >= 0) & (i < G.H) & (j >= 0) & (j < G.W)
    keep = np.zeros(len(preds), bool)
    keep[on] = data.static["pt_loss"][i[on], j[on]]
    preds = preds[keep]

    lab = data.labels[data.labels["period"].isin(periods)].merge(data.fires[["fire_id", "lat", "lon", "start_utc"]], on="fire_id")
    lab["t_h"] = (lab["start_utc"] - lab["period"]).dt.total_seconds() / 3600
    lab_by = {p: g for p, g in lab.groupby("period")}
    rk_by = {p: g for p, g in data.rekindles.groupby("period")} if neutral_rekindles else {}
    status, fire, err, dt = [], [], [], []
    preds = preds.sort_values(["period", "score"], ascending=[True, False])
    for p, g in preds.groupby("period", sort=False):
        L = lab_by.get(p)
        tg = L[L["role"] == "target"] if L is not None else None
        ig = L[L["role"] == "ignore"] if L is not None else None
        free = np.ones(0 if tg is None else len(tg), bool)
        for q in g.itertuples():
            if tg is not None and len(tg):
                d = _haversine(q.lat, q.lon, tg["lat"], tg["lon"])
                ok = free if tau_h is None else free & (np.abs(tg["t_h"].values - q.t_h) <= tau_h)
                d = np.where(ok, d, np.inf)
                if d.min() <= k_km:
                    m = int(d.argmin()); free[m] = False
                    status.append("tp"); fire.append(tg["fire_id"].iat[m]); err.append(d[m])
                    dt.append(q.t_h - tg["t_h"].iat[m] if timed else np.nan)
                    continue
            rk = rk_by.get(p)
            if ig is not None and len(ig) and _haversine(q.lat, q.lon, ig["lat"], ig["lon"]).min() <= k_km:
                status.append("ignored")
            elif rk is not None and _haversine(q.lat, q.lon, rk["lat"], rk["lon"]).min() <= k_km:
                status.append("ignored")
            else:
                status.append("fp")
            fire.append(-1); err.append(np.nan); dt.append(np.nan)
    preds["status"], preds["fire_id"], preds["err_km"] = status, fire, err
    if timed:
        preds["dt_h"] = dt                                   # predicted - ICNF start, true positives only
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


def evaluate(preds, data, split, k_km, groups=("isolated", "size_class"), periods=None, neutral_rekindles=False,
             tau_h=None):
    """periods: optional subset of the split's periods (e.g. one year); default all periods.
    neutral_rekindles: second protocol; tau_h: space-time protocol (module docstring)."""
    all_p = data.periods.loc[data.periods["split"] == split, "period"]
    periods = all_p if periods is None else all_p[all_p.isin(set(periods))]
    scored, targets = match(preds, data, set(periods), k_km, neutral_rekindles, tau_h)
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
    if "dt_h" in scored:
        a = tp["dt_h"].abs()
        res.update({"median |dt| TP [h]": a.median(), "median dt TP [h]": tp["dt_h"].median(),
                    "|dt| <= 1 h": (a <= 1).mean(), "|dt| <= 2 h": (a <= 2).mean()})
    attrs = report_attrs(data.fires).set_index("fire_id")
    if set(groups) & set(CONTEXT_GROUPS):
        attrs = attrs.join(target_context(data)[list(CONTEXT_GROUPS)])
    attrs = attrs.loc[targets["fire_id"]]
    hit = targets["fire_id"].isin(tp["fire_id"]).values
    for gname in groups:
        for val, h in pd.Series(hit, index=attrs[gname].astype(str).values).groupby(level=0):
            res[f"recall | {gname}={val}"] = h.mean()
        if "dt_h" in scored and len(tp):
            g = tp.assign(grp=tp["fire_id"].map(attrs[gname].astype(str)))
            for val, d in g.groupby("grp")["dt_h"]:
                res[f"median |dt| | {gname}={val}"] = d.abs().median()
    curve = pd.DataFrame({"recall": rec, "precision": prec, "score": thr})
    return res, curve, scored
