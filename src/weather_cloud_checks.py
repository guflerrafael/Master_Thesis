"""Would wind / fire weather or a cloud channel help cnn3d_st? Two go/no-go checks on validation
(notes.md 9.13). No new data: fire weather from the ICNF SGIF export, cloud from the LSA SAF Quality
product of 2015-2016 (src/download_msg_quality.py, notes.md 6.20).

Models: ablations_st/cnn3d_st_s0 (decoding D1) and, for reference, ablations_georef/cnn3d_final_s0,
both trained on 2004-2014. Validation 2015-2016 only; the test split is never used. Operating
point = best F1 at 10 km on validation.

Fire weather. Every ICNF record carries the Canadian FWI system values of its day and place
(SGIF, reporting only: ICNF never enters the model input). Wind speed is not stored, but the
initial spread index is ISI = 0.208 f(W) f(FFMC) with f(W) = exp(0.05039 W) (Van Wagner 1987), so
W [km/h, noon] = ln(ISI / (0.208 f(FFMC))) / 0.05039. Two levels:
  own      the target's own record (targets only)
  day      per point: median over all ICNF records (any size) of the UTC day of the period middle,
           nationwide ('national', no leak) or within 50 km excluding records <= 10 km from the point
           ('local'; nationwide if none). The local value can leak a little (records exist where
           fires are), so the national one is the reference.
  W1  targets by quartile of own wind and own FWI (wind bins: quartiles of the train targets
      2004-2014; FWI bins: evaluate.FWI_BINS): recall at the operating point, model and rule
      time error; the same for already-burning targets (wind below / above the train median)
  W2  re-scoring: logistic regression of hit vs. false alarm on logit(score) (+ day wind and FWI),
      fitted on one validation year and applied to the other; AP of the combined scores against
      the same with logit(score) only, 95 % interval by resampling periods
Cloud. Not-observable share of the 5 x 5 window around the target (cloud_diagnostic.unobs_share)
  C1  in the 3 h after the ICNF start ('unobs_3h', independent of when SEVIRI detected the fire) and
      from the start to the first detection <= 1.5 cells; first-detection lag; for the true
      positives of D1 @ 10 km: model and rule time error by cloud class (clear < 0.2, partly,
      cloudy >= 0.5), all and fresh targets
  C2  recall at the operating point by cloud class (both models)

Decisions, fixed before seeing the results:
  wind / fire weather  GO (get ERA5 wind for this model) if W2 gains >= 0.02 AP with an interval
                       above 0. WEAK if not, but the recall of the top and bottom own-wind quartile
                       (or of already-burning targets above / below the median) differs by >= 0.10
                       with an interval excluding 0, or the model's median |dt| does by >= 0.5 h.
                       NO-GO otherwise (wind stays an objective-(iv) ablation).
  cloud for time       WORTH A RUN if cloudy true positives are >= 10 % of all and their median
                       model |dt| is >= 0.5 h above that of clear ones, interval above 0.
                       Otherwise NO.

Run from the repository root:
  .venv/bin/python -u src/weather_cloud_checks.py | tee outputs/weather_cloud/checks.txt
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import eda_figures as E
from baselines import first_detection_times, training_lag_h
from cloud_diagnostic import Cloud, slot_of, unobs_share
from dataset import FireData
from download_msg_quality import DAYS_DIR
from evaluate import FWI_BINS, FWI_LABELS, average_precision, evaluate, target_context
from tiling_design import PERIOD_H

OUT = E.ROOT / "outputs" / "weather_cloud"
SPLIT, K_KM = "val", 10
BOOT = 1000
LOCAL_KM, EXCL_KM = 50, 10
BURN_PX = 1.5
CLEAR, CLOUDY = 0.2, 0.5
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- data

def wind_kmh(isi, ffmc):
    """Noon wind speed from ISI and FFMC (inverse of the ISI equation of the Canadian FWI system)."""
    isi, ffmc = np.asarray(isi, float), np.asarray(ffmc, float)
    m = 147.2 * (101 - ffmc) / (59.5 + ffmc)
    f_f = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(isi / (0.208 * f_f)) / 0.05039
    return np.where(isi > 0, w, np.nan)


def icnf_weather():
    a = pd.read_parquet(E.PROC / "icnf_fires.parquet",
                        columns=["fire_id", "start_utc", "lat", "lon", "fwi", "isi", "ffmc"])
    a["wind"] = wind_kmh(a["isi"], a["ffmc"])
    a = a.dropna(subset=["start_utc"])
    a["day"] = a["start_utc"].dt.floor("D")
    return a


def predictions(data):
    """D1 points of cnn3d_st with model and rule time, points of cnn3d_final; cached."""
    OUT.mkdir(parents=True, exist_ok=True)
    f_st, f_fin = OUT / "val_preds_cnn3d_st_s0.parquet", OUT / "val_preds_cnn3d_final_s0.parquet"
    if not (f_st.exists() and f_fin.exists()):
        from train import get_device, load_model, predict
        device = get_device()
        per = data.periods.loc[(data.periods["split"] == SPLIT) & (data.periods["n_det"] > 0), "period"]
        m, cfg, _ = load_model("ablations_st/cnn3d_st_s0", device)
        st = predict(m, data, per, cfg, device)
        lag = training_lag_h(data, "train")
        st["t_rule"] = np.clip(first_detection_times(st.drop(columns="t_h"), data)["t_h"] - lag, 0, PERIOD_H)
        st.to_parquet(f_st, index=False)
        m, cfg, _ = load_model("ablations_georef/cnn3d_final_s0", device)
        predict(m, data, per, cfg, device).to_parquet(f_fin, index=False)
        print("  predictions written", flush=True)
    return pd.read_parquet(f_st), pd.read_parquet(f_fin)


def operating(scored, curve):
    f1 = 2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]).clip(lower=1e-9)
    thr = float(curve["score"].iloc[int(f1.values.argmax())])
    return thr, set(scored.loc[(scored["score"] >= thr) & (scored["status"] == "tp"), "fire_id"])


def boot_diff(a, b, stat=np.mean, n=BOOT):
    """stat(b) - stat(a) and a 95 % interval, resampling each group."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan
    v = [stat(rng.choice(b, len(b))) - stat(rng.choice(a, len(a))) for _ in range(n)]
    return stat(b) - stat(a), *np.percentile(v, [2.5, 97.5])


def fmt(est, lo, hi):
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ---------------------------------------------------------------- fire weather

def day_values(points, a):
    """National and local (50 km, records <= 10 km excluded) median wind and FWI of the UTC day of
    the period middle."""
    day = (pd.to_datetime(points["period"], utc=True) + pd.Timedelta(hours=PERIOD_H / 2)).dt.floor("D")
    nat = a.groupby("day")[["wind", "fwi"]].median()
    out = pd.DataFrame(index=points.index)
    out["wind_nat"] = day.map(nat["wind"]).values
    out["fwi_nat"] = day.map(nat["fwi"]).values
    by_day = {d: g for d, g in a.groupby("day")}
    wl, fl = [], []
    for d, lat, lon in zip(day, points["lat"], points["lon"]):
        g = by_day.get(d)
        if g is None:
            wl.append(np.nan); fl.append(np.nan); continue
        dist = E.haversine_km(lat, lon, g["lat"].values, g["lon"].values)
        sel = (dist <= LOCAL_KM) & (dist > EXCL_KM)
        wl.append(np.nanmedian(g["wind"].values[sel]) if sel.any() else np.nan)
        fl.append(np.nanmedian(g["fwi"].values[sel]) if sel.any() else np.nan)
    out["wind_loc"] = np.where(np.isnan(wl), out["wind_nat"], wl)
    out["fwi_loc"] = np.where(np.isnan(fl), out["fwi_nat"], fl)
    return out


def logistic(X, y, ridge=1e-3, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + ridge * w
        Hm = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def rescore(table, feats):
    """Two-fold (by year) logistic re-scoring; returns the new score per row."""
    out = np.full(len(table), np.nan)
    lg = np.log(table["score"].clip(1e-6, 1 - 1e-6) / (1 - table["score"].clip(1e-6, 1 - 1e-6))).values
    F = np.column_stack([lg] + [table[f].values for f in feats])
    y = (table["status"] == "tp").values.astype(float)
    yr = table["year"].values
    for test in np.unique(yr):
        tr = yr != test
        mu, sd = np.nanmean(F[tr], 0), np.nanstd(F[tr], 0) + 1e-9
        Z = np.nan_to_num((F - mu) / sd)
        Z = np.column_stack([np.ones(len(Z)), Z])
        w = logistic(Z[tr], y[tr])
        out[~tr] = Z[~tr] @ w
    return out


def ap_of(table, col, n_targets):
    return average_precision(table.assign(score=table[col]), n_targets)[0]


def boot_ap_gain(table, base, new, targets_per_period, n=BOOT):
    periods = np.array(sorted(set(table["period"]) | set(targets_per_period.index)))
    idx = table.groupby("period").indices
    est = ap_of(table, new, targets_per_period.sum()) - ap_of(table, base, targets_per_period.sum())
    v = []
    for _ in range(n):
        pick = periods[rng.integers(0, len(periods), len(periods))]
        rows = np.concatenate([idx[p] for p in pick if p in idx]) if len(pick) else np.array([], int)
        t = table.iloc[rows]
        nt = targets_per_period.reindex(pick).fillna(0).sum()
        v.append(ap_of(t, new, nt) - ap_of(t, base, nt))
    return est, *np.percentile(v, [2.5, 97.5])


def fire_weather(data, a, st, fin):
    print("\n=== Fire weather (wind from ISI and FFMC) ===")
    own = a.set_index("fire_id")[["wind", "fwi"]]
    tc = target_context(data)
    res_s, curve_s, sc_s = evaluate(st, data, SPLIT, K_KM, groups=())
    res_f, curve_f, sc_f = evaluate(fin, data, SPLIT, K_KM, groups=())
    thr_s, hit_s = operating(sc_s, curve_s)
    thr_f, hit_f = operating(sc_f, curve_f)
    print(f"cnn3d_st D1: AP@10 {res_s['AP']:.3f}, operating point p >= {thr_s:.3f}; cnn3d_final: AP@10 {res_f['AP']:.3f}, p >= {thr_f:.3f}")

    lab = data.labels[data.labels["role"] == "target"]
    split_of = data.periods.set_index("period")["split"]
    tr = lab[lab["period"].map(split_of) == "train"].join(own, on="fire_id")
    wind_edges = tr["wind"].quantile([0.25, 0.5, 0.75]).values
    w_med = wind_edges[1]
    print(f"wind of the train targets (2004-2014), quartile edges {np.round(wind_edges, 1)} km/h; "
          f"share with a wind value {tr['wind'].notna().mean():.1%}")

    t = lab[lab["period"].map(split_of) == SPLIT].join(own, on="fire_id").join(tc, on="fire_id")
    t["hit_st"], t["hit_fin"] = t["fire_id"].isin(hit_s), t["fire_id"].isin(hit_f)
    tp = sc_s[sc_s["status"] == "tp"].set_index("fire_id")
    t = t.join(tp[["dt_h"]], on="fire_id")
    t["dt_rule"] = t["fire_id"].map(tp["t_rule"]) - t["t_h"]
    t["wind_q"] = pd.cut(t["wind"], np.r_[-np.inf, wind_edges, np.inf], labels=["q1", "q2", "q3", "q4"])
    t["fwi_q"] = pd.cut(t["fwi"], FWI_BINS, labels=FWI_LABELS, right=False)
    print(f"validation targets {len(t)}, with wind {t['wind'].notna().sum()}, with FWI {t['fwi'].notna().sum()}")

    def table(col, sub, name):
        g = sub.groupby(col, observed=False)
        tab = pd.DataFrame({
            "targets": g.size(), "wind [km/h]": g["wind"].median(), "FWI": g["fwi"].median(),
            "recall st": g["hit_st"].mean(), "recall final": g["hit_fin"].mean(),
            "already burning": g["already_burning"].mean(), "in pair": g["in_pair"].mean(),
            "|dt| model [h]": g["dt_h"].apply(lambda x: x.abs().median()),
            "|dt| rule [h]": g["dt_rule"].apply(lambda x: x.abs().median()),
            "dt model [h]": g["dt_h"].median()})
        print(f"\n{name}"); print(tab.round(3).to_string())

    table("wind_q", t, "W1  targets by own-wind quartile (recall at the operating point; time of the D1 true positives @ 10 km, all predictions)")
    table("fwi_q", t, "W1  targets by own-FWI class (evaluate.FWI_BINS)")
    q1, q4 = t[t["wind_q"] == "q1"], t[t["wind_q"] == "q4"]
    d_rec = boot_diff(q1["hit_st"], q4["hit_st"])
    d_dt = boot_diff(q1["dt_h"].abs(), q4["dt_h"].abs(), stat=np.median)
    print(f"\n    recall cnn3d_st, q4 - q1 wind: {fmt(*d_rec)}; cnn3d_final: {fmt(*boot_diff(q1['hit_fin'], q4['hit_fin']))}")
    print(f"    median |dt| model, q4 - q1 wind: {fmt(*d_dt)} h; rule: {fmt(*boot_diff(q1['dt_rule'].abs(), q4['dt_rule'].abs(), stat=np.median))} h")
    ab = t[t["already_burning"]]
    lo_w, hi_w = ab[ab["wind"] < w_med], ab[ab["wind"] >= w_med]
    d_ab = boot_diff(lo_w["hit_st"], hi_w["hit_st"])
    print(f"    already burning ({len(ab)}): wind < {w_med:.1f} km/h {len(lo_w)} targets, recall st {lo_w['hit_st'].mean():.3f}, "
          f"final {lo_w['hit_fin'].mean():.3f}; >= {len(hi_w)} targets, st {hi_w['hit_st'].mean():.3f}, final {hi_w['hit_fin'].mean():.3f}; "
          f"st high - low {fmt(*d_ab)}")
    fq1, fq4 = t[t["fwi_q"] == FWI_LABELS[0]], t[t["fwi_q"] == FWI_LABELS[-1]]
    print(f"    recall cnn3d_st, FWI q4 - q1: {fmt(*boot_diff(fq1['hit_st'], fq4['hit_st']))}")

    # W2: re-scoring with day-level weather
    s = sc_s[sc_s["status"] != "ignored"].copy()
    s["year"] = pd.to_datetime(s["period"], utc=True).dt.year
    s = pd.concat([s, day_values(s, a)], axis=1)
    print(f"\nW2  re-scoring, {len(s):,} points ({(s['status'] == 'tp').sum()} hits, {(s['status'] == 'fp').sum()} false alarms); "
          f"day value missing: national {s['wind_nat'].isna().mean():.1%}, local from 50 km records {np.mean(s['wind_loc'] != s['wind_nat']):.1%} of points")
    m = s.groupby("status")[["wind_nat", "fwi_nat", "wind_loc", "fwi_loc"]].median()
    print("    medians by status:\n" + m.round(2).to_string())
    n_tg = lab[lab["period"].map(split_of) == SPLIT].groupby("period").size()
    s["base"] = rescore(s, [])
    rows = []
    for name, feats in (("wind (national day)", ["wind_nat"]), ("wind + FWI (national day)", ["wind_nat", "fwi_nat"]),
                        ("wind (local day)", ["wind_loc"]), ("wind + FWI (local day)", ["wind_loc", "fwi_loc"])):
        s["new"] = rescore(s, feats)
        est, lo, hi = boot_ap_gain(s, "base", "new", n_tg)
        rows.append(dict(features=name, AP_base=ap_of(s, "base", n_tg.sum()), AP_new=ap_of(s, "new", n_tg.sum()),
                         gain=est, lo=lo, hi=hi))
    w2 = pd.DataFrame(rows)
    print(f"    AP of the raw scores {ap_of(s, 'score', n_tg.sum()):.3f}")
    print(w2.round(4).to_string(index=False))

    go = ((w2["gain"] >= 0.02) & (w2["lo"] > 0)).any()
    weak = any(abs(e) >= 0.10 and (lo_ > 0 or hi_ < 0) for e, lo_, hi_ in (d_rec, d_ab)) or \
        (abs(d_dt[0]) >= 0.5 and (d_dt[1] > 0 or d_dt[2] < 0))
    print(f"\nDECISION wind / fire weather: {'GO' if go else 'WEAK' if weak else 'NO-GO'}")
    return t


# ---------------------------------------------------------------- cloud

def cloud_check(data, t):
    print("\n=== Cloud cover and time error ===")
    first_day = pd.Timestamp("2015-01-01", tz="UTC") - timedelta(days=3)
    cloud = Cloud(DAYS_DIR, first_day, pd.Timestamp("2017-01-02", tz="UTC"))
    start = t["fire_id"].map(data.fires.set_index("fire_id")["start_utc"])
    s0 = np.array([slot_of(x) for x in start])
    slot = data.det["slot"].values
    di, dj = data._i, data._j
    end = np.array([slot_of(p) for p in t["period"]]) + (PERIOD_H + 6) * 4
    u3, ufirst, lag = [], [], []
    for a_, e_, fi, fj in zip(s0, end, t["i_f"].values, t["j_f"].values):
        i, j = int(round(fi)), int(round(fj))
        u3.append(cloud.unobs(a_, a_ + 12, i, j))
        lo, hi = np.searchsorted(slot, [a_ - 4, e_])
        near = np.hypot(di[lo:hi] - fi, dj[lo:hi] - fj) <= BURN_PX
        if near.any():
            f = int(slot[lo:hi][near].min())
            lag.append((f - a_) / 4)
            ufirst.append(unobs_share(cloud.window(a_, f, i, j)) if f > a_ else 0.0)
        else:
            lag.append(np.nan); ufirst.append(np.nan)
    t = t.assign(unobs_3h=u3, unobs_to_first=ufirst, first_lag_h=lag)
    t["cloud"] = pd.cut(t["unobs_3h"], [-0.01, CLEAR, CLOUDY, 1.0], labels=["clear", "partly", "cloudy"], right=False)
    print(f"targets {len(t)}; no cloud value (water / water-edge window) {t['unobs_3h'].isna().sum()}; "
          f"unobs_3h vs unobs_to_first correlation {t[['unobs_3h', 'unobs_to_first']].corr().iloc[0, 1]:.2f}")

    def block(sub, name):
        g = sub.groupby("cloud", observed=False)
        tp_ = sub[sub["dt_h"].notna()]
        gt = tp_.groupby("cloud", observed=False)
        tab = pd.DataFrame({
            "targets": g.size(), "recall st": g["hit_st"].mean(), "recall final": g["hit_fin"].mean(),
            "first-det lag [h]": g["first_lag_h"].median(), "unobs to first det": g["unobs_to_first"].mean(),
            "TPs": gt.size(), "|dt| model [h]": gt["dt_h"].apply(lambda x: x.abs().median()),
            "dt model [h]": gt["dt_h"].median(), "model <= 1 h": gt["dt_h"].apply(lambda x: (x.abs() <= 1).mean()),
            "|dt| rule [h]": gt["dt_rule"].apply(lambda x: x.abs().median()), "dt rule [h]": gt["dt_rule"].median()})
        print(f"\n{name}"); print(tab.round(3).to_string())
        c, k = tp_[tp_["cloud"] == "clear"], tp_[tp_["cloud"] == "cloudy"]
        d = boot_diff(c["dt_h"].abs(), k["dt_h"].abs(), stat=np.median)
        dr = boot_diff(c["dt_rule"].abs(), k["dt_rule"].abs(), stat=np.median)
        print(f"    median |dt| cloudy - clear: model {fmt(*d)} h, rule {fmt(*dr)} h; cloudy share of TPs {len(k) / max(len(tp_), 1):.1%}")
        return d, len(k) / max(len(tp_), 1)

    d_all, share = block(t, "C1/C2  all validation targets by cloud in the 3 h after the ICNF start")
    block(t[~t["already_burning"]], "C1/C2  fresh targets only")
    # cloud between start and first detection, for the true positives
    tp_ = t[t["dt_h"].notna() & t["unobs_to_first"].notna()].copy()
    tp_["cloud_first"] = pd.cut(tp_["unobs_to_first"], [-0.01, CLEAR, CLOUDY, 1.0], labels=["clear", "partly", "cloudy"], right=False)
    g = tp_.groupby("cloud_first", observed=False)
    print("\nC1  true positives by cloud between the ICNF start and the first detection")
    print(pd.DataFrame({"TPs": g.size(), "first-det lag [h]": g["first_lag_h"].median(),
                        "|dt| model [h]": g["dt_h"].apply(lambda x: x.abs().median()), "dt model [h]": g["dt_h"].median(),
                        "|dt| rule [h]": g["dt_rule"].apply(lambda x: x.abs().median()), "dt rule [h]": g["dt_rule"].median()}).round(3).to_string())
    worth = share >= 0.10 and d_all[0] >= 0.5 and d_all[1] > 0
    print(f"\nDECISION cloud channel for time: {'WORTH A RUN' if worth else 'NO'}")


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    data = FireData()
    a = icnf_weather()
    v = a[a["start_utc"].dt.year.between(2015, 2016)]
    print(f"ICNF records 2015-2016: {len(v):,}; with wind {v['wind'].notna().mean():.1%}; wind median {v['wind'].median():.1f} km/h, "
          f"IQR {v['wind'].quantile(.25):.1f}-{v['wind'].quantile(.75):.1f}, max {v['wind'].max():.1f}")
    st, fin = predictions(data)
    t = fire_weather(data, a, st, fin)
    cloud_check(data, t)


if __name__ == "__main__":
    main()
