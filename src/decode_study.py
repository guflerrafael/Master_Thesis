"""Decode-only options for the space-time model on validation (notes.md 9.21; planned in 9.4 items 6-7, 9.6 step 3).

No training. The maps of the three dev runs ablations_all/cnn3d_st_all_s0-2 (trained without 2015-2016) are
decoded in several ways and scored on validation 2015-2016; the test split is never used, so whatever is
chosen here can still be pre-registered for a future test:
  D1          default: 3D peaks over 3 bins x 3 x 3 cells, 3 x 3 centroid in the peak's bin, peak value as score
  D1 lp       position from a log-parabola per axis through the peak and its two neighbours (exact for a
              Gaussian peak; the separable form of DARK, Zhang et al. 2020; model_review.md 3.1)
  D1 tau2     duplicate window +-2 h (5 bins): a peak is dropped if a stronger one lies within 1 cell and 2 h
  D1 noisy_or score = 1 - prod(1 - p) over the peak hour +-1 h; D1 x2d: peak x auxiliary 2D map
  ensemble    mean of the three seeds' probability maps (3D and 2D), then D1 / D1 lp
Scores: AP @ 5 / 10 km (place only, the primary protocol), AP with tau = 2 h @ 10 km, recall in pair @ 10 km,
median error of the hits @ 5 km, number of points.

Run from the repository root:
  .venv/bin/python src/decode_study.py > outputs/error_analysis/decode_study_val.txt
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import grid as G
from dataset import FireData
from evaluate import evaluate
from train import _centroids, get_device, iter_maps, load_model, peak_time

RUNS = [f"ablations_all/cnn3d_st_all_s{s}" for s in range(3)]
SPLIT = "val"


def _lp_offsets(m, ii, jj):
    """Sub-pixel offsets from a log-parabola per axis through the peak cell and its two neighbours
    (bounded to half a cell; no shift at the grid edge or if the log-profile is not concave)."""
    H, W = m.shape
    lm = np.log(np.clip(m.astype(float), 1e-6, 1.0))
    di, dj = np.zeros(len(ii)), np.zeros(len(ii))
    for n, (a, b) in enumerate(zip(ii, jj)):
        if 0 < a < H - 1:
            den = lm[a - 1, b] - 2 * lm[a, b] + lm[a + 1, b]
            if den < 0:
                di[n] = np.clip(0.5 * (lm[a - 1, b] - lm[a + 1, b]) / den, -0.5, 0.5)
        if 0 < b < W - 1:
            den = lm[a, b - 1] - 2 * lm[a, b] + lm[a, b + 1]
            if den < 0:
                dj[n] = np.clip(0.5 * (lm[a, b - 1] - lm[a, b + 1]) / den, -0.5, 0.5)
    return ii + di, jj + dj


def decode3d(p3, p2=None, window_t=3, score="peak", subpixel="centroid", thr=0.05, topk=50):
    """D1 with options (same peak rule as train.heatmap_to_points_3d)."""
    p = torch.as_tensor(p3)[None, None]
    pool = F.max_pool3d(p, (window_t, 3, 3), stride=1, padding=(window_t // 2, 1, 1))
    kk, ii, jj = (v.numpy() for v in torch.nonzero(((p == pool) & (p >= thr))[0, 0], as_tuple=True))
    s = p3[kk, ii, jj].astype(float)
    if score == "noisy_or":
        s = np.array([1 - np.prod(1 - p3[max(k - 1, 0):k + 2, a, b]) for k, a, b in zip(kk, ii, jj)])
    elif score == "x2d":
        s = s * p2[ii, jj]
    order = np.argsort(-s, kind="stable")[:topk]
    kk, ii, jj, s = kk[order], ii[order], jj[order], s[order]
    fi, fj = np.zeros(len(kk)), np.zeros(len(kk))
    for n, (k, a, b) in enumerate(zip(kk, ii, jj)):
        if subpixel == "lp":
            fi[n:n + 1], fj[n:n + 1] = _lp_offsets(p3[k], np.array([a]), np.array([b]))
        else:
            fi[n:n + 1], fj[n:n + 1] = _centroids(p3[k], [a], [b])
    t = np.array([peak_time(p3, k, a, b, 1.0) for k, a, b in zip(kk, ii, jj)])
    return fi, fj, s, t


def to_points(period, fi, fj, s, t, georef_fix):
    if georef_fix:
        fi, fj = G.sensor_to_true_ij(fi, fj, period)
    lat, lon = G.ij_to_latlon(fi, fj)
    return pd.DataFrame({"period": period, "lat": lat, "lon": lon, "score": s, "t_h": t})


VARIANTS = {"D1": {}, "D1 lp": dict(subpixel="lp"), "D1 tau2": dict(window_t=5), "D1 noisy_or": dict(score="noisy_or"),
            "D1 x2d": dict(score="x2d")}
ENS_VARIANTS = {"ensemble D1": {}, "ensemble D1 lp": dict(subpixel="lp"), "ensemble D1 lp tau2": dict(subpixel="lp", window_t=5)}


def main():
    data = FireData()
    device = get_device()
    models = [load_model(r, device)[:2] for r in RUNS]
    periods = list(data.periods.loc[(data.periods["split"] == SPLIT) & (data.periods["n_det"] > 0), "period"])
    pts = {(v, s): [] for v in VARIANTS for s in range(len(RUNS))}
    pts.update({(v, -1): [] for v in ENS_VARIANTS})
    gens = [iter_maps(m, data, periods, cfg, device) for m, cfg in models]
    for batch in zip(*gens):                                  # the same periods for every seed
        for p in batch[0]:
            maps = [b[p] for b in batch]
            for s, (p3, p2) in enumerate(maps):
                for v, kw in VARIANTS.items():
                    pts[(v, s)].append(to_points(p, *decode3d(p3, p2, **kw), data.georef_fix))
            e3 = np.mean([m[0] for m in maps], 0)
            e2 = np.mean([m[1] for m in maps], 0)
            for v, kw in ENS_VARIANTS.items():
                pts[(v, -1)].append(to_points(p, *decode3d(e3, e2, **kw), data.georef_fix))
    rows = []
    for (v, s), parts in pts.items():
        p = pd.concat(parts, ignore_index=True)
        r = {"decoding": v, "seed": s, "points": len(p)}
        for k in (5, 10):
            res = evaluate(p, data, SPLIT, k, groups=("in_pair",))[0]
            r[f"AP@{k}"] = res["AP"]
            if k == 5:
                r["median error TP @5 [km]"] = res["median error TP [km]"]
            else:
                r["recall in pair @10"] = res.get("recall | in_pair=True", np.nan)
                r["recall @10"] = res["recall (all preds)"]
        r["AP tau=2h @10"] = evaluate(p, data, SPLIT, 10, groups=(), tau_h=2)[0]["AP"]
        rows.append(r)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"Decode-only options, validation 2015-2016, runs {RUNS[0][:-1]}0-2 (test never used)\n")
    print("Per seed:")
    print(t.round(4).to_string(index=False))
    single = t[t["seed"] >= 0].groupby("decoding", sort=False).mean(numeric_only=True).drop(columns="seed")
    ens = t[t["seed"] < 0].set_index("decoding").drop(columns="seed")
    out = pd.concat([single, ens])
    base = out.loc["D1"]
    print("\nMean of 3 seeds (single models) and the 3-seed ensemble; differences to D1 in the last columns:")
    out["dAP@5"], out["dAP@10"], out["dAP tau2"] = out["AP@5"] - base["AP@5"], out["AP@10"] - base["AP@10"], out["AP tau=2h @10"] - base["AP tau=2h @10"]
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
