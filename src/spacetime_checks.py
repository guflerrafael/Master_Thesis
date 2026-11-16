"""Step 1 of the space-time ignition heatmap (notes.md 9.6): checks of the targets, loss weights,
flips, decoder and evaluation, and the time error of the rule-based time baselines. No training.

  A  space-time targets and weights (all non-test target periods, full grid, K = 12, sigma_t = 1 h):
     every target has >= 1 positive voxel (pos_threshold 0.667) with weight > 0; max over the bins
     of y3 <= y2; w3 <= w2; the 2D part equals the time_bins = 0 sample
  B  flips mirror space only: the hour of every target peak is unchanged
  C  decoder on synthetic Gaussians and on the real targets: time from the log-parabola,
     position identical to the 2D decoder inside the peak's bin; D2 (max over bins) = 2D decoder
  D  unchanged scores: nucleus-birth baseline and ablations_georef/cnn3d_final_s0-2 on validation
  E  time baselines on validation (train for the lag): nucleus birth, first detection near the
     cnn3d_final peaks, each raw and with the training median lag removed; time error of the hits by
     subgroup and AP with a time tolerance tau = 1, 2, 3 h
The test split (2024-2025) is never used.

Run from the repository root:
  .venv/bin/python src/spacetime_checks.py > outputs/spacetime_checks.txt
"""
import numpy as np
import pandas as pd

from baselines import first_detection_times, predictions as baseline_predictions, training_lag_h
from dataset import FireData, crop_origin
from evaluate import evaluate, target_context
from train import decode, flip, get_device, heatmap_to_points, load_model, pos_threshold, predict
from tiling_design import ACQ_OFFSET_H

K, SIGMA_T = 12, 1.0
SPLITS_DEV = ("train", "val", "train2", "train3")     # never 'test'
GROUPS = ("isolated", "in_pair", "already_burning", "rekindle")


def part_a(data):
    thr = pos_threshold(1.0, SIGMA_T, 12 / K)
    per = data.periods[data.periods["split"].isin(SPLITS_DEV) & (data.periods["n_targets"] > 0)]
    n_t = no_pos = no_w = bad_max = bad_w = bad_2d = 0
    npos = []
    for p in per["period"]:
        _, y, w, lab = data.sample(p)
        _, (y3, y2), (w3, w2), lab3 = data.sample(p, time_bins=K, sigma_t=SIGMA_T)
        bad_2d += not (np.array_equal(y, y2) and np.array_equal(w, w2))
        bad_max += int((y3.max(0) > y2 + 1e-6).sum())
        bad_w += int((w3 > w2[None] + 1e-6).sum())
        r, c = np.mgrid[0:y2.shape[0], 0:y2.shape[1]]
        for q in lab3[lab3["role"] == "target"].itertuples():
            n_t += 1
            t_h = (data._start[q.fire_id] - p).total_seconds() / 3600
            tc = (np.arange(K) + 0.5) * 12 / K
            own = (np.exp(-(tc - t_h) ** 2 / (2 * SIGMA_T ** 2))[:, None, None]
                   * np.exp(-((r - q.i_loc) ** 2 + (c - q.j_loc) ** 2) / 2)[None])
            pos = (own >= thr) & (y3 >= thr)
            npos.append(pos.sum())
            no_pos += pos.sum() == 0
            no_w += (pos & (w3 > 0)).sum() == 0
    print(f"A  {len(per):,} periods, {n_t:,} targets (K = {K}, sigma_t = {SIGMA_T} h, positive threshold {thr:.3f})")
    print(f"   targets without a positive voxel: {no_pos}; without a positive voxel of weight > 0: {no_w}; "
          f"positive voxels per target: mean {np.mean(npos):.2f}, min {np.min(npos)}")
    print(f"   cells with max_k y3 > y2: {bad_max}; voxels with w3 > w2: {bad_w}; periods whose 2D part differs "
          f"from the time_bins = 0 sample: {bad_2d}")


def part_b(data, n=200, seed=0):
    rng = np.random.default_rng(seed)
    tg = data.labels[(data.labels["role"] == "target")
                     & data.labels["period"].isin(data.periods.loc[data.periods["split"] == "train", "period"])]
    bad = 0
    for q in tg.sample(n, random_state=seed).itertuples():
        crop = crop_origin(q.i_f, q.j_f, 32, rng, 8)
        x, (y3, y2), (w3, w2), lab = data.sample(q.period, crop, time_bins=K)
        a, b = int(round(q.i_f)) - crop[0], int(round(q.j_f)) - crop[1]
        if not (0 <= a < 32 and 0 <= b < 32):
            continue
        k0 = int(y3[:, a, b].argmax())
        for ax, (a2, b2) in ((-1, (a, 31 - b)), (-2, (31 - a, b))):
            f3, f2 = flip((y3, y2), ax)
            fw3, _ = flip((w3, w2), ax)
            fx = flip(x, ax)
            bad += (int(f3[:, a2, b2].argmax()) != k0 or f3[k0, a2, b2] != y3[k0, a, b] or fw3[k0, a2, b2] != w3[k0, a, b]
                    or not np.array_equal(fx[:, :, a2, b2], x[:, :, a, b]))
    print(f"B  flips on {n} training crops (both axes): hour, value, weight or input changed at the target: {bad}")


def part_c(data, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[0:15, 0:15]
    tc = np.arange(K) + 0.5
    et, es = [], []
    for _ in range(n):
        i, j, t = 7 + rng.uniform(-0.5, 0.5), 7 + rng.uniform(-0.5, 0.5), rng.uniform(1, 11)
        g = np.exp(-((r - i) ** 2 + (c - j) ** 2) / 2)
        y3 = (np.exp(-(tc - t) ** 2 / 2)[:, None, None] * g[None]).astype(np.float32)
        fi, fj, s, th = decode(y3, "3d")
        fi2, fj2, _ = heatmap_to_points(y3[int(round(th[0] - 0.5))])
        et.append(th[0] - t); es.append(max(abs(fi[0] - fi2[0]), abs(fj[0] - fj2[0])))
    et = np.abs(et)
    print(f"C  synthetic Gaussians (t in [1, 11) h, {n}): |time error| max {et.max():.2e} h; "
          f"position minus 2D decoder in the same bin: max {max(es):.2e} cells")
    # real targets of validation: decode the target tensors themselves
    ctx = target_context(data)
    per = data.periods[(data.periods["split"] == "val") & (data.periods["n_targets"] > 0)]
    rows = []
    for p in per["period"]:
        _, (y3, y2), _, lab = data.sample(p, time_bins=K)
        tg = lab[lab["role"] == "target"]
        for mode in ("3d", "max"):
            fi, fj, s, th = decode(y3, mode, thr=0.3)
            for q in tg.itertuples():
                d = np.hypot(fi - q.i_loc, fj - q.j_loc)
                m = int(d.argmin()) if len(d) else -1
                rows.append(dict(mode=mode, fire_id=q.fire_id, d=d[m] if m >= 0 else np.inf,
                                 dt=th[m] - ctx.loc[q.fire_id, "t_h"] if m >= 0 else np.nan))
        f2, g2, _ = heatmap_to_points(y2, thr=0.3)
        fm, gm, _, _ = decode(y3, "max", thr=0.3)
        rows.append(dict(mode="max == 2D decoder of max_k y3", fire_id=-1, d=float(np.allclose(np.sort(fm), np.sort(heatmap_to_points(y3.max(0), thr=0.3)[0]))), dt=np.nan))
    r_ = pd.DataFrame(rows)
    for mode in ("3d", "max"):
        g = r_[r_["mode"] == mode]
        ok = g["d"] <= 0.5
        print(f"   decoding '{mode}' of the validation target tensors: nearest peak <= 0.5 cells for {ok.mean():.1%} of "
              f"{len(g)} targets; |time error| of those: median {g.loc[ok, 'dt'].abs().median():.3f} h, "
              f"<= 0.25 h {np.mean(g.loc[ok, 'dt'].abs() <= 0.25):.1%}, max {g.loc[ok, 'dt'].abs().max():.2f} h")
    print(f"   D2 equals the 2D decoder applied to max_k y3: {r_.loc[r_['fire_id'] == -1, 'd'].mean():.0%} of periods")


def score_table(name, preds, data, split="val"):
    rows = []
    for k in (5, 10):
        res, _, _ = evaluate(preds, data, split, k, groups=GROUPS)
        row = {"method": name, "k": k, "AP": res["AP"], "recall": res["recall (all preds)"]}
        if "median |dt| TP [h]" in res:
            row.update({"median |dt| [h]": res["median |dt| TP [h]"], "median dt [h]": res["median dt TP [h]"],
                        "|dt| <= 1 h": res["|dt| <= 1 h"], "|dt| <= 2 h": res["|dt| <= 2 h"]})
            for tau in (1, 2, 3):
                row[f"AP tau={tau} h"] = evaluate(preds, data, split, k, groups=(), tau_h=tau)[0]["AP"]
            row.update({c.replace("median |dt| | ", "|dt| "): v for c, v in res.items() if c.startswith("median |dt| | ")})
        rows.append(row)
    return rows


def main():
    pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60)
    data = FireData()
    part_a(data); print()
    part_b(data); print()
    part_c(data); print()

    lag = training_lag_h(data, "train")
    print(f"D/E  acquisition offset {ACQ_OFFSET_H * 60:.0f} min; median lag first detection (+ offset) - ICNF start, "
          f"train targets: {lag:+.2f} h\n")
    rows = []
    base = baseline_predictions(data)["birth"]
    rows += score_table("nucleus birth", base, data)
    rows += score_table("nucleus birth, lag removed", base.assign(t_h=base["t_h"] - lag), data)
    device = get_device()
    for s in (0, 1, 2):
        model, cfg, ck = load_model(f"ablations_georef/cnn3d_final_s{s}", device)
        per = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
        pts = predict(model, data, per, cfg, device)
        stored = ck["val"]
        rows += [dict(r, stored_AP=stored[f"AP@{r['k']}"]) for r in score_table(f"cnn3d_final_s{s} (no time)", pts, data)]
        pf = first_detection_times(pts, data)
        rows += score_table(f"cnn3d_final_s{s} + first detection", pf, data)
        rows += score_table(f"cnn3d_final_s{s} + first detection, lag removed", pf.assign(t_h=np.clip(pf["t_h"] - lag, 0, 12)), data)
        print(f"   cnn3d_final_s{s}: {len(pts):,} points, {pf['t_h'].isna().mean():.1%} without a detection within 1.5 cells", flush=True)
    tab = pd.DataFrame(rows)
    tab["method"] = tab["method"].str.replace(r"_s\d", "", regex=True)
    num = [c for c in tab.columns if c not in ("method", "k")]
    g = tab.groupby(["method", "k"], sort=False)[num]
    out = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
    out = out.replace({"nan ± 0.0": "", "nan ± nan": ""})
    print("\nValidation 2015-2016 (cnn3d_final: mean ± std over 3 seeds; baseline deterministic):")
    main_cols = ["AP", "stored_AP", "recall", "median |dt| [h]", "median dt [h]", "|dt| <= 1 h", "|dt| <= 2 h",
                 "AP tau=1 h", "AP tau=2 h", "AP tau=3 h"]
    print(out[[c for c in main_cols if c in out]].to_string())
    print("\nmedian |dt| of the hits by subgroup [h]:")
    print(out[[c for c in out.columns if c.startswith("|dt| ") and "<=" not in c]].to_string())


if __name__ == "__main__":
    main()
