"""Predicted ignitions vs. ICNF targets of a final model on the TEST split 2017-2019 (notes.md 6.17).

Descriptive figures made after the single locked test evaluation (evaluate_test.py); nothing here
is used for any model choice. Periods are chosen by fixed rules (random choice with seed 0 inside
each case type, no hand-picking). Operating point = best F1 at 10 km on the test split (the
threshold behind "false alarms / day @ best F1" in outputs/final/test_results.md).
  outbreak      the test period with the most targets
  hit           a period with exactly 1 target that is found
  miss          a period with a missed target
  merged pair   a period with two targets ≤ 10 km apart and at least one of them missed
  false alarm   a period without targets but with a prediction above the threshold
  random 1, 2   periods with ≥ 1 target, no condition on the outcome
Columns: (a) input only (hour of first SEVIRI detection, no ICNF information), (b) predicted
probability with the predicted points by match status, (c) ICNF targets, target heatmap and
loss mask. All positions in the sensor frame (labels and peaks before the shift back, 6.12).

Run from the repository root:
  .venv/bin/python src/plot_test_examples.py --name final/cnn3d_final_s0
Outputs: figures/final/test_examples_{1,2}.png, outputs/final/test_examples.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from dataset import FireData
from evaluate import evaluate
from train import get_device, heatmap_to_points, load_model, predict_maps, sample_kwargs

K_KM = 10
PROB_VMAX = 0.6
STATUS_STYLE = {"tp": dict(color=E.BLUE, label="predicted, hit (≤ 10 km)"),
                "fp": dict(color=E.AQUA, label="predicted, false alarm"),       # not orange: it would vanish in the heatmap
                "ignored": dict(color=E.INK2, label="predicted, near an ignore fire")}


def points(maps, georef_fix):
    """Peaks of every map with sensor-frame (fi, fj) and true-frame (lat, lon) side by side."""
    rows = []
    for p, prob in maps.items():
        fi, fj, s = heatmap_to_points(prob)
        ti, tj = G.sensor_to_true_ij(fi, fj, p) if georef_fix else (fi, fj)
        lat, lon = G.ij_to_latlon(ti, tj)
        rows.append(pd.DataFrame({"period": p, "lat": lat, "lon": lon, "score": s, "fi": fi, "fj": fj}))
    return pd.concat(rows, ignore_index=True)


def outcomes(data, scored, thr):
    """Per test period: targets, hits, misses, false alarms at the threshold; per target: hit or not."""
    above = scored[scored["score"] >= thr]
    hit_ids = set(above.loc[above["status"] == "tp", "fire_id"])
    lab = data.labels[(data.labels["role"] == "target")]
    lab = lab[lab["period"].isin(set(data.periods.loc[data.periods["split"] == "test", "period"]))].copy()
    lab["hit"] = lab["fire_id"].isin(hit_ids)
    per = data.periods[data.periods["split"] == "test"].set_index("period")
    per["hits"] = lab.groupby("period")["hit"].sum().reindex(per.index, fill_value=0)
    per["misses"] = per["n_targets"] - per["hits"]
    per["false_alarms"] = above[above["status"] == "fp"].groupby("period").size().reindex(per.index, fill_value=0)
    return per, lab


def merged_pair_periods(lab):
    """Periods with two targets ≤ K_KM apart (true positions) and at least one of them missed."""
    fires = lab.groupby("period")
    out = []
    for p, g in fires:
        if len(g) < 2 or g["hit"].all():
            continue
        d = E.haversine_km(g["lat"].values[:, None], g["lon"].values[:, None], g["lat"].values[None], g["lon"].values[None])
        close = (d <= K_KM) & ~np.eye(len(g), dtype=bool)
        if (close & ~(g["hit"].values[:, None] & g["hit"].values[None])).any():
            out.append(p)
    return out


def choose_cases(per, lab):
    rng = np.random.default_rng(0)
    pick = lambda idx: idx[int(rng.integers(len(idx)))]
    pos = per[per["n_targets"] > 0]
    rand = list(pos.index[rng.choice(len(pos), 2, replace=False)])
    return [("outbreak (most targets)", per["n_targets"].idxmax()),
            ("hit (1 target, found)", pick(per.index[(per["n_targets"] == 1) & (per["hits"] == 1)])),
            ("miss (a target not found)", pick(per.index[(per["n_targets"] == 1) & (per["misses"] == 1)])),
            ("merged pair (≤ 10 km apart, one missed)", pick(merged_pair_periods(lab))),
            ("false alarm (no target)", pick(per.index[(per["n_targets"] == 0) & (per["false_alarms"] > 0)])),
            ("random period with a target", rand[0]),
            ("random period with a target", rand[1])]


def window(data, pts, lab_p, margin=5, min_size=22):
    """Zoom window (i0, i1, j0, j1) around the targets and the predictions above the threshold."""
    ii = np.r_[lab_p["i_f"].values, pts["fi"].values]
    jj = np.r_[lab_p["j_f"].values, pts["fj"].values]
    if not len(ii):
        ii, jj = np.nonzero(data.static["pt"])
    ci, cj = (ii.min() + ii.max()) / 2, (jj.min() + jj.max()) / 2
    h = max((ii.max() - ii.min()) / 2 + margin, (jj.max() - jj.min()) / 2 + margin, min_size / 2)   # square panels
    return ci - h, ci + h, cj - h, cj + h


def plot_rows(cases, data, maps, pts_all, thr, name, out, bi, bj, cfg):
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 4.0 * len(cases)), layout="constrained")
    axes = np.atleast_2d(axes)
    for r, (label, period) in enumerate(cases):
        x, y, w, lab_p = data.sample(period, **sample_kwargs(cfg))
        prob = maps[period]
        pts = pts_all[(pts_all["period"] == period) & (pts_all["score"] >= thr)]
        tg, ig = lab_p[lab_p["role"] == "target"], lab_p[lab_p["role"] == "ignore"]
        det = x[:, 2] > 0
        first_h = np.where(det.any(0), det.argmax(0) / 4, np.nan)
        i0, i1, j0, j1 = window(data, pts, tg)
        n_hit = int((pts["status"] == "tp").sum())
        n_fa = int((pts["status"] == "fp").sum())

        panels = [(first_h, E.SEQ_BLUE.reversed(), (0, 24), "(a) input: hour of first SEVIRI detection (SEVIRI only)"),
                  (prob, E.SEQ_ORANGE, (0, PROB_VMAX), f"(b) predicted probability, points p ≥ {thr:.2f}"),
                  (y, E.SEQ_ORANGE, (0, 1), "(c) ICNF targets and target heatmap (grey = no loss)")]
        for c, (img, cmap, (v0, v1), title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            if c == 2:
                ax.imshow(np.where(w == 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3, alpha=0.5, interpolation="nearest")
            ax.plot(bj, bi, color=E.MUTED, lw=0.8)
            if c == 1:
                for st, sty in STATUS_STYLE.items():
                    P = pts[pts["status"] == st]
                    ax.scatter(P["fj"], P["fi"], marker="x", s=55, lw=1.8, zorder=6, color=sty["color"], label=sty["label"])
                # where the targets are, as hollow rings, so hits and misses can be read off panel (b)
                ax.scatter(tg["j_f"], tg["i_f"], marker="o", s=110, facecolor="none", edgecolor=E.INK, lw=0.9,
                           zorder=5, label="ICNF target (≥ 50 ha)")
            if c == 2:
                ax.scatter(ig["j_f"], ig["i_f"], marker="x", s=30, lw=1.1, color=E.ORANGE, zorder=4, label="ICNF ignore fire")
                ax.scatter(tg["j_f"], tg["i_f"], marker="+", s=80, lw=1.6, color=E.INK, zorder=5, label="ICNF target (≥ 50 ha)")
            ax.set_xlim(j0, j1); ax.set_ylim(i1, i0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.7, label=("hour in the 24 h window (6–18 = period)", "p", "y")[c])
        axes[r, 0].set_ylabel(f"{label}\n{period:%Y-%m-%d %H:%M} UTC + 12 h\n"
                              f"{len(tg)} targets: {n_hit} hit, {len(tg) - n_hit} missed; {n_fa} false alarms", fontsize=9)
    axes[0, 1].legend(loc="lower left", fontsize=7, framealpha=0.9)
    axes[0, 2].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle(f"{name} on the test split 2017–2019: prediction vs. ICNF target; one cell ≈ 3.2 × 4.5 km, sensor frame",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"figure: {out.relative_to(E.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="final/cnn3d_final_s0")
    args = ap.parse_args()
    device = get_device()
    model, cfg, _ = load_model(args.name, device)
    data = FireData()
    assert data.georef_fix, "expects the geo-referencing-corrected build (6.12)"

    periods = data.periods.loc[(data.periods["split"] == "test") & (data.periods["n_det"] > 0), "period"]
    maps = predict_maps(model, data, periods, cfg, device)
    pts = points(maps, data.georef_fix)
    res, curve, scored = evaluate(pts[["period", "lat", "lon", "score"]], data, "test", K_KM)
    f1 = 2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]).clip(lower=1e-9)
    thr = float(curve["score"].iloc[int(f1.values.argmax())])
    print(f"{args.name}: test AP @ {K_KM} km {res['AP']:.3f}; best-F1 threshold p ≥ {thr:.3f}")

    # match status back onto the sensor-frame points (evaluate keeps the row index of its input)
    pts = pts.join(scored[["status", "fire_id", "err_km"]], how="inner")
    per, tlab = outcomes(data, scored, thr)
    tlab = tlab.merge(data.fires[["fire_id", "lat", "lon"]], on="fire_id")
    cases = choose_cases(per, tlab)
    for label, p in cases:
        print(f"  {label:42s} {p:%Y-%m-%d %H:%M}  targets {per.at[p, 'n_targets']}, hits {per.at[p, 'hits']}, "
              f"false alarms {per.at[p, 'false_alarms']}")

    _, mainland = E.load_borders()
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    fig_dir = E.ROOT / "figures" / "final"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = args.name.split("/")[-1]
    plot_rows(cases[:4], data, maps, pts, thr, tag, fig_dir / "test_examples_1.png", bi, bj, cfg)
    plot_rows(cases[4:], data, maps, pts, thr, tag, fig_dir / "test_examples_2.png", bi, bj, cfg)
    pd.DataFrame([dict(case=l, period=p, **per.loc[p, ["n_targets", "hits", "misses", "false_alarms"]].to_dict())
                  for l, p in cases]).assign(model=tag, threshold=thr).to_csv(
        E.ROOT / "outputs" / "final" / "test_examples.csv", index=False)
    print("written: outputs/final/test_examples.csv")


if __name__ == "__main__":
    main()
