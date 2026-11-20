"""Where cnn3d_st struggles: example periods chosen by fixed rules, with a space-time diagram (notes.md 9.21).

Descriptive figures. Cases are picked from the outcome of the space-time model at its best-F1 threshold at
10 km (random choice with seed 0 inside each case type, every period used once, no hand-picking):
  weak signal          a target found only below the threshold, with <= 5 detections in its first 6 h or an FRE
                       in the lowest quartile of the split's targets (not in a pair, not next to an active fire)
  next to active fire  a target found only below the threshold that is already burning / seen via an older nucleus
  close pair           a target <= 2 cells from another target of its period, not hit
  re-flare             a false alarm above the threshold at a place that burned in the 72 h before the input window
  unexplained          a false alarm above the threshold with no ICNF record within 10 km and no earlier activity
  time duplicate       a false alarm above the threshold within 10 km of a hit of its period, >= 2 h from it
  late / early time    a hit whose model time is >= 3 h off the ICNF start
  outbreak             the period with the most targets
Columns: (a) input: hour of the first SEVIRI detection in the 24 h window; (b) space-time diagram around the
focus: every detection <= 7 cells away from 72 h before the period to the end of the window (distance vs
hours since the period start, colour = FRP); the input window is shaded; ICNF records nearby are marked;
(c) cnn3d_final probability and points above its threshold; (d) cnn3d_st maximum over the 12 hours and its
points with their hour; (e) cnn3d_st probability by hour at the focus with the ICNF start(s).
Positions in the sensor frame (as the maps).

Run from the repository root:
  .venv/bin/python src/plot_failure_examples.py --split val
  .venv/bin/python src/plot_failure_examples.py --split test --confirm-test      (descriptive, after notes.md 9.20)
Outputs: figures/error_analysis/<split>_failure_examples_{1,2}.png, outputs/error_analysis/<split>_failure_examples.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from dataset import FireData
from error_analysis import best_f1_threshold, cached_points, fp_classes, icnf_small, point_signal, target_table
from evaluate import evaluate
from tiling_design import HISTORY_H, PERIOD_H, TAIL_H
from train import get_device, load_model, predict_maps

RUNS = {"val": ("ablations_all/cnn3d_st_all_s0", "ablations_all/cnn3d_final_all_s0"),
        "test": ("final_2023/cnn3d_st_s0", "final_2023/cnn3d_final_s0")}
K_KM, VMAX, HALF, R_DIAG = 10, 0.6, 11, 7.0
SLUGS = {"weak signal": "weak_signal", "next to an active fire": "active_fire", "close pair": "close_pair", "re-flare": "reflare",
         "unexplained": "unexplained_false_alarm", "time duplicate": "time_duplicate", "hit, but": "time_off", "outbreak": "outbreak"}
STATUS = {"tp": (E.BLUE, "x", "hit (≤ 10 km)"), "fp": (E.AQUA, "X", "false alarm"), "ignored": (E.INK2, "+", "near an ignore fire")}


def sensor_ij(df):
    i, j = G.latlon_to_ij(df["lat"].values, df["lon"].values)
    return G.true_to_sensor_ij(i, j, df["period"])


def scored_points(name, split, data):
    pts = cached_points(name, split)
    res, curve, sc = evaluate(pts, data, split, K_KM, groups=())
    thr = best_f1_threshold(curve)
    sc = sc.copy()
    sc["fi"], sc["fj"] = sensor_ij(sc)
    return sc, thr, res


def choose(data, split, tg, st, st_thr, small):
    """(case label, period, focus (i, j) in the sensor frame, description) for every case type."""
    rng = np.random.default_rng(0)
    used = set()
    hits = st[(st["status"] == "tp") & (st["score"] >= st_thr)]
    matched = st[st["status"] == "tp"].set_index("fire_id")["score"]
    tg = tg.copy()
    tg["score"] = tg["fire_id"].map(matched)
    tg["hit"] = tg["score"] >= st_thr
    q25 = tg["fre_gj"].quantile(0.25)
    below = tg["score"].notna() & ~tg["hit"]
    cls = fp_classes(st, data, K_KM, small)
    fa = cls[(cls["status"] == "fp") & (cls["score"] >= st_thr)].copy()
    fa = point_signal(fa, data)
    out = []

    def pick(rows, key="period"):
        rows = rows[~rows[key].isin(used)].sort_values([key] + [c for c in ("fire_id", "score") if c in rows])
        if not len(rows):
            return None
        q = rows.iloc[int(rng.integers(len(rows)))]
        used.add(q[key])
        return q

    n_t = tg.groupby("period").size()
    ob = n_t.idxmax(); used.add(ob)
    cand = {
        "weak signal: found, but scored below the threshold": tg[below & ((tg["n_det_6h"] <= 5) | (tg["fre_gj"] <= q25)) & ~tg["in_pair"] & ~tg["already_burning"] & ~tg["seen_via_older"]],
        "next to an active fire: found, but scored below the threshold": tg[below & (tg["already_burning"] | tg["seen_via_older"]) & ~tg["in_pair"]],
        "close pair: two ignitions <= 2 cells apart, one not hit": tg[tg["in_pair"] & ~tg["hit"]],
    }
    for label, rows in cand.items():
        q = pick(rows)
        if q is not None:
            out.append((label, q["period"], (q["i_f"], q["j_f"]),
                        f"{q['burned_area_ha']:.0f} ha, ICNF {q['t_h']:.1f} h, score {q['score']:.2f} (thr {st_thr:.2f}), "
                        f"{int(q['n_det_6h'])} det. in 6 h"))
    fcases = {
        "re-flare: false alarm where it burned in the 72 h before the window": fa[(fa["p_prev_h"] < 72) & (fa["fp_class"] != "duplicate")],
        "unexplained false alarm (no ICNF record <= 10 km, no earlier activity)": fa[(fa["fp_class"] == "no ICNF evidence") & ~(fa["p_prev_h"] < 72) & (fa["p_hist"] == 0)],
        "time duplicate: second peak of a found fire, >= 2 h later or earlier": fa[(fa["fp_class"] == "duplicate") & (fa["dup_dt_h"] >= 2)],
    }
    for label, rows in fcases.items():
        q = pick(rows)
        if q is not None:
            prev = "no activity in 7 d" if np.isnan(q["p_prev_h"]) else f"last activity {q['p_prev_h']:.0f} h before the window"
            out.append((label, q["period"], (q["si"], q["sj"]), f"score {q['score']:.2f}, class '{q['fp_class']}', {prev}"))
    if "dt_h" in st:
        late = st[(st["status"] == "tp") & (st["score"] >= st_thr) & (st["dt_h"].abs() >= 3)].merge(
            tg[["fire_id", "i_f", "j_f", "t_h"]].rename(columns={"t_h": "t_icnf"}), on="fire_id")
        q = pick(late)
        if q is not None:
            out.append(("hit, but the model time is >= 3 h off", q["period"], (q["i_f"], q["j_f"]),
                        f"ICNF {q['t_icnf']:.1f} h, model {q['t_h']:.1f} h"))
    T = tg[tg["period"] == ob]
    out.append(("outbreak: the period with the most targets", ob, (T["i_f"].median(), T["j_f"].median()),
                f"{len(T)} targets, {int(T['hit'].sum())} hit at the threshold"))
    return out


def diagram(ax, data, period, fi, fj, labels, rk, small, pts_st):
    """Detections <= R_DIAG cells from (fi, fj): distance [km] vs hours since the period start, from 72 h before."""
    s0 = data.first_slot(period)
    a, b = np.searchsorted(data._slot, [s0 - (72 - HISTORY_H) * 4, s0 + (HISTORY_H + PERIOD_H + TAIL_H) * 4])
    ii, jj = data._i[a:b], data._j[a:b]
    near = np.hypot(ii - fi, jj - fj) <= R_DIAG
    la0, lo0 = G.ij_to_latlon(fi, fj)
    la, lo = G.ij_to_latlon(ii[near], jj[near])
    d = E.haversine_km(la0, lo0, la, lo)
    t = (data._slot[a:b][near] - s0) / 4 - HISTORY_H
    frp = data.det["frp_mw"].values[a:b][near]
    ax.axvspan(-HISTORY_H, PERIOD_H + TAIL_H, color=E.NEUTRAL, zorder=0)
    ax.axvspan(0, PERIOD_H, color=E.GRID, zorder=0)
    sc = ax.scatter(t, d, c=np.clip(frp, 10, 1000), s=9, cmap=E.SEQ_ORANGE, norm=LogNorm(10, 1000), lw=0, zorder=3)
    p0 = pd.Timestamp(period)

    def mark(df, color, marker, size, label, z=5):
        if not len(df):
            return
        si, sj = G.true_to_sensor_ij(*G.latlon_to_ij(df["lat"].values, df["lon"].values), period)
        dd = E.haversine_km(la0, lo0, *G.ij_to_latlon(si, sj))
        keep = dd <= R_DIAG * 4
        tt = (df["start_utc"] - p0).dt.total_seconds().values / 3600
        ax.scatter(tt[keep], dd[keep], marker=marker, s=size, color=color, lw=1.4, zorder=z, label=label)

    lab = labels[labels["period"] == period]
    mark(lab[lab["role"] == "target"], E.BLUE, "*", 110, "ICNF target ≥ 50 ha (start)", 6)
    mark(lab[lab["role"] == "ignore"], E.INK2, "D", 28, "ICNF 10–50 ha / unseen (start)")
    w0, w1 = p0 - pd.Timedelta(hours=72), p0 + pd.Timedelta(hours=PERIOD_H + TAIL_H)
    mark(rk[(rk["start_utc"] >= w0) & (rk["start_utc"] < w1)], "#8e5bd0", "P", 40, "ICNF rekindle record")
    mark(small[(small["start_utc"] >= w0) & (small["start_utc"] < w1)], E.MUTED, "|", 40, "ICNF fire < 10 ha")
    P = pts_st[pts_st["period"] == period]
    if len(P):
        dd = E.haversine_km(la0, lo0, *G.ij_to_latlon(P["fi"].values, P["fj"].values))
        k = dd <= R_DIAG * 4
        ax.scatter(P["t_h"].values[k], dd[k], marker="v", s=36, facecolor="none", edgecolor=E.INK, lw=1.1, zorder=7,
                   label="cnn3d_st point (above thr), its hour")
    ax.set_xlim(-72, PERIOD_H + TAIL_H); ax.set_ylim(0, R_DIAG * 4)
    ax.set_xticks([-72, -48, -24, -6, 0, 12, 18])
    ax.set_xlabel("hours since the period start (shaded: 24 h input window, darker: target period)", fontsize=7.5)
    ax.set_ylabel("distance from the focus [km]", fontsize=8)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args()
    assert args.split == "val" or args.confirm_test, "--split test needs --confirm-test (descriptive, after 9.20)"
    split = args.split
    data = FireData()
    device = get_device()
    st_name, f_name = RUNS[split]
    st, st_thr, st_res = scored_points(st_name, split, data)
    fn, f_thr, _ = scored_points(f_name, split, data)
    tg = target_table(data, split)
    small = icnf_small(data, split)
    cases = choose(data, split, tg, st, st_thr, small)
    for c in cases:
        print(f"  {c[0]:75s} {c[1]:%Y-%m-%d %H} UTC  {c[3]}")
    m_st, cfg_st, _ = load_model(st_name, device)
    m_f, cfg_f, _ = load_model(f_name, device)
    periods = [c[1] for c in cases]
    maps_st = predict_maps(m_st, data, periods, cfg_st, device)
    maps_f = predict_maps(m_f, data, periods, cfg_f, device)
    _, mainland = E.load_borders()
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    fig_dir, out_dir = E.ROOT / "figures" / "error_analysis", E.ROOT / "outputs" / "error_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    lab = data.labels.merge(data.fires[["fire_id", "start_utc", "lat", "lon", "burned_area_ha"]], on="fire_id")
    rk = data.rekindles
    st_above = st[st["score"] >= st_thr]
    f_above = fn[fn["score"] >= f_thr]
    title = ('Validation 2015–2016 (dev runs, trained without these years)' if split == 'val'
             else 'Test 2024–2025 (final runs; descriptive, after the single test)')

    def draw_row(fig, axrow, case, first):
        label, period, (ci, cj), desc = case
        x = data.sample(period)[0]
        det = x[:, 2] > 0
        first_h = np.where(det.any(0), det.argmax(0) / 4, np.nan)
        p3, _ = maps_st[period]
        i0, i1, j0, j1 = ci - HALF, ci + HALF, cj - HALF, cj + HALF
        T = lab[(lab["period"] == period) & (lab["role"] == "target")]
        Ig = lab[(lab["period"] == period) & (lab["role"] == "ignore")]
        panels = [(first_h, E.SEQ_BLUE.reversed(), (0, 24), "(a) input: hour of first detection in the 24 h window"),
                  None,
                  (maps_f[period], E.SEQ_ORANGE, (0, VMAX), f"(c) cnn3d_final: p, points p ≥ {f_thr:.2f}"),
                  (p3.max(0), E.SEQ_ORANGE, (0, VMAX), f"(d) cnn3d_st: max over hours, points p ≥ {st_thr:.2f} (hour)")]
        for c, pnl in enumerate(panels):
            ax = axrow[c]
            if pnl is None:
                sc = diagram(ax, data, period, ci, cj, lab, rk, small, st_above.assign(t_h=st_above.get("t_h", np.nan)))
                if first:
                    ax.set_title("(b) detections ≤ 7 cells from the focus: distance vs time (colour = FRP)", fontsize=9)
                    fig.colorbar(sc, ax=ax, shrink=0.7, label="FRP [MW]")
                    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
                continue
            img, cmap, (v0, v1), title = pnl
            im = ax.imshow(img, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            ax.plot(bj, bi, color=E.MUTED, lw=0.8)
            if c >= 2:
                ax.scatter(T["j_f"], T["i_f"], marker="o", s=130, facecolor="none", edgecolor=E.INK, lw=1.0, zorder=5, label="ICNF target ≥ 50 ha")
                ax.scatter(Ig["j_f"], Ig["i_f"], marker="D", s=30, facecolor="none", edgecolor=E.INK2, lw=0.8, zorder=5, label="ICNF ignore (10–50 ha / unseen)")
                P = (f_above if c == 2 else st_above)
                P = P[P["period"] == period]
                for s_, (col, mk, lb) in STATUS.items():
                    Q = P[P["status"] == s_]
                    ax.scatter(Q["fj"], Q["fi"], marker=mk, s=60, lw=1.8, color=col, zorder=6, label=f"point: {lb}")
                if c == 3 and "t_h" in P:
                    vis = P[P["fi"].between(i0, i1) & P["fj"].between(j0, j1)]
                    for q in vis.itertuples():
                        ax.annotate(f"{q.t_h:.1f}h", (q.fj, q.fi), xytext=(4, -9), textcoords="offset points", fontsize=6.5,
                                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75), zorder=8)
            ax.plot(cj, ci, marker="s", ms=17, mfc="none", mec=E.ORANGE, mew=1.3, zorder=7)
            ax.set_xlim(j0, j1); ax.set_ylim(i1, i0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if first:
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.7, label=("hour in the window (6–18 = period)", "", "p", "p (max over hours)")[c])
                if c == 2:
                    ax.legend(fontsize=6.3, loc="lower left", framealpha=0.9)
        # (e) hourly profile at the focus and at the targets nearby
        ax = axrow[4]
        tc = np.arange(p3.shape[0]) + 0.5
        a, b = int(round(ci)), int(round(cj))
        ax.plot(tc, p3[:, max(a - 1, 0):a + 2, max(b - 1, 0):b + 2].max((1, 2)), color=E.ORANGE, lw=2, marker=".", label="focus (□)")
        near = T[np.hypot(T["i_f"] - ci, T["j_f"] - cj) <= 3]
        for k, q in enumerate(near.itertuples()):
            col = (E.BLUE, E.AQUA, "#8e5bd0", E.INK2)[k % 4]
            ii, jj = int(round(q.i_f)), int(round(q.j_f))
            if (ii, jj) != (a, b):
                ax.plot(tc, p3[:, max(ii - 1, 0):ii + 2, max(jj - 1, 0):jj + 2].max((1, 2)), color=col, lw=1.2, ls="--")
            ax.axvline((q.start_utc - pd.Timestamp(period)).total_seconds() / 3600, color=col, ls=":", lw=1.4,
                       label=f"ICNF start, {q.burned_area_ha:.0f} ha")
        ax.axhline(st_thr, color=E.MUTED, lw=0.9, ls="-.", label="threshold")
        ax.set_xlim(0, PERIOD_H); ax.set_ylim(0, max(0.8, ax.get_ylim()[1])); ax.set_xticks(range(0, 13, 2))
        ax.set_xlabel(f"hours since {pd.Timestamp(period):%H:%M} UTC", fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
        if first:
            ax.set_title("(e) cnn3d_st p by hour (max of 3 × 3 cells)", fontsize=9)
        axrow[0].set_ylabel(f"{label}\n{pd.Timestamp(period):%Y-%m-%d %H:%M} UTC + 12 h\n{desc}", fontsize=8.3)
        return dict(case=label, period=period, focus_i=ci, focus_j=cj, detail=desc, n_targets=len(T))

    rows = []
    for n, part in enumerate((cases[:4], cases[4:]), 1):
        if not part:
            continue
        fig, axes = plt.subplots(len(part), 5, figsize=(24, 4.3 * len(part)), layout="constrained",
                                 gridspec_kw=dict(width_ratios=[1, 1.45, 1, 1, 1.1]))
        axes = np.atleast_2d(axes)
        for r, case in enumerate(part):
            rows.append(draw_row(fig, axes[r], case, r == 0))
        fig.suptitle(f"{title}: where cnn3d_st struggles; one cell ≈ 3.2 × 4.7 km; orange square = focus", fontsize=11, x=0.01, ha="left")
        out = fig_dir / f"{split}_failure_examples_{n}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"figure: {out.relative_to(E.ROOT)}")
    case_dir = fig_dir / "cases"                          # one figure per case, for slides
    case_dir.mkdir(exist_ok=True)
    for n, case in enumerate(cases, 1):
        fig, axes = plt.subplots(1, 5, figsize=(24, 5.0), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1.45, 1, 1, 1.1]))
        draw_row(fig, axes, case, True)
        fig.suptitle(f"{title}: {case[0]}", fontsize=11, x=0.01, ha="left")
        slug = next(v for k, v in SLUGS.items() if case[0].startswith(k))
        fig.savefig(case_dir / f"{split}_{n}_{slug}.png", dpi=130); plt.close(fig)
    print(f"per-case figures: {case_dir.relative_to(E.ROOT)}/{split}_*.png")
    pd.DataFrame(rows).to_csv(out_dir / f"{split}_failure_examples.csv", index=False)


if __name__ == "__main__":
    main()
