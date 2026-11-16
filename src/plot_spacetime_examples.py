"""Example periods of the space-time model cnn3d_st next to cnn3d_final (notes.md 9.12).

Validation 2015-2016 only (both models are trained on 2004-2014; the test split stays locked).
Descriptive: nothing here is used for a model choice. Operating point of each model = its best F1 at
10 km on validation (cnn3d_st: decoding D1, 3D peaks). Periods are chosen by fixed rules, random
choice with seed 0 inside each case type, no hand-picking:
  close pair found      a target <= 2 cells from another of its period, hit by cnn3d_st, missed by cnn3d_final
                        (periods with <= 6 targets preferred, the outbreak excluded; every period used once)
  already burning       a target with a detection <= 1.5 cells in the 6 h before its start, hit by cnn3d_st
  outbreak              the validation period with the most targets
  fresh, isolated       a period with one fresh, isolated target hit by both models
Cases with a focus target (pair, already burning, fresh, missed) are zoomed on it.
  missed                a target missed by cnn3d_st
  false alarm           a period without targets with a cnn3d_st point above its threshold
  random 1, 2           periods with >= 1 target, no condition on the outcome
Columns: (a) input only, hour of the first SEVIRI detection in the 24 h window; (b) cnn3d_final
probability and points; (c) cnn3d_st, max over the 12 hourly maps, its points with the predicted
hour; (d) per labelled target: cnn3d_st probability at the target over the 12 hours of the period,
ICNF start, model time and first-detection rule time (lag removed) of the matched point.
Positions in the sensor frame, as the maps.

Run from the repository root:
  .venv/bin/python src/plot_spacetime_examples.py
Outputs: figures/spacetime/val_examples_{1,2}.png, outputs/spacetime/val_examples.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from baselines import first_detection_times, training_lag_h
from dataset import FireData
from evaluate import evaluate, target_context
from tiling_design import HISTORY_H, PERIOD_H
from train import get_device, load_model, predict, predict_maps, sample_kwargs

K_KM = 10
PROB_VMAX = 0.6
SPLIT = "val"
MAX_LABELLED = 5
TARGET_COLORS = [E.BLUE, E.ORANGE, E.AQUA, "#8a5cd1", "#b58a12"]
STATUS_COLOR = {"tp": E.BLUE, "fp": E.AQUA, "ignored": E.INK2}
STATUS_STYLE = {"tp": dict(marker="x", label="hit (≤ 10 km)"),
                "fp": dict(marker="X", label="false alarm"),
                "ignored": dict(marker="+", label="near an ignore fire")}


def sensor_ij(pts):
    i, j = G.latlon_to_ij(pts["lat"].values, pts["lon"].values)
    return G.true_to_sensor_ij(i, j, pts["period"])


def score(preds, data):
    """Matched points (with sensor-frame fi, fj) and the best-F1 threshold at K_KM."""
    res, curve, scored = evaluate(preds, data, SPLIT, K_KM, groups=())
    f1 = 2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]).clip(lower=1e-9)
    thr = float(curve["score"].iloc[int(f1.values.argmax())])
    scored = scored.copy()
    scored["fi"], scored["fj"] = sensor_ij(scored)
    return res, thr, scored


def hits(scored, thr):
    s = scored[(scored["score"] >= thr) & (scored["status"] == "tp")]
    return set(s["fire_id"])


def choose_cases(data, tg, st_sc, st_thr):
    """(label, period, focus fire_ids); every period used once. Cases with a focus are zoomed on it."""
    rng = np.random.default_rng(0)
    per = data.periods[data.periods["split"] == SPLIT].set_index("period")
    n_t = tg.groupby("period").size()
    fa = st_sc[(st_sc["score"] >= st_thr) & (st_sc["status"] == "fp")].groupby("period").size()
    outbreak = n_t.idxmax()
    used = {outbreak}

    def pick(cands, prefer_small=True):
        """Random target of the candidate rows (seed 0), period not used yet; periods with <= 6 targets first."""
        c = cands[~cands["period"].isin(used)]
        if prefer_small and (c["period"].map(n_t) <= 6).any():
            c = c[c["period"].map(n_t) <= 6]
        c = c.sort_values(["period", "fire_id"])
        q = c.iloc[int(rng.integers(len(c)))]
        used.add(q["period"])
        return q

    def partners(q):
        g = tg[tg["period"] == q["period"]]
        d = np.hypot(g["i_f"] - q["i_f"], g["j_f"] - q["j_f"])
        return list(g.loc[d <= 2.0, "fire_id"])

    single = set(n_t.index[n_t == 1])
    q_pair = pick(tg[tg["in_pair"] & tg["hit_st"] & ~tg["hit_final"]])
    q_burn = pick(tg[tg["already_burning"] & tg["hit_st"]])
    q_fresh = pick(tg[tg["period"].isin(single) & ~tg["already_burning"] & tg["isolated"] & tg["hit_st"] & tg["hit_final"]])
    q_miss = pick(tg[~tg["hit_st"]])
    fa_p = sorted((set(per.index[per["n_targets"] == 0]) & set(fa.index)) - used)
    p_fa = fa_p[int(rng.integers(len(fa_p)))]; used.add(p_fa)
    rand = [p for p in sorted(n_t.index) if p not in used]
    r1, r2 = (rand[k] for k in rng.choice(len(rand), 2, replace=False))
    return [("close pair: found by cnn3d_st, missed by cnn3d_final", q_pair["period"], partners(q_pair)),
            ("already burning: new fire next to an active one, found by cnn3d_st", q_burn["period"], [q_burn["fire_id"]]),
            ("outbreak (most targets)", outbreak, []),
            ("fresh, isolated fire, found by both", q_fresh["period"], [q_fresh["fire_id"]]),
            ("missed by cnn3d_st", q_miss["period"], partners(q_miss)),
            ("false alarm of cnn3d_st (no target)", p_fa, []),
            ("random period with a target", r1, []),
            ("random period with a target", r2, [])]


def window(ii, jj, margin=5, min_size=22):
    ci, cj = (ii.min() + ii.max()) / 2, (jj.min() + jj.max()) / 2
    h = max((ii.max() - ii.min()) / 2 + margin, (jj.max() - jj.min()) / 2 + margin, min_size / 2)
    return ci - h, ci + h, cj - h, cj + h


def plot_rows(cases, data, tg, maps_f, maps_st, fin, st, cfg_f, bi, bj, out):
    (f_sc, f_thr), (s_sc, s_thr) = fin, st
    fig, axes = plt.subplots(len(cases), 4, figsize=(19, 4.1 * len(cases)), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    axes = np.atleast_2d(axes)
    rows = []
    for r, (label, period, focus) in enumerate(cases):
        x = data.sample(period, **sample_kwargs(cfg_f))[0]
        det = x[:, 2] > 0
        first_h = np.where(det.any(0), det.argmax(0) / 4, np.nan)
        p3, _ = maps_st[period]
        T = tg[tg["period"] == period].sort_values("t_h")
        Pf = f_sc[(f_sc["period"] == period) & (f_sc["score"] >= f_thr)]
        Ps = s_sc[(s_sc["period"] == period) & (s_sc["score"] >= s_thr)]
        F = T[T["fire_id"].isin(focus)]
        if len(F):                                        # zoom on the case's fire(s)
            i0, i1, j0, j1 = window(F["i_f"].values, F["j_f"].values, margin=8)
        elif len(T) + len(Pf) + len(Ps):
            i0, i1, j0, j1 = window(np.r_[T["i_f"], Pf["fi"], Ps["fi"]], np.r_[T["j_f"], Pf["fj"], Ps["fj"]])
        else:
            i0, i1, j0, j1 = window(*np.nonzero(data.static["pt"]))
        # labelled targets: the focus first, then in-pair / already-burning ones, then by start time
        inside = T[T["i_f"].between(i0, i1) & T["j_f"].between(j0, j1)].assign(focus=lambda d: d["fire_id"].isin(focus))
        lab = inside.sort_values(["focus", "in_pair", "already_burning", "t_h"], ascending=[False, False, False, True]).head(MAX_LABELLED)
        lab = lab.sort_values("t_h")
        letters = {fid: chr(65 + n) for n, fid in enumerate(lab["fire_id"])}

        panels = [(first_h, E.SEQ_BLUE.reversed(), (0, 24), "(a) input: hour of first SEVIRI detection in the 24 h window"),
                  (maps_f[period], E.SEQ_ORANGE, (0, PROB_VMAX), f"(b) cnn3d_final: probability, points p ≥ {f_thr:.2f}"),
                  (p3.max(0), E.SEQ_ORANGE, (0, PROB_VMAX), f"(c) cnn3d_st: max over the 12 hours, points p ≥ {s_thr:.2f}\n(label = predicted hour since the period start)")]
        for c, (img, cmap, (v0, v1), title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img, cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
            ax.plot(bj, bi, color=E.MUTED, lw=0.8)
            if c >= 1:
                ax.scatter(T["j_f"], T["i_f"], marker="o", s=120, facecolor="none", edgecolor=E.INK, lw=0.9, zorder=5,
                           label="ICNF target (≥ 50 ha)")
                for q in lab.itertuples():
                    ax.annotate(letters[q.fire_id], (q.j_f, q.i_f), xytext=(5, 5), textcoords="offset points",
                                fontsize=9, fontweight="bold", color=TARGET_COLORS[ord(letters[q.fire_id]) - 65], zorder=7)
            if c in (1, 2):
                P_all = Pf if c == 1 else Ps
                for s_, sty in STATUS_STYLE.items():
                    P = P_all[P_all["status"] == s_]
                    ax.scatter(P["fj"], P["fi"], marker=sty["marker"], s=55, lw=1.8, zorder=6, color=STATUS_COLOR[s_],
                               label=f"predicted, {sty['label']}")
                if c == 2:                                # predicted hour of each point
                    vis = P_all[P_all["fi"].between(i0, i1) & P_all["fj"].between(j0, j1)]
                    for q in vis.itertuples():
                        ax.annotate(f"{q.t_h:.1f}h", (q.fj, q.fi), xytext=(4, -9), textcoords="offset points", fontsize=6.5,
                                    color=E.INK, zorder=8, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
            ax.set_xlim(j0, j1); ax.set_ylim(i1, i0)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, shrink=0.7, label=("hour in the 24 h window (6–18 = period)", "p", "p (max over hours)")[c])

        # (d) time profiles of the labelled targets
        ax = axes[r, 3]
        tc = (np.arange(p3.shape[0]) + 0.5) * PERIOD_H / p3.shape[0]
        for q in lab.itertuples():
            col = TARGET_COLORS[ord(letters[q.fire_id]) - 65]
            ii, jj = int(round(q.i_f)), int(round(q.j_f))
            prof = p3[:, max(ii - 1, 0):ii + 2, max(jj - 1, 0):jj + 2].max((1, 2))
            ax.plot(tc, prof, color=col, lw=1.6, marker=".", ms=4)
            ax.axvline(q.t_h, color=col, ls="--", lw=1.1)
            m = Ps[(Ps["status"] == "tp") & (Ps["fire_id"] == q.fire_id)]
            note = ""
            if len(m):
                mt, rt = m["t_h"].iat[0], m["t_rule"].iat[0]
                ax.plot(mt, -0.03, marker="o", color=col, ms=7, clip_on=False, zorder=6)
                if np.isfinite(rt):
                    ax.plot(rt, -0.08, marker="^", color=col, ms=7, clip_on=False, zorder=6)
                note = f"model {mt - q.t_h:+.1f} h, rule {rt - q.t_h:+.1f} h" if np.isfinite(rt) else f"model {mt - q.t_h:+.1f} h, rule –"
            else:
                note = "not found by cnn3d_st"
            tags = ", ".join(t for t, on in (("pair", q.in_pair), ("already burning", q.already_burning)) if on)
            ax.plot([], [], color=col, lw=1.6, label=f"{letters[q.fire_id]}: ICNF {q.t_h:.1f} h{' (' + tags + ')' if tags else ''}; {note}")
            rows.append(dict(case=label, period=period, target=letters[q.fire_id], fire_id=q.fire_id, icnf_t_h=q.t_h,
                             in_pair=q.in_pair, already_burning=q.already_burning, hit_st=q.hit_st, hit_final=q.hit_final,
                             model_t_h=m["t_h"].iat[0] if len(m) else np.nan, rule_t_h=m["t_rule"].iat[0] if len(m) else np.nan))
        ax.set_xlim(0, PERIOD_H); ax.set_ylim(-0.12, max(1.0, ax.get_ylim()[1]))
        ax.axhline(s_thr, color=E.MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(0, PERIOD_H + 1, 2))
        ax.set_xlabel(f"hours since {period:%H:%M} UTC", fontsize=8)
        ax.set_ylabel("cnn3d_st p at the target (max of 3 × 3 cells)", fontsize=8)
        if len(lab):
            ax.legend(loc="upper left", fontsize=6.8, framealpha=0.9, handlelength=1.2)
        else:
            ax.text(0.5, 0.5, "no target in this period", transform=ax.transAxes, ha="center", color=E.MUTED)
        if r == 0:
            ax.set_title("(d) per target: line = p by hour, dashed = ICNF start,\n● model time, ▲ rule time (lag removed), dotted = threshold",
                         fontsize=9)
        n_t = len(T)
        hs, hf = int(T["hit_st"].sum()), int(T["hit_final"].sum())
        axes[r, 0].set_ylabel(f"{label}\n{period:%Y-%m-%d %H:%M} UTC + 12 h\n{n_t} targets; hit: cnn3d_st {hs}, cnn3d_final {hf}\n"
                              f"false alarms: cnn3d_st {int((Ps['status'] == 'fp').sum())}, cnn3d_final {int((Pf['status'] == 'fp').sum())}",
                              fontsize=8.5)
    axes[0, 1].legend(loc="lower left", fontsize=7, framealpha=0.9)
    axes[0, 2].legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.suptitle("Validation 2015–2016: cnn3d_final vs. cnn3d_st (space-time); one cell ≈ 3.2 × 4.5 km, sensor frame",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"figure: {out.relative_to(E.ROOT)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", default="ablations_st/cnn3d_st_s0")
    ap.add_argument("--final", default="ablations_georef/cnn3d_final_s0")
    args = ap.parse_args()
    device = get_device()
    data = FireData()
    assert data.georef_fix, "expects the geo-referencing-corrected build (6.12)"
    m_f, cfg_f, _ = load_model(args.final, device)
    m_s, cfg_s, _ = load_model(args.st, device)
    periods = data.periods.loc[(data.periods["split"] == SPLIT) & (data.periods["n_det"] > 0), "period"]

    pts_f = predict(m_f, data, periods, cfg_f, device)
    pts_s = predict(m_s, data, periods, cfg_s, device)          # D1
    lag = training_lag_h(data, "train")
    pts_s["t_rule"] = np.clip(first_detection_times(pts_s.drop(columns="t_h"), data)["t_h"] - lag, 0, PERIOD_H)
    res_f, f_thr, f_sc = score(pts_f, data)
    res_s, s_thr, s_sc = score(pts_s, data)
    s_sc["t_rule"] = pts_s.loc[s_sc.index, "t_rule"]
    print(f"cnn3d_final AP@10 {res_f['AP']:.3f}, threshold {f_thr:.3f}; cnn3d_st AP@10 {res_s['AP']:.3f}, threshold {s_thr:.3f}")

    tg = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(set(periods))][["period", "fire_id", "i_f", "j_f"]]
    tg = tg.join(target_context(data)[["t_h", "in_pair", "already_burning"]], on="fire_id")
    tg = tg.join(data.fires.set_index("fire_id")["isolated"], on="fire_id")
    tg["hit_st"] = tg["fire_id"].isin(hits(s_sc, s_thr))
    tg["hit_final"] = tg["fire_id"].isin(hits(f_sc, f_thr))
    cases = choose_cases(data, tg, s_sc, s_thr)
    for label, p, _ in cases:
        print(f"  {label:66s} {p:%Y-%m-%d %H:%M}")

    chosen = [p for _, p, _ in cases]
    maps_f = predict_maps(m_f, data, chosen, cfg_f, device)
    maps_s = predict_maps(m_s, data, chosen, cfg_s, device)
    _, mainland = E.load_borders()
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    fig_dir, out_dir = E.ROOT / "figures" / "spacetime", E.ROOT / "outputs" / "spacetime"
    fig_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, part in enumerate((cases[:4], cases[4:]), 1):
        rows += plot_rows(part, data, tg, maps_f, maps_s, (f_sc, f_thr), (s_sc, s_thr), cfg_f, bi, bj,
                          fig_dir / f"val_examples_{n}.png")
    pd.DataFrame(rows).assign(threshold_st=s_thr, threshold_final=f_thr).to_csv(out_dir / "val_examples.csv", index=False)
    print("written: outputs/spacetime/val_examples.csv")


if __name__ == "__main__":
    main()
