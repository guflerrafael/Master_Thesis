"""Dense samples for the heatmap model, built on demand from data/processed/model/ (notes.md 5.1).

A sample is one 12 h target period:
  x  (T, C, h, w) float32   T = 96 steps of 15 min (6 h history, 12 h period, 6 h tail);
                            channels: log(1 + FRP) / log(1 + 1100), confidence / 100, detected
  y  (h, w)       float32   max of Gaussians (sigma = SIGMA px) at every target ignition
  w  (h, w)       float32   loss weight: 1 inside the loss mask (mainland Portugal + 1 px),
                            0 outside and within IGNORE_R px of every 'ignore' fire, except
                            within TARGET_CORE px of a target (targets win over ignore regions)
for the full grid (177 x 106) or a crop (i0, j0, size). Framework-free (numpy).

Run from the repository root (checks + example figure):
  .venv/bin/python src/dataset.py > outputs/dataset_check_log.txt
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from tiling_design import HISTORY_H, PERIOD_H, TAIL_H

DATA = E.PROC / "model"
FIG = E.ROOT / "figures" / "dataset"
T = (HISTORY_H + PERIOD_H + TAIL_H) * 4
CHANNELS = ("frp", "confidence", "detected")
FRP_NORM = np.log1p(1100.0)
SIGMA, IGNORE_R, TARGET_CORE = 1.0, 2.0, 1.0


class FireData:
    def __init__(self, root=DATA):
        self.det = pd.read_parquet(root / "detections.parquet").sort_values("slot").reset_index(drop=True)
        self.fires = pd.read_parquet(root / "fires.parquet")
        self.labels = pd.read_parquet(root / "labels.parquet")
        self.periods = pd.read_parquet(root / "periods.parquet")
        self.static = dict(np.load(root / "static.npz"))
        self._slot = self.det["slot"].values
        self._i, self._j = self.det["i"].values.astype(int), self.det["j"].values.astype(int)
        self._frp = (np.log1p(self.det["frp_mw"].values) / FRP_NORM).astype(np.float32)
        self._conf = (self.det["confidence"].values / 100).astype(np.float32)
        self._lab = {p: g for p, g in self.labels.groupby("period")}

    @staticmethod
    def first_slot(period):
        return (pd.Timestamp(period) - pd.Timedelta(hours=HISTORY_H)).value // (15 * 60 * 10**9)

    def sample(self, period, crop=None):
        i0, j0, h, w = (0, 0, G.H, G.W) if crop is None else (crop[0], crop[1], crop[2], crop[2])
        s0 = self.first_slot(period)
        a, b = np.searchsorted(self._slot, [s0, s0 + T])
        ii, jj = self._i[a:b] - i0, self._j[a:b] - j0
        keep = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
        t = self._slot[a:b][keep] - s0
        ii, jj = ii[keep], jj[keep]
        x = np.zeros((T, len(CHANNELS), h, w), np.float32)
        x[t, 0, ii, jj] = self._frp[a:b][keep]
        x[t, 1, ii, jj] = self._conf[a:b][keep]
        x[t, 2, ii, jj] = 1.0

        y = np.zeros((h, w), np.float32)
        wt = self.static["pt_loss"][i0:i0 + h, j0:j0 + w].astype(np.float32)
        lab = self._lab.get(pd.Timestamp(period), self.labels.iloc[:0]).copy()
        lab["i_loc"], lab["j_loc"] = lab["i_f"] - i0, lab["j_f"] - j0
        r, c = np.mgrid[0:h, 0:w]
        core = np.zeros((h, w), bool)
        for q in lab.itertuples():
            d2 = (r - q.i_loc) ** 2 + (c - q.j_loc) ** 2
            if q.role == "target":
                y = np.maximum(y, np.exp(-d2 / (2 * SIGMA ** 2)).astype(np.float32))
                core |= d2 <= TARGET_CORE ** 2
            else:
                wt[d2 <= IGNORE_R ** 2] = 0.0
        wt[core] = self.static["pt_loss"][i0:i0 + h, j0:j0 + w][core]
        inside = lab["i_loc"].between(-0.5, h - 0.5) & lab["j_loc"].between(-0.5, w - 0.5)
        return x, y, wt, lab[inside]


def crop_origin(i, j, size, rng=None, jitter=0):
    """Origin of a size x size crop containing image position (i, j), optionally shifted at
    random by up to `jitter` pixels, clipped to the grid."""
    di, dj = (rng.integers(-jitter, jitter + 1, 2) if rng is not None and jitter else (0, 0))
    i0 = int(np.clip(round(i) - size // 2 + di, 0, G.H - size))
    j0 = int(np.clip(round(j) - size // 2 + dj, 0, G.W - size))
    return i0, j0, size


# ---------------------------------------------------------------- checks and example

def checks(data, n=300, seed=0):
    rng = np.random.default_rng(seed)
    per = data.periods
    pick = per.sample(n, random_state=seed)
    bad_count, bad_peak = 0, 0
    for p in pick.itertuples():
        x, y, wt, lab = data.sample(p.period)
        bad_count += int(x[:, 2].sum()) != p.n_det          # every detection of the window is in x
        tg = lab[lab["role"] == "target"]
        for q in tg.itertuples():                            # heatmap peaks at the targets
            bad_peak += y[int(round(q.i_loc)), int(round(q.j_loc))] < np.exp(-0.5)
    print(f"Checks on {n} random periods: detection count mismatches {bad_count}, "
          f"targets without a heatmap peak at their pixel {bad_peak}")

    # targets whose own pixel has weight 0 (inside an ignore disc of another fire)
    pos = per[per["n_targets"] > 0]
    conflicts, total = 0, 0
    for p in pos.itertuples():
        _, _, wt, lab = data.sample(p.period)
        tg = lab[lab["role"] == "target"]
        total += len(tg)
        conflicts += int((wt[np.round(tg["i_loc"]).astype(int), np.round(tg["j_loc"]).astype(int)] == 0).sum())
    print(f"Targets inside an ignore region of another fire (weight 0 at their pixel): {conflicts} of {total}")

    # crops: every target can be placed in a 32 x 32 crop
    tg = data.labels[data.labels["role"] == "target"].sample(200, random_state=seed)
    ok = 0
    for q in tg.itertuples():
        crop = crop_origin(q.i_f, q.j_f, 32, rng, jitter=8)
        _, y, _, lab = data.sample(q.period, crop)
        ok += (lab["fire_id"] == q.fire_id).any() and y.max() > 0.5
    print(f"Random 32 × 32 crops (jitter ±8 px) around 200 targets contain their target: {ok} of 200")
    x, y, wt, _ = data.sample(pos["period"].iloc[0])
    print(f"Sample shapes: x {x.shape} ({x.nbytes / 1e6:.1f} MB), y {y.shape}, w {wt.shape}; "
          f"weighted cells per period: {wt.sum():,.0f}")


def fig_example(data, period="2017-10-15 06:00"):
    FIG.mkdir(parents=True, exist_ok=True)
    period = pd.Timestamp(period, tz="UTC")
    x, y, wt, lab = data.sample(period)
    _, mainland = E.load_borders()
    bi, bj = G.latlon_to_ij(np.array(mainland.exterior.xy[1]), np.array(mainland.exterior.xy[0]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 7), layout="constrained")
    parts = [("input: max FRP, 24 h", np.log1p(np.expm1(x[:, 0] * FRP_NORM).max(0)), E.SEQ_BLUE, (0, np.log1p(1000))),
             ("input: first detection [h since window start]", None, E.SEQ_BLUE, (0, T / 4)),
             ("target heatmap and loss weight", y, E.SEQ_ORANGE, (0, 1))]
    for ax, (title, img, cmap, (v0, v1)) in zip(axes, parts):
        if img is None:
            det = x[:, 2] > 0
            first = np.where(det.any(0), det.argmax(0) / 4, np.nan)
            img = first
        im = ax.imshow(img if title.startswith("target") else np.where(np.isnan(img) | (img == 0), np.nan, img),
                       cmap=cmap, vmin=v0, vmax=v1, interpolation="nearest")
        if title.startswith("target"):
            ax.imshow(np.where(wt == 0, 1, np.nan), cmap="Greys", vmin=0, vmax=3, alpha=0.6, interpolation="nearest")
            tg = lab[lab["role"] == "target"]
            ax.scatter(tg["j_loc"], tg["i_loc"], marker="+", s=25, color=E.INK, lw=0.8)
        ax.plot(bj, bi, color=E.MUTED, lw=0.7)
        ax.set_xlim(-0.5, G.W - 0.5); ax.set_ylim(G.H - 0.5, -0.5)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.6)
    n_t = (lab["role"] == "target").sum()
    fig.suptitle(f"Full-domain sample, target period {period:%Y-%m-%d %H:%M} UTC + 12 h: {n_t} target ignitions (+), "
                 f"{(lab['role'] == 'ignore').sum()} ignore regions; grey = no loss (outside Portugal or ignore)",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(FIG / "example_full_domain.png"); plt.close(fig)
    print("figure: figures/dataset/example_full_domain.png")


if __name__ == "__main__":
    data = FireData()
    checks(data)
    fig_example(data)
