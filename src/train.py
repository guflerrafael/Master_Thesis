"""Train an ignition-heatmap model and validate it like the baselines (notes.md 6, 6.7).

Training: square crops (default 32 x 32). Positives = one crop per target ignition of the
training years (random jitter +-8 px), negatives = hard-negative crops around a random detection
in periods without a target (default as many as positives, drawn anew every epoch); random
horizontal/vertical flips. Validation: full-domain prediction for every validation period with
detections, peaks -> points, scored with src/evaluate.py (k = 5 and 10 km). The best epoch (mean
of AP@5 and AP@10) is kept. The test split is never touched here.

All settings are command-line options; DEFAULTS reproduce v1 (hourly). Old checkpoints (v1, v2)
store only some of the keys; config_from() fills in the defaults, so they load unchanged.

Time resolution: --step-agg sets how many 15-min slots form one model step
(4 = hourly, 24 steps, v1; 1 = native 15 min, 96 steps, v2). Each step is the maximum over
its slots. 15-min steps make training ~4x slower and use ~4x more activation memory.

Space-time model (notes.md 9.4-9.6): --arch cnn3d_st --time-bins 12 [--sigma-t 1 --aux2d-weight 1] predicts
one ignition map per hour of the period plus an auxiliary 2D map; loss = focal loss on the space-time
targets + aux2d_weight x focal loss on the 2D targets. Validation scores three decodings of the same
maps (D1 '3d', D2 'max', D0 '2d'); the checkpoint is chosen on D1, as for the 2D models. Validation
points are made batch by batch (no full set of maps in memory).

Final models (notes.md 6.16): --train-splits train,val --no-val trains on 2004-2016 without
validation and saves the last epoch; the test split can never be used for training. The splits are
defined in tiling_design.SPLIT (train 2004-2014, val 2015-2016, train2 2017-2019, train3 2020-2023,
test 2024-2025; notes.md 8.8).

Run from the repository root:
  .venv/bin/python src/train.py --name convlstm_v1 > outputs/convlstm_v1_log.txt
  .venv/bin/python src/train.py --name convlstm_v2_15min --step-agg 1 > outputs/convlstm_v2_15min_log.txt
  .venv/bin/python src/train.py --name smoke --epochs 1 --max-batches 5 --val-periods 20   (quick check)
Ablation batches: src/run_ablations.py.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import eda_figures as E
import grid as G
from dataset import FireData, crop_origin
from evaluate import evaluate
from model import LOSSES, build_model
from tiling_design import HISTORY_H, PERIOD_H, TAIL_H

MODELS = E.ROOT / "models"
OUTPUTS = E.ROOT / "outputs"

ALL_CHANNELS = ("frp", "confidence", "detected", "births", "period", "time", "pt", "persistent")
DEFAULT_CHANNELS = ("frp", "confidence", "detected", "period", "time", "pt", "persistent")
DEFAULTS = dict(step_agg=4, channels=list(DEFAULT_CHANNELS), history_h=HISTORY_H, tail_h=TAIL_H,
                arch="convlstm", layers=2, hidden=32, kernel=3, readout="last",
                crop=32, sigma=1.0, ignore_r=2.0, min_target_ha=None, loss="focal", neg_ratio=1.0, flip=True,
                rekindle_ignore=False, time_bins=0, sigma_t=1.0, aux2d_weight=1.0)


def get_device():
    """NVIDIA GPU (e.g. Google Colab) > Apple GPU (MPS) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def config_from(saved):
    """Full configuration from a (possibly older) saved config: missing keys get their defaults."""
    cfg = {**DEFAULTS, **{k: v for k, v in saved.items() if v is not None or k == "min_target_ha"}}
    cfg["channels"] = [c for c in ALL_CHANNELS if c in cfg["channels"]]   # canonical order
    return cfg


def pos_threshold(sigma, sigma_t=None, bin_h=1.0):
    """Positive pixels of the loss: y >= this value. 0.75 for sigma >= 1 px (v1/v2); lower for
    narrower targets so that the pixel nearest each sub-pixel ignition always counts.
    Space-time targets (sigma_t in hours, bins of bin_h hours): the voxel nearest an ignition is at
    most half a cell and half a bin away, so the threshold also drops by exp(-bin_h^2 / (8 sigma_t^2)),
    e.g. 0.667 for sigma = 1 px, sigma_t = 1 h, 1 h bins (notes.md 9.4, item 1)."""
    e = -0.5 / (2 * sigma ** 2)
    if sigma_t is not None:
        e -= bin_h ** 2 / (8 * sigma_t ** 2)
    return min(0.75, 0.97 * float(np.exp(e)))


# ---------------------------------------------------------------- features

def features(x, static, cfg):
    """(96, 3 or 4, h, w) sample input -> (T', C', h, w) model input.
    Window: the history / tail are cut to history_h / tail_h hours. Steps: maximum over each group
    of step_agg 15-min slots (1 = native slots). Channels (cfg['channels'], canonical order of
    ALL_CHANNELS): log FRP, confidence, detected, nucleus births (only if the sample has them),
    'inside target period' flag, normalised time, mainland mask, persistent-pixel flag."""
    step = cfg["step_agg"]
    lo, hi = (HISTORY_H - cfg["history_h"]) * 4, (HISTORY_H + PERIOD_H + cfg["tail_h"]) * 4
    x = x[lo:hi]
    T, C, h, w = x.shape
    hist = cfg["history_h"] * 4
    assert T % step == 0 and hist % step == 0 and (PERIOD_H * 4) % step == 0, \
        "step_agg must divide the window and the period boundaries"
    t_ = T // step
    xa = x.reshape(t_, step, C, h, w).max(1)
    t = np.arange(t_)
    a, b = hist // step, (hist + PERIOD_H * 4) // step
    period = np.broadcast_to(((t >= a) & (t < b)).astype(np.float32)[:, None, None], (t_, h, w))
    tnorm = np.broadcast_to((t / max(t_ - 1, 1)).astype(np.float32)[:, None, None], (t_, h, w))
    maps = {"frp": xa[:, 0], "confidence": xa[:, 1], "detected": xa[:, 2],
            "period": period, "time": tnorm,
            "pt": np.broadcast_to(static[0].astype(np.float32), (t_, h, w)),
            "persistent": np.broadcast_to(static[1].astype(np.float32), (t_, h, w))}
    if C > 3:
        maps["births"] = xa[:, 3]
    return np.stack([maps[c] for c in cfg["channels"]], 1).astype(np.float32)


def sample_kwargs(cfg):
    kw = dict(sigma=cfg["sigma"], ignore_r=cfg["ignore_r"], min_target_ha=cfg["min_target_ha"],
              births="births" in cfg["channels"], rekindle_ignore=cfg["rekindle_ignore"])
    if cfg["time_bins"]:
        kw.update(time_bins=cfg["time_bins"], sigma_t=cfg["sigma_t"])
    return kw


def static_crop(data, crop=None):
    i0, j0, s = (0, 0, None) if crop is None else crop
    sl = (slice(None), slice(None)) if crop is None else (slice(i0, i0 + s), slice(j0, j0 + s))
    return data.static["pt"][sl], data.static["persistent"][sl]


class CropSet(torch.utils.data.Dataset):
    def __init__(self, data, splits, cfg, jitter=8, seed=0):
        self.data, self.cfg, self.size, self.jitter = data, cfg, cfg["crop"], jitter
        self.rng = np.random.default_rng(seed)
        splits = [splits] if isinstance(splits, str) else list(splits)
        per = data.periods[data.periods["split"].isin(splits)]
        pos = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(per["period"])]
        if cfg["min_target_ha"]:
            pos = pos[pos["fire_id"].map(data._area) >= cfg["min_target_ha"]]
        self.pos = pos.reset_index(drop=True)
        self._pt_idx = {}
        # n_det_pt uses the polygon test on each detection's lat/lon, the crop centre needs a detection on a
        # 'pt' cell; 4 periods of 2021-2022 have the first but not the second (notes.md 7.4)
        cand = per.loc[(per["n_targets"] == 0) & (per["n_det_pt"] > 0), "period"]
        self.neg_periods = cand[[len(self._pt_detections(p)) > 0 for p in cand]].reset_index(drop=True)
        self.n_neg = int(round(cfg["neg_ratio"] * len(self.pos)))
        self.resample()

    def resample(self):
        """New hard negatives for the next epoch."""
        self.neg = self.neg_periods.sample(self.n_neg, replace=self.n_neg > len(self.neg_periods),
                                           random_state=int(self.rng.integers(1 << 31))).reset_index(drop=True)

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def _pt_detections(self, period):
        """Indices of the window's detections on a mainland ('pt') cell."""
        if period not in self._pt_idx:
            d = self.data
            s0 = d.first_slot(period)
            a, b = np.searchsorted(d._slot, [s0, s0 + 96])
            idx = np.arange(a, b)
            self._pt_idx[period] = idx[d.static["pt"][d._i[idx], d._j[idx]]]
        return self._pt_idx[period]

    def _random_pt_detection(self, period):
        k = self.rng.choice(self._pt_detections(period))
        return self.data._i[k], self.data._j[k]

    def __getitem__(self, k):
        if k < len(self.pos):
            q = self.pos.iloc[k]
            period, ci, cj = q["period"], q["i_f"], q["j_f"]
        else:
            period = self.neg.iloc[k - len(self.pos)]
            ci, cj = self._random_pt_detection(period)
        crop = crop_origin(ci, cj, self.size, self.rng, self.jitter)
        x, y, w, _ = self.data.sample(period, crop, **sample_kwargs(self.cfg))
        f = features(x, static_crop(self.data, crop), self.cfg)
        if self.cfg["flip"] and self.rng.random() < 0.5:
            f, y, w = flip(f, -1), flip(y, -1), flip(w, -1)
        if self.cfg["flip"] and self.rng.random() < 0.5:
            f, y, w = flip(f, -2), flip(y, -2), flip(w, -2)
        return to_tensor(f), to_tensor(y), to_tensor(w)


def flip(a, axis):
    """Mirror a feature / target / weight array along a SPATIAL axis (-1 = east-west, -2 = north-south).
    The spatial axes are always the last two, so 2D maps (h, w), space-time targets (K, h, w) and
    features (T, C, h, w) are flipped alike and time is never flipped (notes.md 9.4, item 3).
    Tuples (space-time targets with their 2D part) are flipped element by element."""
    assert axis in (-1, -2)
    if isinstance(a, tuple):
        return tuple(flip(v, axis) for v in a)
    return a[..., ::-1] if axis == -1 else a[..., ::-1, :]


def to_tensor(a):
    return tuple(to_tensor(v) for v in a) if isinstance(a, tuple) else torch.from_numpy(np.ascontiguousarray(a))


def _worker_init(_):
    """Each DataLoader worker gets its own random stream (otherwise all workers would copy the
    same generator and draw identical jitters / flips)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


# ---------------------------------------------------------------- prediction

def heatmap_to_points(prob, thr=0.05, topk=50, window=3, subpixel=True, return_cells=False):
    """Local maxima (window x window) above thr; position refined to sub-pixel by the
    probability-weighted 3 x 3 centroid (subpixel=False: pixel centre). return_cells=True also
    returns the integer peak cells (used to read the time of a peak from a space-time map)."""
    p = torch.as_tensor(prob)[None, None]
    peak = (p == F.max_pool2d(p, window, stride=1, padding=window // 2)) & (p >= thr)
    ii, jj = torch.nonzero(peak[0, 0], as_tuple=True)
    s = p[0, 0, ii, jj]
    order = torch.argsort(s, descending=True)[:topk]
    ii, jj, s = ii[order].numpy(), jj[order].numpy(), s[order].numpy()
    if not subpixel:
        out = ii.astype(float), jj.astype(float), s
    else:
        fi, fj = _centroids(np.asarray(prob), ii, jj)
        out = fi, fj, s
    return (*out, ii, jj) if return_cells else out


def _centroids(prob, ii, jj):
    """Probability-weighted 3 x 3 centroid around each peak cell of a 2D map (zero-padded edges)."""
    pr = np.pad(prob, 1)
    di, dj = np.mgrid[-1:2, -1:2]
    fi, fj = [], []
    for a, b in zip(ii, jj):
        win = pr[a:a + 3, b:b + 3]
        fi.append(a + (win * di).sum() / win.sum()); fj.append(b + (win * dj).sum() / win.sum())
    return np.array(fi, float), np.array(fj, float)


def peak_time(prob3, k, a, b, bin_h=1.0):
    """Time of a peak at bin k, cell (a, b) of a space-time map (K, H, W), in hours since the period
    start: log-parabola through bins k - 1, k, k + 1 (exact for a Gaussian in time), shift limited to
    half a bin; bin centre at the first and last bin or if the log-profile is not concave."""
    K = prob3.shape[0]
    d = 0.0
    if 0 < k < K - 1:
        lp = np.log(np.clip(prob3[k - 1:k + 2, a, b].astype(float), 1e-6, 1.0))
        den = lp[0] - 2 * lp[1] + lp[2]
        if den < 0:
            d = float(np.clip(0.5 * (lp[0] - lp[2]) / den, -0.5, 0.5))
    return (k + 0.5 + d) * bin_h


def heatmap_to_points_3d(prob3, thr=0.05, topk=50, window_t=3, score="peak", prob2d=None, bin_h=1.0):
    """Peaks of a space-time map (K, H, W) (decoding D1 of notes.md 9.4, items 6-8): local maxima over
    window_t bins x 3 x 3 cells above thr (window_t = 2 tau + 1 drops a peak with a stronger one within
    1 cell and tau bins); position = 3 x 3 centroid inside the peak's bin; time = peak_time.
    score: 'peak' = peak probability; 'noisy_or' = 1 - prod(1 - p) over bins k - 1 .. k + 1 at the
    peak cell; 'x2d' = peak probability x the 2D map prob2d at the peak cell (auxiliary head).
    Returns fi, fj, t_h, score, ranked by score (topk)."""
    prob3 = np.asarray(prob3, np.float32)
    p = torch.as_tensor(prob3)[None, None]
    pool = F.max_pool3d(p, (window_t, 3, 3), stride=1, padding=(window_t // 2, 1, 1))
    kk, ii, jj = (v.numpy() for v in torch.nonzero(((p == pool) & (p >= thr))[0, 0], as_tuple=True))
    s = prob3[kk, ii, jj].astype(float)
    if score == "noisy_or":
        s = np.array([1 - np.prod(1 - prob3[max(k - 1, 0):k + 2, a, b]) for k, a, b in zip(kk, ii, jj)])
    elif score == "x2d":
        s = s * np.asarray(prob2d)[ii, jj]
    else:
        assert score == "peak", f"unknown score {score}"
    order = np.argsort(-s, kind="stable")[:topk]
    kk, ii, jj, s = kk[order], ii[order], jj[order], s[order]
    fi, fj = np.zeros(len(kk)), np.zeros(len(kk))
    for n, (k, a, b) in enumerate(zip(kk, ii, jj)):
        fi[n:n + 1], fj[n:n + 1] = _centroids(prob3[k], [a], [b])
    t_h = np.array([peak_time(prob3, k, a, b, bin_h) for k, a, b in zip(kk, ii, jj)])
    return fi, fj, t_h, s


def decode(m, decode="3d", bin_h=None, **peak):
    """One period's map -> fi, fj, score, t_h (None for a 2D map). m is a 2D map (H, W), a space-time
    map (K, H, W) or a pair (space-time map, 2D map of the auxiliary head). Decodings (notes.md 9.4,
    item 6): '3d' = D1, peaks in (time, i, j); 'max' = D2, maximum over the bins, then the 2D decoder;
    '2d' = D0, the 2D head with the 2D decoder. For 'max' and '2d' the time is read at the 2D peak cell
    from the bin with the highest probability (peak_time)."""
    p3, p2 = (m if isinstance(m, tuple) else (m, None))
    p3 = np.asarray(p3)
    unknown = set(peak) - {"thr", "topk", "window", "subpixel", "window_t", "score"}
    assert not unknown, f"unknown peak options {unknown}"
    k2 = {k: v for k, v in peak.items() if k in ("thr", "topk", "window", "subpixel")}     # 2D decoder
    k3 = {k: v for k, v in peak.items() if k in ("thr", "topk", "window_t", "score")}      # 3D decoder
    if p3.ndim == 2:
        fi, fj, s = heatmap_to_points(p3, **k2)
        return fi, fj, s, None
    bin_h = bin_h or 12 / p3.shape[0]
    if decode == "3d":
        fi, fj, t, s = heatmap_to_points_3d(p3, prob2d=p2, bin_h=bin_h, **k3)
        return fi, fj, s, t
    m2 = p3.max(0) if decode == "max" else (None if p2 is None else np.asarray(p2))
    assert m2 is not None and m2.ndim == 2, "decode '2d' needs the 2D map of the auxiliary head"
    fi, fj, s, ii, jj = heatmap_to_points(m2, return_cells=True, **k2)
    t = np.array([peak_time(p3, int(p3[:, a, b].argmax()), a, b, bin_h) for a, b in zip(ii, jj)])
    return fi, fj, s, t


@torch.no_grad()
def iter_maps(model, data, periods, cfg, device, batch=16, amp=False):
    """Yield {period: map} for `batch` periods at a time. A map is (H, W), or for cnn3d_st a pair
    ((K, H, W), (H, W)) (space-time map, auxiliary 2D map) or (K, H, W) without the 2D head.
    With one period per call the GPU mostly waits on the many small ConvLSTM steps (validation was
    ~90 % model time at batch 1). amp=True: half precision on CUDA (faster on e.g. T4; scores not
    bit-identical, differences tiny)."""
    model.eval()
    st = static_crop(data)
    periods = list(periods)
    kw = {k: v for k, v in sample_kwargs(cfg).items() if k not in ("time_bins", "sigma_t")}   # input only
    for k in range(0, len(periods), batch):
        chunk = periods[k:k + batch]
        f = np.stack([features(data.sample(p, **kw)[0], st, cfg) for p in chunk])
        # MPS returns silently wrong cnn3d_st logits for 16 full-grid periods per call (largest activation
        # 16 x 48 x 24 x 177 x 106 = 3.5e8 values; correct for <= 8, checked against the CPU on 2026-11-14),
        # so the model call is split there. Results do not depend on the split.
        sub = 8 if (device.type == "mps" and cfg["arch"] == "cnn3d_st") else len(chunk)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            outs = [model(torch.from_numpy(f[i:i + sub]).to(device)) for i in range(0, len(chunk), sub)]
        logits = (tuple(torch.cat(v) for v in zip(*outs)) if isinstance(outs[0], tuple) else torch.cat(outs))
        if isinstance(logits, tuple):
            p3, p2 = (torch.sigmoid(v.float()).cpu().numpy() for v in logits)
            yield dict(zip(chunk, zip(p3, p2)))
        else:
            yield dict(zip(chunk, torch.sigmoid(logits.float()).cpu().numpy()))


def predict_maps(model, data, periods, cfg, device, batch=16, amp=False):
    """Full-domain probability map of every period: {period: map} (see iter_maps). Keeps all maps in
    memory (validation of cnn3d_st: ~0.8 GB); predict() does not."""
    out = {}
    for m in iter_maps(model, data, periods, cfg, device, batch, amp):
        out.update(m)
    return out


def maps_to_points(maps, georef_fix=False, **peak):
    """Peaks of every map -> points (period, lat, lon, score) in the TRUE frame: with the
    geo-referencing correction (data.georef_fix), the sensor-frame peaks of periods before
    grid.GEOREF_CHANGE are moved half a pixel south-east (notes.md 6.11). Space-time maps (see
    decode(), options decode / window_t / score / bin_h in **peak) also give t_h, the ignition time in
    hours since the period start."""
    rows = []
    for p, prob in maps.items():
        fi, fj, s, t = decode(prob, **peak)
        if georef_fix:
            fi, fj = G.sensor_to_true_ij(fi, fj, p)
        lat, lon = G.ij_to_latlon(fi, fj)
        df = pd.DataFrame({"period": p, "lat": lat, "lon": lon, "score": s})
        if t is not None:
            df["t_h"] = t
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["period", "lat", "lon", "score"])


def predict(model, data, periods, cfg, device, batch=16, amp=False, decodes=None, **peak):
    """Points of every period, made batch by batch. decodes=None: one DataFrame (decoding from **peak,
    default '3d' for space-time maps); decodes=('3d', 'max', '2d'): {decoding: DataFrame} from one pass."""
    modes = [peak.pop("decode", "3d")] if decodes is None else list(decodes)
    parts = {m: [] for m in modes}
    for maps in iter_maps(model, data, periods, cfg, device, batch, amp):
        for m in modes:
            parts[m].append(maps_to_points(maps, data.georef_fix, decode=m, **peak))
    out = {m: pd.concat(v, ignore_index=True) if v else maps_to_points({}, data.georef_fix) for m, v in parts.items()}
    return out[modes[0]] if decodes is None else out


def load_model(name, device):
    """Model, full configuration and raw checkpoint of models/<name>.pt (own checkpoints; v1
    stored numpy scalars in its metrics, hence weights_only=False)."""
    ck = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    cfg = config_from(ck["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, cfg, ck


# ---------------------------------------------------------------- training

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convlstm_v1", help="run name; may contain a sub-folder, e.g. ablations/ref_s0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke test)")
    ap.add_argument("--val-periods", type=int, default=0, help="limit validation periods (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (Colab: 2)")
    ap.add_argument("--val-batch", type=int, default=16, help="validation periods per model call")
    ap.add_argument("--val-amp", action="store_true", help="half-precision validation on CUDA (faster, not bit-identical)")
    ap.add_argument("--train-splits", default="train", help="comma list of splits to train on (final models: train,val)")
    ap.add_argument("--no-val", action="store_true",
                    help="no validation; the checkpoint of the last epoch is saved (final models trained on train+val)")
    # input
    ap.add_argument("--step-agg", type=int, default=DEFAULTS["step_agg"], help="15-min slots per model step (4 = hourly)")
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS), help=f"comma list from {','.join(ALL_CHANNELS)}")
    ap.add_argument("--history-h", type=int, default=DEFAULTS["history_h"], help=f"hours of history kept (0-{HISTORY_H})")
    ap.add_argument("--tail-h", type=int, default=DEFAULTS["tail_h"], help=f"hours of tail kept (0-{TAIL_H})")
    # model
    ap.add_argument("--arch", default=DEFAULTS["arch"], choices=["convlstm", "cnn3d", "cnn3d_st"])
    ap.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    ap.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    ap.add_argument("--kernel", type=int, default=DEFAULTS["kernel"])
    ap.add_argument("--readout", default=DEFAULTS["readout"], choices=["last", "max", "mean"])
    # targets, loss, sampling
    ap.add_argument("--crop", type=int, default=DEFAULTS["crop"])
    ap.add_argument("--sigma", type=float, default=DEFAULTS["sigma"], help="target Gaussian sigma [px]")
    ap.add_argument("--ignore-r", type=float, default=DEFAULTS["ignore_r"], help="ignore radius [px] in training (0 = none)")
    ap.add_argument("--min-target-ha", type=float, default=None, help="targets below this size become ignore (default 50 ha as built)")
    ap.add_argument("--loss", default=DEFAULTS["loss"], choices=sorted(LOSSES))
    ap.add_argument("--neg-ratio", type=float, default=DEFAULTS["neg_ratio"], help="hard negatives per positive")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--rekindle-ignore", action="store_true", help="ignore regions at all ICNF rekindle records (ablation)")
    ap.add_argument("--time-bins", type=int, default=DEFAULTS["time_bins"], help="space-time targets: bins in the 12 h period (cnn3d_st: 12)")
    ap.add_argument("--sigma-t", type=float, default=DEFAULTS["sigma_t"], help="space-time targets: Gaussian width in time [h]")
    ap.add_argument("--aux2d-weight", type=float, default=DEFAULTS["aux2d_weight"], help="cnn3d_st: weight of the 2D head's loss (0 = no 2D head)")
    args = ap.parse_args(argv)
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(chans) - set(ALL_CHANNELS)
    assert not unknown, f"unknown channels {unknown}"
    assert 0 <= args.history_h <= HISTORY_H and 0 <= args.tail_h <= TAIL_H
    assert (args.arch == "cnn3d_st") == (args.time_bins > 0), "--time-bins goes with --arch cnn3d_st (and vice versa)"
    return args, chans


def main():
    args, chans = parse_args()
    cfg = config_from({**vars(args), "channels": chans, "flip": not args.no_flip})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device()
    data = FireData()
    train_splits = [x.strip() for x in args.train_splits.split(",") if x.strip()]
    assert "test" not in train_splits, "never train on the test split"
    assert args.no_val or "val" not in train_splits, "training on 'val' requires --no-val (no validation data left)"
    train = CropSet(data, train_splits, cfg, seed=args.seed)
    val_periods = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    if args.val_periods:
        val_periods = val_periods.sample(args.val_periods, random_state=0)
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = LOSSES[cfg["loss"]]
    loss_kw = {} if cfg["loss"] == "mse" else {"pos_thr": pos_threshold(cfg["sigma"])}
    st_model = cfg["time_bins"] > 0
    if st_model:
        loss_kw3 = {} if cfg["loss"] == "mse" else {"pos_thr": pos_threshold(cfg["sigma"], cfg["sigma_t"], PERIOD_H / cfg["time_bins"])}

    def compute_loss(out, y, w):
        if not st_model:
            return loss_fn(out, y, w, **loss_kw)
        l3, l2 = out if isinstance(out, tuple) else (out, None)
        loss = loss_fn(l3, y[0], w[0], **loss_kw3)
        if l2 is not None:
            loss = loss + cfg["aux2d_weight"] * loss_fn(l2, y[1], w[1], **loss_kw)
        return loss

    to_dev = lambda v: [u.to(device) for u in v] if isinstance(v, (list, tuple)) else v.to(device)
    print(f"device {device}; parameters {n_par:,}; train crops per epoch {len(train):,} "
          f"({len(train.pos):,} positive); validation periods {len(val_periods):,}; config {cfg}", flush=True)

    ck_path = MODELS / f"{args.name}.pt"
    hist_path = OUTPUTS / f"{args.name}_history.csv"
    ck_path.parent.mkdir(parents=True, exist_ok=True); hist_path.parent.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); train.resample()
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                             worker_init_fn=_worker_init if args.workers else None,
                                             pin_memory=device.type == "cuda")
        t0, losses = time.time(), []
        for b, (f, y, w) in enumerate(loader):
            if args.max_batches and b >= args.max_batches:
                break
            f, y, w = f.to(device), to_dev(y), to_dev(w)
            loss = compute_loss(model(f), y, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_min": (time.time() - t0) / 60}
        if args.no_val:
            if epoch == args.epochs:                                   # last epoch = final model
                torch.save({"state_dict": model.state_dict(), "config": {**vars(args), **cfg}, "epoch": epoch, "val": {}},
                           ck_path)
                rec["saved"] = True
        elif epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            decodes = (("3d", "max", "2d") if model.head2d is not None else ("3d", "max")) if st_model else ("3d",)
            preds_all = predict(model, data, val_periods, cfg, device, batch=args.val_batch, amp=args.val_amp, decodes=decodes)
            preds = preds_all["3d"]                       # 2D models: plain 2D decoding; cnn3d_st: D1
            groups = ("isolated", "in_pair", "already_burning") if st_model else ("isolated",)
            for k in (5, 10):
                res, _, _ = evaluate(preds, data, "val", k, groups=groups)
                rec.update({f"AP@{k}": res["AP"], f"recall@{k}": res["recall (all preds)"], f"F1@{k}": res["best F1"],
                            f"median_err@{k}": res["median error TP [km]"],
                            f"recall_nonisolated@{k}": res.get("recall | isolated=False", np.nan),
                            f"recall_isolated@{k}": res.get("recall | isolated=True", np.nan)})
                if st_model:
                    rec.update({f"recall_in_pair@{k}": res.get("recall | in_pair=True", np.nan),
                                f"recall_already_burning@{k}": res.get("recall | already_burning=True", np.nan),
                                f"median_abs_dt@{k}": res.get("median |dt| TP [h]", np.nan),
                                f"dt_le_1h@{k}": res.get("|dt| <= 1 h", np.nan)})
                    rec[f"AP_tau2h@{k}"] = evaluate(preds, data, "val", k, groups=(), tau_h=2)[0]["AP"]
                    for d in decodes[1:]:                 # D2 ('max') and D0 ('2d') of the same maps
                        rec[f"AP@{k}_{d}"] = evaluate(preds_all[d], data, "val", k, groups=())[0]["AP"]
            rec["val_min"] = (time.time() - t1) / 60
            score = (rec["AP@5"] + rec["AP@10"]) / 2
            if score > best:
                best = score
                val = {k: float(v) for k, v in rec.items() if k != "epoch"}   # plain floats: loads with weights_only
                torch.save({"state_dict": model.state_dict(), "config": {**vars(args), **cfg}, "epoch": epoch, "val": val},
                           ck_path)
                rec["saved"] = True
        history.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        pd.DataFrame(history).to_csv(hist_path, index=False)          # after every epoch (resumable logs)
    done = {"name": args.name, "best_mean_ap": best, "epochs": args.epochs, "config": cfg}
    (OUTPUTS / f"{args.name}_done.json").write_text(json.dumps(done, indent=1, default=str))   # run complete
    print(f"best mean AP (val) {best:.4f}; checkpoint {ck_path.relative_to(E.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
