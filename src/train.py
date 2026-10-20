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
                crop=32, sigma=1.0, ignore_r=2.0, min_target_ha=None, loss="focal", neg_ratio=1.0, flip=True)


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


def pos_threshold(sigma):
    """Positive pixels of the loss: y >= this value. 0.75 for sigma >= 1 px (v1/v2); lower for
    narrower targets so that the pixel nearest each sub-pixel ignition always counts."""
    return min(0.75, 0.97 * float(np.exp(-0.5 / (2 * sigma ** 2))))


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
    return dict(sigma=cfg["sigma"], ignore_r=cfg["ignore_r"], min_target_ha=cfg["min_target_ha"],
                births="births" in cfg["channels"])


def static_crop(data, crop=None):
    i0, j0, s = (0, 0, None) if crop is None else crop
    sl = (slice(None), slice(None)) if crop is None else (slice(i0, i0 + s), slice(j0, j0 + s))
    return data.static["pt"][sl], data.static["persistent"][sl]


class CropSet(torch.utils.data.Dataset):
    def __init__(self, data, split, cfg, jitter=8, seed=0):
        self.data, self.cfg, self.size, self.jitter = data, cfg, cfg["crop"], jitter
        self.rng = np.random.default_rng(seed)
        per = data.periods[data.periods["split"] == split]
        pos = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(per["period"])]
        if cfg["min_target_ha"]:
            pos = pos[pos["fire_id"].map(data._area) >= cfg["min_target_ha"]]
        self.pos = pos.reset_index(drop=True)
        self.neg_periods = per.loc[(per["n_targets"] == 0) & (per["n_det_pt"] > 0), "period"].reset_index(drop=True)
        self.n_neg = int(round(cfg["neg_ratio"] * len(self.pos)))
        self._pt_idx = {}
        self.resample()

    def resample(self):
        """New hard negatives for the next epoch."""
        self.neg = self.neg_periods.sample(self.n_neg, replace=self.n_neg > len(self.neg_periods),
                                           random_state=int(self.rng.integers(1 << 31))).reset_index(drop=True)

    def __len__(self):
        return len(self.pos) + len(self.neg)

    def _random_pt_detection(self, period):
        if period not in self._pt_idx:
            d = self.data
            s0 = d.first_slot(period)
            a, b = np.searchsorted(d._slot, [s0, s0 + 96])
            idx = np.arange(a, b)
            self._pt_idx[period] = idx[d.static["pt"][d._i[idx], d._j[idx]]]
        k = self.rng.choice(self._pt_idx[period])
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
            f, y, w = f[..., ::-1], y[:, ::-1], w[:, ::-1]
        if self.cfg["flip"] and self.rng.random() < 0.5:
            f, y, w = f[..., ::-1, :], y[::-1], w[::-1]
        return torch.from_numpy(f.copy()), torch.from_numpy(y.copy()), torch.from_numpy(w.copy())


def _worker_init(_):
    """Each DataLoader worker gets its own random stream (otherwise all workers would copy the
    same generator and draw identical jitters / flips)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


# ---------------------------------------------------------------- prediction

def heatmap_to_points(prob, thr=0.05, topk=50, window=3, subpixel=True):
    """Local maxima (window x window) above thr; position refined to sub-pixel by the
    probability-weighted 3 x 3 centroid (subpixel=False: pixel centre)."""
    p = torch.as_tensor(prob)[None, None]
    peak = (p == F.max_pool2d(p, window, stride=1, padding=window // 2)) & (p >= thr)
    ii, jj = torch.nonzero(peak[0, 0], as_tuple=True)
    s = p[0, 0, ii, jj]
    order = torch.argsort(s, descending=True)[:topk]
    ii, jj, s = ii[order].numpy(), jj[order].numpy(), s[order].numpy()
    if not subpixel:
        return ii.astype(float), jj.astype(float), s
    pr = np.pad(np.asarray(prob), 1)
    di, dj = np.mgrid[-1:2, -1:2]
    fi, fj = [], []
    for a, b in zip(ii, jj):
        win = pr[a:a + 3, b:b + 3]
        fi.append(a + (win * di).sum() / win.sum()); fj.append(b + (win * dj).sum() / win.sum())
    return np.array(fi), np.array(fj), s


@torch.no_grad()
def predict_maps(model, data, periods, cfg, device, batch=16, amp=False):
    """Full-domain probability map of every period: {period: (H, W) array}. Periods are run
    through the model `batch` at a time: with one period per call the GPU mostly waits on the
    many small ConvLSTM steps (validation was ~90 % model time at batch 1). amp=True: half
    precision on CUDA (faster on e.g. T4; scores not bit-identical, differences tiny)."""
    model.eval()
    st = static_crop(data)
    periods = list(periods)
    out = {}
    for k in range(0, len(periods), batch):
        chunk = periods[k:k + batch]
        f = np.stack([features(data.sample(p, **sample_kwargs(cfg))[0], st, cfg) for p in chunk])
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(torch.from_numpy(f).to(device))
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        out.update(zip(chunk, prob))
    return out


def maps_to_points(maps, **peak):
    rows = []
    for p, prob in maps.items():
        fi, fj, s = heatmap_to_points(prob, **peak)
        lat, lon = G.ij_to_latlon(fi, fj)
        rows.append(pd.DataFrame({"period": p, "lat": lat, "lon": lon, "score": s}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["period", "lat", "lon", "score"])


def predict(model, data, periods, cfg, device, batch=16, amp=False, **peak):
    return maps_to_points(predict_maps(model, data, periods, cfg, device, batch, amp), **peak)


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
    # input
    ap.add_argument("--step-agg", type=int, default=DEFAULTS["step_agg"], help="15-min slots per model step (4 = hourly)")
    ap.add_argument("--channels", default=",".join(DEFAULT_CHANNELS), help=f"comma list from {','.join(ALL_CHANNELS)}")
    ap.add_argument("--history-h", type=int, default=DEFAULTS["history_h"], help=f"hours of history kept (0-{HISTORY_H})")
    ap.add_argument("--tail-h", type=int, default=DEFAULTS["tail_h"], help=f"hours of tail kept (0-{TAIL_H})")
    # model
    ap.add_argument("--arch", default=DEFAULTS["arch"], choices=["convlstm", "cnn3d"])
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
    args = ap.parse_args(argv)
    chans = [c.strip() for c in args.channels.split(",") if c.strip()]
    unknown = set(chans) - set(ALL_CHANNELS)
    assert not unknown, f"unknown channels {unknown}"
    assert 0 <= args.history_h <= HISTORY_H and 0 <= args.tail_h <= TAIL_H
    return args, chans


def main():
    args, chans = parse_args()
    cfg = config_from({**vars(args), "channels": chans, "flip": not args.no_flip})
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = get_device()
    data = FireData()
    train = CropSet(data, "train", cfg, seed=args.seed)
    val_periods = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    if args.val_periods:
        val_periods = val_periods.sample(args.val_periods, random_state=0)
    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = LOSSES[cfg["loss"]]
    loss_kw = {} if cfg["loss"] == "mse" else {"pos_thr": pos_threshold(cfg["sigma"])}
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
            f, y, w = f.to(device), y.to(device), w.to(device)
            loss = loss_fn(model(f), y, w, **loss_kw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_min": (time.time() - t0) / 60}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            preds = predict(model, data, val_periods, cfg, device, batch=args.val_batch, amp=args.val_amp)
            for k in (5, 10):
                res, _, _ = evaluate(preds, data, "val", k)
                rec.update({f"AP@{k}": res["AP"], f"recall@{k}": res["recall (all preds)"], f"F1@{k}": res["best F1"],
                            f"median_err@{k}": res["median error TP [km]"],
                            f"recall_nonisolated@{k}": res.get("recall | isolated=False", np.nan),
                            f"recall_isolated@{k}": res.get("recall | isolated=True", np.nan)})
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
