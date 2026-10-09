"""Train the ConvLSTM ignition-heatmap model and validate it like the baselines (notes.md 6).

Training: 32 x 32 crops. Positives = one crop per target ignition of the training years
(random jitter +-8 px), negatives = as many crops around a random detection in periods without
a target (hard negatives), drawn anew every epoch; random horizontal/vertical flips.
Validation: full-domain prediction for every validation period with detections, peaks ->
points, scored with src/evaluate.py (k = 5 and 10 km). The best epoch (mean of AP@5 and
AP@10) is kept. The test split is never touched here.

Run from the repository root:
  .venv/bin/python src/train.py --name convlstm_v1 > outputs/convlstm_v1_log.txt
  .venv/bin/python src/train.py --name smoke --epochs 1 --max-batches 5 --val-periods 20   (quick check)
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
from model import IgnitionConvLSTM, focal_loss
from tiling_design import HISTORY_H, PERIOD_H

MODELS = E.ROOT / "models"
OUTPUTS = E.ROOT / "outputs"


# ---------------------------------------------------------------- features

def features(x, static, step_agg):
    """(96, 3, h, w) sample input -> (T', 7, h, w): per-hour max of FRP / confidence / detected,
    'inside target period' flag, normalised time, mainland mask, persistent-pixel flag."""
    T, C, h, w = x.shape
    t_ = T // step_agg
    xa = x.reshape(t_, step_agg, C, h, w).max(1)
    t = np.arange(t_)
    a, b = HISTORY_H * 4 // step_agg, (HISTORY_H + PERIOD_H) * 4 // step_agg
    tc = np.stack([((t >= a) & (t < b)).astype(np.float32), (t / max(t_ - 1, 1)).astype(np.float32)], 1)
    tc = np.broadcast_to(tc[:, :, None, None], (t_, 2, h, w))
    st = np.broadcast_to(np.stack(static)[None].astype(np.float32), (t_, 2, h, w))
    return np.concatenate([xa, tc, st], 1).astype(np.float32)


def static_crop(data, crop=None):
    i0, j0, s = (0, 0, None) if crop is None else crop
    sl = (slice(None), slice(None)) if crop is None else (slice(i0, i0 + s), slice(j0, j0 + s))
    return data.static["pt"][sl], data.static["persistent"][sl]


class CropSet(torch.utils.data.Dataset):
    def __init__(self, data, split, step_agg, size=32, jitter=8, seed=0):
        self.data, self.step_agg, self.size, self.jitter = data, step_agg, size, jitter
        self.rng = np.random.default_rng(seed)
        per = data.periods[data.periods["split"] == split]
        self.pos = data.labels[(data.labels["role"] == "target") & data.labels["period"].isin(per["period"])].reset_index(drop=True)
        self.neg_periods = per.loc[(per["n_targets"] == 0) & (per["n_det_pt"] > 0), "period"].reset_index(drop=True)
        self._pt_idx = {}
        self.resample()

    def resample(self):
        """New hard negatives for the next epoch."""
        self.neg = self.neg_periods.sample(len(self.pos), replace=len(self.pos) > len(self.neg_periods),
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
        x, y, w, _ = self.data.sample(period, crop)
        f = features(x, static_crop(self.data, crop), self.step_agg)
        if self.rng.random() < 0.5:
            f, y, w = f[..., ::-1], y[:, ::-1], w[:, ::-1]
        if self.rng.random() < 0.5:
            f, y, w = f[..., ::-1, :], y[::-1], w[::-1]
        return torch.from_numpy(f.copy()), torch.from_numpy(y.copy()), torch.from_numpy(w.copy())


# ---------------------------------------------------------------- prediction

def heatmap_to_points(prob, thr=0.05, topk=50):
    """Local maxima (3 x 3) above thr -> sub-pixel position (probability-weighted 3 x 3 centroid)."""
    p = torch.as_tensor(prob)[None, None]
    peak = (p == F.max_pool2d(p, 3, stride=1, padding=1)) & (p >= thr)
    ii, jj = torch.nonzero(peak[0, 0], as_tuple=True)
    s = p[0, 0, ii, jj]
    order = torch.argsort(s, descending=True)[:topk]
    ii, jj, s = ii[order].numpy(), jj[order].numpy(), s[order].numpy()
    pr = np.pad(np.asarray(prob), 1)
    fi, fj = [], []
    for a, b in zip(ii, jj):
        win = pr[a:a + 3, b:b + 3]
        di, dj = np.mgrid[-1:2, -1:2]
        fi.append(a + (win * di).sum() / win.sum()); fj.append(b + (win * dj).sum() / win.sum())
    return np.array(fi), np.array(fj), s


@torch.no_grad()
def predict(model, data, periods, step_agg, device):
    model.eval()
    st = static_crop(data)
    rows = []
    for p in periods:
        x, _, _, _ = data.sample(p)
        f = torch.from_numpy(features(x, st, step_agg))[None].to(device)
        prob = torch.sigmoid(model(f))[0].cpu().numpy()
        fi, fj, s = heatmap_to_points(prob)
        lat, lon = G.ij_to_latlon(fi, fj)
        rows.append(pd.DataFrame({"period": p, "lat": lat, "lon": lon, "score": s}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["period", "lat", "lon", "score"])


# ---------------------------------------------------------------- training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convlstm_v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step-agg", type=int, default=4, help="15-min steps per model step (4 = hourly)")
    ap.add_argument("--val-every", type=int, default=2)
    ap.add_argument("--max-batches", type=int, default=0, help="limit batches per epoch (smoke test)")
    ap.add_argument("--val-periods", type=int, default=0, help="limit validation periods (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = FireData()
    train = CropSet(data, "train", args.step_agg, seed=args.seed)
    val_periods = data.periods.loc[(data.periods["split"] == "val") & (data.periods["n_det"] > 0), "period"]
    if args.val_periods:
        val_periods = val_periods.sample(args.val_periods, random_state=0)
    model = IgnitionConvLSTM(in_ch=7).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"device {device}; parameters {n_par:,}; train crops per epoch {len(train):,} "
          f"({len(train.pos):,} positive); validation periods {len(val_periods):,}; config {vars(args)}", flush=True)

    MODELS.mkdir(exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); train.resample()
        loader = torch.utils.data.DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
        t0, losses = time.time(), []
        for b, (f, y, w) in enumerate(loader):
            if args.max_batches and b >= args.max_batches:
                break
            f, y, w = f.to(device), y.to(device), w.to(device)
            loss = focal_loss(model(f), y, w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); losses.append(loss.item())
        sched.step()
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_min": (time.time() - t0) / 60}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            t1 = time.time()
            preds = predict(model, data, val_periods, args.step_agg, device)
            for k in (5, 10):
                res, _, _ = evaluate(preds, data, "val", k)
                rec.update({f"AP@{k}": res["AP"], f"recall@{k}": res["recall (all preds)"], f"F1@{k}": res["best F1"],
                            f"median_err@{k}": res["median error TP [km]"]})
            rec["val_min"] = (time.time() - t1) / 60
            score = (rec["AP@5"] + rec["AP@10"]) / 2
            if score > best:
                best = score
                val = {k: float(v) for k, v in rec.items() if k != "epoch"}   # plain floats: loads with weights_only
                torch.save({"state_dict": model.state_dict(), "config": vars(args), "epoch": epoch, "val": val},
                           MODELS / f"{args.name}.pt")
                rec["saved"] = True
        history.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    pd.DataFrame(history).to_csv(OUTPUTS / f"{args.name}_history.csv", index=False)
    print(f"best mean AP (val) {best:.4f}; checkpoint models/{args.name}.pt", flush=True)


if __name__ == "__main__":
    main()
