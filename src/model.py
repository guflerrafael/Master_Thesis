"""Ignition-heatmap models and losses (notes.md 6.1, 6.7).

IgnitionConvLSTM: ConvLSTM cell after Shi et al. (2015), without peephole connections. The
network reads the input sequence step by step and predicts one ignition heatmap (logits) for the
target period. Readout 'last' (v1/v2): head on the final hidden state of the top layer; 'max' /
'mean': head on the maximum / mean of the top hidden state over all steps.
TemporalCNN3D: non-recurrent comparison model (3D convolutions over time and space, then max and
mean over time), for the ablation "is the recurrent memory needed?".
All models are fully convolutional: trained on crops, applied to the full grid.

build_model(cfg) creates the model of a training configuration (see train.DEFAULTS).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        i, f, o, g = torch.split(self.gates(torch.cat([x, h], 1)), self.hid_ch, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


def _init_prior(conv, prior):
    """Start with a low ignition probability everywhere (focal-loss initialisation)."""
    nn.init.constant_(conv.bias, -torch.log(torch.tensor((1 - prior) / prior)).item())


class IgnitionConvLSTM(nn.Module):
    def __init__(self, in_ch, enc_ch=16, hid=(32, 32), kernel=3, prior=0.01, readout="last"):
        super().__init__()
        assert readout in ("last", "max", "mean")
        self.readout = readout
        self.encoder = nn.Sequential(nn.Conv2d(in_ch, enc_ch, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(enc_ch, enc_ch, 3, padding=1), nn.ReLU())
        chans = (enc_ch,) + tuple(hid)
        self.cells = nn.ModuleList(ConvLSTMCell(chans[k], chans[k + 1], kernel) for k in range(len(hid)))
        self.head = nn.Sequential(nn.Conv2d(hid[-1], 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        _init_prior(self.head[-1], prior)

    def forward(self, x):
        """x: (B, T, C, H, W) -> logits (B, H, W)"""
        B, T, _, H, W = x.shape
        states = [(x.new_zeros(B, cell.hid_ch, H, W), x.new_zeros(B, cell.hid_ch, H, W)) for cell in self.cells]
        pooled = None
        for t in range(T):
            z = self.encoder(x[:, t])
            for k, cell in enumerate(self.cells):
                states[k] = cell(z, states[k])
                z = states[k][0]
            if self.readout == "max":
                pooled = z if pooled is None else torch.maximum(pooled, z)
            elif self.readout == "mean":
                pooled = z / T if pooled is None else pooled + z / T
        return self.head(z if self.readout == "last" else pooled).squeeze(1)


class TemporalCNN3D(nn.Module):
    """No recurrence: 3D convolutions over (time, space), time pooled by max and mean."""

    def __init__(self, in_ch, width=32, prior=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv3d(16, width, 3, padding=1, stride=(2, 1, 1)), nn.ReLU(),
            nn.Conv3d(width, width, 3, padding=1, stride=(2, 1, 1)), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Conv2d(2 * width, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        _init_prior(self.head[-1], prior)

    def forward(self, x):
        z = self.net(x.permute(0, 2, 1, 3, 4))            # (B, C, T', H, W)
        return self.head(torch.cat([z.amax(2), z.mean(2)], 1)).squeeze(1)


def build_model(cfg):
    in_ch = len(cfg["channels"])
    if cfg["arch"] == "cnn3d":
        return TemporalCNN3D(in_ch, width=cfg["hidden"])
    return IgnitionConvLSTM(in_ch, hid=(cfg["hidden"],) * cfg["layers"], kernel=cfg["kernel"], readout=cfg["readout"])


# ---------------------------------------------------------------- losses

def focal_loss(logits, y, w, alpha=2.0, beta=4.0, pos_thr=0.75):
    """Penalty-reduced focal loss of CenterNet (Zhou et al., 2019) on Gaussian targets, with a
    per-pixel weight w (0 = ignore). Positives: pixels with y >= pos_thr (the pixel nearest each
    sub-pixel ignition always qualifies for sigma = 1 px). Normalised by the number of positives."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (y >= pos_thr).float() * w
    neg = (y < pos_thr).float() * w
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - y) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)


def bce_loss(logits, y, w, pos_thr=0.75):
    """Weighted binary cross-entropy against the soft Gaussian targets, normalised like the focal
    loss (per positive pixel) so the learning rate stays comparable."""
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w
    return l.sum() / ((y >= pos_thr).float() * w).sum().clamp(min=1.0)


def mse_loss(logits, y, w):
    """Weighted mean squared error between the probability map and the Gaussian targets."""
    return (((torch.sigmoid(logits) - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)


LOSSES = {"focal": focal_loss, "bce": bce_loss, "mse": mse_loss}
