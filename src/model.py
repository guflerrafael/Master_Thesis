"""ConvLSTM ignition-heatmap model (notes.md 6.1).

ConvLSTM cell after Shi et al. (2015), without peephole connections. The network reads the
input sequence step by step and predicts one ignition heatmap (logits) for the target period
from the final hidden state. Fully convolutional: trained on crops, applied to the full grid.
"""
import torch
import torch.nn as nn


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


class IgnitionConvLSTM(nn.Module):
    def __init__(self, in_ch, enc_ch=16, hid=(32, 32), kernel=3, prior=0.01):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_ch, enc_ch, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(enc_ch, enc_ch, 3, padding=1), nn.ReLU())
        chans = (enc_ch,) + tuple(hid)
        self.cells = nn.ModuleList(ConvLSTMCell(chans[k], chans[k + 1], kernel) for k in range(len(hid)))
        self.head = nn.Sequential(nn.Conv2d(hid[-1], 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1))
        # start with a low ignition probability everywhere (focal-loss initialisation)
        nn.init.constant_(self.head[-1].bias, -torch.log(torch.tensor((1 - prior) / prior)).item())

    def forward(self, x):
        """x: (B, T, C, H, W) -> logits (B, H, W)"""
        B, T, _, H, W = x.shape
        states = [(x.new_zeros(B, cell.hid_ch, H, W), x.new_zeros(B, cell.hid_ch, H, W)) for cell in self.cells]
        for t in range(T):
            z = self.encoder(x[:, t])
            for k, cell in enumerate(self.cells):
                states[k] = cell(z, states[k])
                z = states[k][0]
        return self.head(z).squeeze(1)


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
