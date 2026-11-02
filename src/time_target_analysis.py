"""Can the ignition time be a target? Label quality and simple time estimators (notes.md 6.19).

Targets (≥ 50 ha, seen) of the train + validation years 2004-2016 only (design analysis; the
test years stay out). Positions in the sensor frame (labels.parquet), detections on their cells.
  A  precision of the ICNF start times (minute of the hour)
  B  where the start falls in the 24 h input window (history / period / tail)
  C  oracle-location estimator: first SEVIRI detection within R_PX cells of the ICNF point,
     [start - 1 h, start + 6 h] -> lag to the ICNF start = the floor a learned time could reach
     from detections alone; plus the share of targets whose place was already burning
     (a detection within R_PX cells in [start - 6 h, start - 30 min]), where timing is ambiguous
  D  location-free estimator: birth time of the nucleus born nearest to the ICNF point
     (≤ 5 km, birth in [start - 6 h, start + 12 h]), i.e. what the baseline would report
  E  close target pairs in the same period (≤ 2 cells apart): their start-time differences,
     i.e. how many merged pairs a time axis could separate

Run from the repository root:
  .venv/bin/python src/time_target_analysis.py > outputs/time_target_analysis.txt
Figure: figures/time/time_target_analysis.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from dataset import FireData
from tiling_design import HISTORY_H, PERIOD_H

FIG = E.ROOT / "figures" / "time"
R_PX = 1.5          # oracle-location window around the ICNF point (cells)
PAIR_PX = 2.0       # "close pair": two targets of one period at most this far apart (cells)


def describe(name, lag_h):
    lag_h = pd.Series(lag_h).dropna()
    q = lag_h.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"{name}: n = {len(lag_h)}, median {q[0.5]:+.2f} h, IQR {q[0.25]:+.2f} … {q[0.75]:+.2f} h, "
          f"p10–p90 {q[0.1]:+.2f} … {q[0.9]:+.2f} h; |error| median {lag_h.abs().median():.2f} h, "
          f"≤ 1 h {np.mean(lag_h.abs() <= 1):.0%}, ≤ 2 h {np.mean(lag_h.abs() <= 2):.0%}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    data = FireData()
    fires = data.fires.set_index("fire_id")
    tg = data.labels[data.labels["role"] == "target"].copy()
    tg = tg[tg["period"].map(data.periods.set_index("period")["split"]).isin(["train", "val"])]
    tg = tg.join(fires[["start_utc", "burned_area_ha", "isolated", "lat", "lon"]], on="fire_id")
    # nuclei come from the uncorrected detections: compare them with the ICNF point in the sensor frame
    tg["lat_s"], tg["lon_s"] = G.true_to_sensor_latlon(tg["lat"].values, tg["lon"].values, tg["start_utc"].values)
    print(f"targets 2004–2016 (train + val): {len(tg)}\n")

    # A: precision of the start times (recorded in local time; minutes are unaffected by the UTC shift)
    m = tg["start_utc"].dt.minute
    print("A  ICNF start time, minute of the hour:")
    print(f"   :00 {np.mean(m == 0):.0%}, :30 {np.mean(m == 30):.0%}, other multiples of 5 {np.mean((m % 5 == 0) & ~m.isin([0, 30])):.0%}, "
          f"not a multiple of 5 {np.mean(m % 5 != 0):.0%}\n")

    # B: position of the start in the 24 h window
    t_win = (tg["start_utc"] - (tg["period"] - pd.Timedelta(hours=HISTORY_H))).dt.total_seconds() / 3600
    tg["t_win"] = t_win
    print("B  start in hours since window start (6–18 = the 12 h period):")
    print(f"   min {t_win.min():.2f}, max {t_win.max():.2f}; by hour of the period: "
          + ", ".join(f"{int(h)}:{n}" for h, n in (t_win - HISTORY_H).astype(int).value_counts().sort_index().items()) + "\n")

    # C: first detection near the ICNF point
    det = data.det
    first = []
    for q in tg.itertuples():
        a, b = q.start_utc - pd.Timedelta(hours=1), q.start_utc + pd.Timedelta(hours=6)
        lo, hi = np.searchsorted(det["time_utc"].values, np.array([a.tz_convert(None), b.tz_convert(None)], dtype="datetime64[ns]"))
        d = det.iloc[lo:hi]
        near = np.hypot(d["i"].values - q.i_f, d["j"].values - q.j_f) <= R_PX
        first.append(d["time_utc"].values[near].min() if near.any() else np.datetime64("NaT"))
    tg["first_det"] = pd.to_datetime(first, utc=True)
    pre = []
    for q in tg.itertuples():
        a, b = q.start_utc - pd.Timedelta(hours=6), q.start_utc - pd.Timedelta(minutes=30)
        lo, hi = np.searchsorted(det["time_utc"].values, np.array([a.tz_convert(None), b.tz_convert(None)], dtype="datetime64[ns]"))
        d = det.iloc[lo:hi]
        pre.append((np.hypot(d["i"].values - q.i_f, d["j"].values - q.j_f) <= R_PX).any())
    tg["already_burning"] = pre
    tg["lag_first"] = (tg["first_det"] - tg["start_utc"]).dt.total_seconds() / 3600
    print(f"C  oracle location: first detection ≤ {R_PX} cells from the ICNF point minus ICNF start")
    describe("   all", tg["lag_first"])
    describe("   isolated", tg.loc[tg["isolated"], "lag_first"])
    describe("   with neighbours", tg.loc[~tg["isolated"], "lag_first"])
    for lo_ha, hi_ha in ((50, 100), (100, 500), (500, 1e9)):
        describe(f"   {lo_ha}–{hi_ha if hi_ha < 1e9 else ''} ha", tg.loc[tg["burned_area_ha"].between(lo_ha, hi_ha, inclusive="left"), "lag_first"])
    # hourly model input: the step of the first detection (max over four 15-min slots)
    lag_hourly = ((tg["first_det"].dt.floor("h") + pd.Timedelta(minutes=30)) - tg["start_utc"]).dt.total_seconds() / 3600
    describe("   same, from hourly steps (centre of the hour)", lag_hourly)
    describe("   not already burning", tg.loc[~tg["already_burning"], "lag_first"])
    print(f"   already burning (detection ≤ {R_PX} cells, 6 h – 30 min before the start): {tg['already_burning'].mean():.1%} "
          f"({tg['already_burning'].sum()}); isolated {tg.loc[tg['isolated'], 'already_burning'].mean():.1%}, "
          f"with neighbours {tg.loc[~tg['isolated'], 'already_burning'].mean():.1%}")
    print()

    # D: nucleus birth nearest to the ICNF point
    nu = pd.read_parquet(E.PROC / "seviri_nuclei.parquet")
    nu = nu[~nu["reappeared"]].sort_values("t_birth").reset_index(drop=True)
    tb = nu["t_birth"].values
    lag_nu = []
    for q in tg.itertuples():
        lo, hi = np.searchsorted(tb, np.array([(q.start_utc - pd.Timedelta(hours=6)).tz_convert(None),
                                               (q.start_utc + pd.Timedelta(hours=12)).tz_convert(None)], dtype="datetime64[ns]"))
        c = nu.iloc[lo:hi]
        d = E.haversine_km(q.lat_s, q.lon_s, c["lat_birth"].values, c["lon_birth"].values) if len(c) else np.array([])
        ok = d <= 5
        lag_nu.append((c["t_birth"].values[ok][np.argmin(d[ok])] - np.datetime64(q.start_utc.tz_convert(None))) / np.timedelta64(1, "h")
                      if ok.any() else np.nan)
    tg["lag_nucleus"] = lag_nu
    print("D  nearest nucleus birth (≤ 5 km, not reappeared; sensor frame) minus ICNF start")
    describe("   all", tg["lag_nucleus"])
    describe("   isolated", tg.loc[tg["isolated"], "lag_nucleus"])
    describe("   with neighbours", tg.loc[~tg["isolated"], "lag_nucleus"])
    print()

    # E: close pairs of targets in one period
    rows = []
    for p, g in tg.groupby("period"):
        if len(g) < 2:
            continue
        v = g[["i_f", "j_f"]].values
        d = np.hypot(*(v[:, None] - v[None]).transpose(2, 0, 1))
        a, b = np.nonzero(np.triu(d <= PAIR_PX, 1))
        for x, y in zip(a, b):
            rows.append(dict(period=p, d_px=d[x, y],
                             a=g["fire_id"].iat[x], b=g["fire_id"].iat[y], dt_h=abs((g["start_utc"].iat[x] - g["start_utc"].iat[y]).total_seconds()) / 3600))
    pairs = pd.DataFrame(rows)
    print(f"E  target pairs ≤ {PAIR_PX} cells apart in the same period: {len(pairs)} pairs in {pairs['period'].nunique()} periods "
          f"({len(set(pairs['a']) | set(pairs['b']))} of {len(tg)} targets)")
    dt = pairs["dt_h"]
    print(f"   start-time difference: median {dt.median():.2f} h, IQR {dt.quantile(.25):.2f} … {dt.quantile(.75):.2f} h; "
          f"= 0 h {np.mean(dt == 0):.0%}, < 1 h {np.mean(dt < 1):.0%}, ≥ 1 h {np.mean(dt >= 1):.0%}, ≥ 2 h {np.mean(dt >= 2):.0%}, "
          f"≥ 3 h {np.mean(dt >= 3):.0%}")
    lag_iqr = tg["lag_first"].quantile(.75) - tg["lag_first"].quantile(.25)
    print(f"   pairs separable by more than the IQR of the oracle lag ({lag_iqr:.2f} h): {np.mean(dt > lag_iqr):.0%}\n")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    ax = axes[0]
    ax.hist(t_win, bins=np.arange(0, 24.5, 0.5), color=E.BLUE)
    ax.axvspan(0, HISTORY_H, color=E.NEUTRAL, zorder=0); ax.axvspan(HISTORY_H + PERIOD_H, 24, color=E.NEUTRAL, zorder=0)
    ax.set_xlim(0, 24); ax.set_xlabel("ICNF start, hours since window start"); ax.set_ylabel("targets")
    ax.set_title("(a) start time in the 24 h window (grey = history / tail)")
    ax = axes[1]
    bins = np.arange(-1, 6.01, 0.25)
    ax.hist(tg["lag_first"].clip(-1, 6), bins=bins, color=E.BLUE, alpha=0.75, label="first detection ≤ 1.5 cells (oracle location)")
    ax.hist(tg["lag_nucleus"].clip(-1, 6), bins=bins, histtype="step", lw=1.6, color=E.ORANGE, label="nearest nucleus birth ≤ 5 km")
    ax.axvline(0, color=E.INK2, lw=0.8)
    ax.set_xlabel("detection time − ICNF start [h]"); ax.set_ylabel("targets"); ax.legend(fontsize=7.5)
    ax.set_title("(b) how well SEVIRI timing matches the ICNF start")
    ax = axes[2]
    ax.hist(dt.clip(0, 12), bins=np.arange(0, 12.01, 0.5), color=E.BLUE)
    ax.set_xlabel("|start time difference| [h], clipped at 12"); ax.set_ylabel("pairs")
    ax.set_title(f"(c) target pairs ≤ {PAIR_PX:.0f} cells apart in one period")
    fig.savefig(FIG / "time_target_analysis.png", dpi=150); plt.close(fig)
    print("figure: figures/time/time_target_analysis.png")


if __name__ == "__main__":
    main()
