"""Data checks for the extension to 2020-2025 (notes.md 7-8): SEVIRI alignment, ICNF SGIF export,
time zone, geo-referencing offset, and regression of the rebuilt files against the build before the
extension. Data only: no model and no baseline is scored here.

  1  SEVIRI: pixel identity old extract vs LSA SAF download; native grid vs the product's own
     line/pixel indices (raw day files, if present); rows per day on the overlap days of July 2019
     against the old extract, also with the extract's time labels shifted by +-1..4 slots
  2  ICNF: icnf_fires.parquet against the version of commit REF (built from ignic_ICNF.xlsx)
  3  ICNF time zone: winter - summer median lag of the first SEVIRI detection (notes.md 1.4)
  4  position: ICNF true position - FRP-weighted centroid of the first detection slot within 2
     cells in [start - 1 h, start + 6 h], per block of years and per 3-h block of the day, each
     block compared with the same block of 2018-2023 (rule of notes.md 7.5 as corrected in 8.3)
  5  regression: detections, static maps, fires, labels and periods of 2004-2019 against commit REF
  6  hard-negative periods without a detection on a mainland cell (train.CropSet, notes.md 7.4)

Run from the repository root (after clean_data, cluster_hotspots, build_dataset):
  .venv/bin/python src/check_new_years.py > outputs/new_years_check.txt
"""
import glob
import io
import subprocess

import numpy as np
import pandas as pd

import eda_figures as E
import grid as G
from dataset import FireData

REF = "5a0f39d"            # last commit before the extension (2004-2019 extract, ignic_ICNF.xlsx)
MODEL = E.PROC / "model"
RAW_DAYS = E.ROOT / "data" / "raw" / "msg_frp_pt"
BLOCKS = {"2004 - Nov 2017": ("2004-01-01", "2017-12-01"), "2018-2019": ("2018-03-01", "2020-01-01"),
          "2020-2023": ("2020-01-01", "2024-01-01"), "2024-2025": ("2024-01-01", "2026-01-01")}
RULE_PX = 0.25             # half the 2017 shift
rng = np.random.default_rng(0)


def at_ref(path):
    """A tracked parquet file as of commit REF."""
    return pd.read_parquet(io.BytesIO(subprocess.run(["git", "show", f"{REF}:{path}"], cwd=E.ROOT, capture_output=True,
                                                     check=True).stdout))


def boot_median(v, n=4000):
    v = np.asarray(v)
    return np.percentile(np.median(rng.choice(v, (n, len(v))), axis=1), [2.5, 97.5])


def boot_diff(a, b, n=4000):
    a, b = np.asarray(a), np.asarray(b)
    d = np.median(rng.choice(b, (n, len(b))), 1) - np.median(rng.choice(a, (n, len(a))), 1)
    return np.median(b) - np.median(a), np.percentile(d, [2.5, 97.5])


def seviri():
    print("1. SEVIRI")
    old = pd.read_parquet(E.PROC / "seviri_frp.parquet")
    new = pd.read_parquet(E.PROC / "seviri_frp_2020_2025.parquet")
    def pixels(d):
        p = d[["lat", "lon"]].drop_duplicates()
        r, c = G.native_rc(p["lat"].values, p["lon"].values)
        return p.assign(row=np.round(r).astype(int), col=np.round(c).astype(int))
    po, pn = pixels(old), pixels(new)
    both = pn.merge(po, on=["row", "col"], suffixes=("_n", "_o"))
    union = pd.concat([po, pn]).drop_duplicates(["lat", "lon"])
    print(f"  native pixels active in both sources: {len(both):,}; lat/lon differing: "
          f"{int(((both.lat_n != both.lat_o) | (both.lon_n != both.lon_o)).sum())}; collisions in the union: "
          f"{len(union) - len(union[['row', 'col']].drop_duplicates())}; new pixels: {len(pn) - len(both):,}")
    files = sorted(glob.glob(str(RAW_DAYS / "20*" / "*.parquet")))
    if files:
        raw = pd.concat([t for t in map(pd.read_parquet, files) if len(t)])     # days without detections have no columns
        r, c = G.native_rc(raw["latitude"].astype(float).round(2).values, raw["longitude"].astype(float).round(2).values)
        ok = (raw["abs_line"].values == 1857 - np.round(r)) & (raw["abs_pixel"].values == 1857 + np.round(c))
        print(f"  ABS_LINE = 1857 - row and ABS_PIXEL = 1857 + col: {int(ok.sum()):,} of {len(raw):,} rows")
    else:
        print("  raw day files not found: line/pixel check skipped")
    inv = pd.read_csv(RAW_DAYS / "inventory.csv", keep_default_na=False).drop_duplicates("date", keep="last")
    overlap = inv[inv["date"] < "2020-01-01"]
    t = old["time_utc"]
    for s in range(-4, 5):
        c = (t + pd.Timedelta(minutes=15 * s)).dt.strftime("%Y-%m-%d").value_counts()
        match = all(c.get(d, 0) == n for d, n in zip(overlap["date"], overlap["rows"]))
        if s == 0 or match:
            print(f"  overlap days {list(overlap['date'])}: LSA SAF rows {list(overlap['rows'])}, extract shifted {s:+d} slots "
                  f"{[int(c.get(d, 0)) for d in overlap['date']]} -> {'match' if match else 'no match'}")


def icnf():
    print("\n2. ICNF: icnf_fires.parquet vs commit", REF)
    o = at_ref("data/processed/icnf_fires.parquet").set_index("fire_id")
    n = pd.read_parquet(E.PROC / "icnf_fires.parquet").set_index("fire_id")
    print(f"  rows: {len(o):,} at {REF}, {len(n):,} now ({int((n['start_local'].dt.year >= 2024).sum()):,} in 2024-2025); "
          f"sources {n['source'].value_counts().to_dict()}")
    nn = n.loc[o.index]
    diff = {}
    for c in o.columns:
        a, b = o[c], nn[c]
        if str(a.dtype) == "category":
            a, b = a.astype(str), b.astype(str)
        eq = (a == b).fillna(False) | (a.isna() & b.isna())
        if a.dtype.kind == "f":
            eq |= (a - b.astype(float)).abs() < 1e-9
        diff[c] = int((~eq).sum())
    print(f"  fire_id 0-{o.index.max():,}: rows differing per column {diff}")
    end = o.index[~((o["end_local"] == nn["end_local"]) | (o["end_local"].isna() & nn["end_local"].isna()))]
    print(f"  end-time changes: {list(end)} ({o.loc[end, 'burned_area_ha'].round(2).tolist()} ha)")
    s = n[n["source"] == "sgif"]
    by = s.groupby(s["start_local"].dt.year)
    cov = pd.DataFrame({"rows": by.size(), "cause known": by["cause_code"].apply(lambda x: x.notna().mean()),
                        "rekindles": by["rekindle"].sum(), "first intervention": by["first_intervention_local"].apply(lambda x: x.notna().mean()),
                        "fwi": by["fwi"].apply(lambda x: x.notna().mean()), "alert source": by["alert_source"].apply(lambda x: x.notna().mean())})
    print("  SGIF field coverage by year:"); print(cov.round(3).to_string())
    return n


def first_detections(f, sev):
    """Per fire: offset of the ICNF true position from the first detection slot within 2 cells, lag, hour."""
    t = sev["time_utc"].dt.tz_convert(None).values
    rows = []
    for x in f.itertuples():
        s0 = x.start_utc.tz_convert(None)
        a, b = np.searchsorted(t, [(s0 - pd.Timedelta("1h")).to_datetime64(), (s0 + pd.Timedelta("6h")).to_datetime64()])
        d = sev.iloc[a:b]
        d = d[(d["i"] - x.i_f) ** 2 + (d["j"] - x.j_f) ** 2 <= 4]
        if d.empty:
            continue
        d = d[d["time_utc"] == d["time_utc"].min()]
        w = d["frp_mw"]
        rows.append((x.start_utc, x.i_f - np.average(d["i"], weights=w), x.j_f - np.average(d["j"], weights=w),
                     (d["time_utc"].iloc[0] - x.start_utc).total_seconds() / 3600, d["time_utc"].iloc[0].hour))
    return pd.DataFrame(rows, columns=["start", "di", "dj", "lag_h", "hour"])


def block_of(t):
    out = pd.Series(np.nan, index=t.index, dtype=object)
    for k, (a, b) in BLOCKS.items():
        out[(t >= pd.Timestamp(a, tz="UTC")) & (t < pd.Timestamp(b, tz="UTC"))] = k
    return out


def timezone(icnf, sev):
    print("\n3. ICNF time zone (fires >= 10 ha, first detection <= 5 km in [start - 6 h, start + 24 h], no detection "
          "<= 10 km in the 48 h before; recorded start read as UTC)")
    f = icnf[icnf["lat"].notna() & (icnf["burned_area_ha"] >= 10) & icnf["start_utc"].notna()
             & icnf["start_local"].dt.year.between(2004, 2025)].reset_index(drop=True)
    naive = f["start_local"].dt.tz_localize("UTC").values.astype("datetime64[ns]").astype("int64")
    dst = (f["start_utc"].dt.tz_convert("Europe/Lisbon").map(lambda x: x.dst()) > pd.Timedelta(0)).values
    T = sev["time_utc"].values.astype("datetime64[ns]").astype("int64")
    H = 3600 * 10**9
    lat, lon = sev["lat"].values, sev["lon"].values
    i0, i1, b0 = np.searchsorted(T, naive - 6 * H), np.searchsorted(T, naive + 24 * H), np.searchsorted(T, naive - 54 * H)
    lag = np.full(len(f), np.nan)
    for k in range(len(f)):
        if i1[k] <= i0[k]:
            continue
        if b0[k] < i0[k] and (E.haversine_km(f["lat"].iat[k], f["lon"].iat[k], lat[b0[k]:i0[k]], lon[b0[k]:i0[k]]) <= 10).any():
            continue
        near = np.nonzero(E.haversine_km(f["lat"].iat[k], f["lon"].iat[k], lat[i0[k]:i1[k]], lon[i0[k]:i1[k]]) <= 5)[0]
        if len(near):
            lag[k] = (T[i0[k] + near[0]] - naive[k]) / H
    f = f.assign(lag=lag, dst=dst, block=pd.cut(f["start_local"].dt.year, [2003, 2019, 2023, 2025], labels=["2004-2019", "2020-2023", "2024-2025"]))
    for blk, g in f[f["lag"].notna()].groupby("block", observed=True):
        a, b = g.loc[g["dst"], "lag"], g.loc[~g["dst"], "lag"]
        d, ci = boot_diff(a, b)
        print(f"  {blk}: summer n {len(a):,} median {a.median():+.2f} h, winter n {len(b):,} median {b.median():+.2f} h; "
              f"winter - summer {d:+.2f} h [{ci[0]:+.2f}, {ci[1]:+.2f}] (local time: about +1 h)")


def position(data, sev):
    print("\n4. Position of the ICNF ignition relative to the first detection (native px; +di = ICNF south, +dj = ICNF east)")
    tg = data.fires[data.fires["is_target"]]
    off = first_detections(tg, sev)
    off["block"] = block_of(off["start"])
    off["hb"] = off["hour"] // 3 * 3
    for blk, g in off.groupby("block", sort=False):
        ci, cj = boot_median(g["di"]), boot_median(g["dj"])
        print(f"  {blk:16s} n {len(g):5,}  di {g['di'].median():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  "
              f"dj {g['dj'].median():+.2f} [{cj[0]:+.2f}, {cj[1]:+.2f}]  lag {g['lag_h'].median():+.2f} h")
    ref = off[off["block"].isin(["2018-2019", "2020-2023"])]
    new = off[off["block"] == "2024-2025"]
    for c in ("di", "dj", "lag_h"):
        d, ci = boot_diff(off.loc[off["block"] == "2020-2023", c], new[c])
        print(f"  2024-2025 minus 2020-2023, {c}: {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    print(f"  by 3-h block of the first detection, 2024-2025 minus the same block of 2018-2023 (rule: |diff| > {RULE_PX} px "
          "with an interval excluding 0)")
    flags = 0
    for h, g in new.groupby("hb"):
        r = ref[ref["hb"] == h]
        line = f"    {h:02d}-{h + 3:02d} UTC n {len(g):3d} vs {len(r):3d}:"
        for c in ("di", "dj"):
            d, ci = boot_diff(r[c], g[c])
            hit = abs(d) > RULE_PX and (ci[0] > 0 or ci[1] < 0)
            flags += hit
            line += f"  {c} {d:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]{' <- RULE' if hit else ''}"
        print(line)
    print(f"  rule exceeded: {flags} time(s)")


def regression(data):
    print("\n5. Regression of the rebuilt files against commit", REF, "(2004-2019)")
    d0 = at_ref("data/processed/model/detections.parquet")
    a = data.det[data.det["time_utc"] < pd.Timestamp("2020-01-01", tz="UTC")].sort_values(["time_utc", "lat", "lon"]).reset_index(drop=True)
    cols = ["time_utc", "lat", "lon", "frp_mw", "confidence", "event_id", "nucleus_id", "persistent", "in_pt", "i", "j", "slot"]
    print(f"  detections: {len(a):,} vs {len(d0):,}; identical columns: "
          f"{sum(bool((a[c].values == d0[c].values).all()) for c in cols)} of {len(cols)}")
    s0 = dict(np.load(io.BytesIO(subprocess.run(["git", "show", f"{REF}:data/processed/model/static.npz"], cwd=E.ROOT,
                                                capture_output=True, check=True).stdout)))
    print(f"  static maps identical: {all(bool((s0[k] == data.static[k]).all()) for k in s0)}")
    f0 = at_ref("data/processed/model/fires.parquet")
    m = f0.merge(data.fires, on="fire_id", how="left", suffixes=("_0", "_1"))
    same = {c: int((~((m[c + "_0"] == m[c + "_1"]) | (m[c + "_0"].isna() & m[c + "_1"].isna()))).sum())
            for c in ("start_utc", "burned_area_ha", "seen", "is_target", "isolated", "period")}
    print(f"  fires: {int(m['year_1'].notna().sum()):,} of {len(f0):,} found; rows differing {same}; "
          f"largest position difference {np.nanmax(np.abs(m['j_f_0'] - m['j_f_1'])):.1e} px")
    l0 = at_ref("data/processed/model/labels.parquet")
    k = ["period", "fire_id", "role"]
    mm = l0.merge(data.labels, on=k, how="left", suffixes=("_0", "_1"))
    print(f"  labels: {int(mm['i_f_1'].notna().sum()):,} of {len(l0):,} found; largest position difference "
          f"{np.nanmax(np.abs(np.r_[mm['i_f_0'] - mm['i_f_1'], mm['j_f_0'] - mm['j_f_1']])):.1e} px")
    p0 = at_ref("data/processed/model/periods.parquet")
    pp = p0.merge(data.periods, on="period", how="left", suffixes=("_0", "_1"))
    print(f"  periods: {int(pp['n_det_1'].notna().sum()):,} of {len(p0):,} found; counts differing "
          f"{ {c: int((pp[c + '_0'] != pp[c + '_1']).sum()) for c in ('n_det', 'n_det_pt', 'n_targets', 'n_ignore')} }")


def hard_negatives(data):
    print("\n6. Hard-negative periods without a detection on a mainland (pt) cell")
    neg = data.periods[(data.periods["n_targets"] == 0) & (data.periods["n_det_pt"] > 0)]
    bad = []
    for p in neg.itertuples():
        s0 = data.first_slot(p.period)
        a, b = np.searchsorted(data._slot, [s0, s0 + 96])
        if not data.static["pt"][data._i[a:b], data._j[a:b]].any():
            bad.append(f"{p.period:%Y-%m-%d %H} UTC ({p.split})")
    print(f"  {len(bad)} of {len(neg):,}: {bad} (excluded in train.CropSet)")


def main():
    seviri()
    icnf_all = icnf()
    data = FireData()
    sev = data.det.sort_values("time_utc").reset_index(drop=True)
    timezone(icnf_all, sev)
    position(data, sev)
    regression(data)
    hard_negatives(data)


if __name__ == "__main__":
    main()
