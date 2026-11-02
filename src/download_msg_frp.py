"""Download MSG/SEVIRI FRP-PIXEL (LSA SAF, LSA-502) and keep only the Portugal area.

Source:  https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5/<year>/<month>/<day>/
         one full-disk HDF5 "ListProduct" file per 15-min slot (about 9 MB per day).
         Registered account needed: https://mokey.lsasvcs.ipma.pt/auth/signup
Outputs: data/raw/msg_frp_pt/<year>/<yyyymmdd>.parquet  one file per day, all product fields
         data/raw/msg_frp_pt/inventory.csv              per day: files, missing slots, satellite
         data/processed/seviri_frp_2020_2025.parquet    (--merge) same schema as seviri_frp.parquet

The full-disk files are read in memory and not stored (about 20 GB for 2020-2025). A finished day
is never downloaded again, so rerunning the same command resumes; a day with network errors is
left unwritten and retried on the next run.

Run from the repository root:
  .venv/bin/python src/download_msg_frp.py                                   # all of 2020-2025
  .venv/bin/python src/download_msg_frp.py --start 2019-07-20 --end 2019-07-22
  .venv/bin/python src/download_msg_frp.py --merge                           # no login needed
Login: env vars LSASAF_USER and LSASAF_PASSWORD, otherwise asked for at the start.
"""
import argparse
import getpass
import io
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DAYS_DIR = ROOT / "data" / "raw" / "msg_frp_pt"
INVENTORY = DAYS_DIR / "inventory.csv"
OUT = ROOT / "data" / "processed" / "seviri_frp_2020_2025.parquet"

BASE_URL = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5"
FIRST_DAY, LAST_DAY = date(2020, 1, 1), date(2025, 12, 31)

# Slightly wider than the 2004-2019 extract (lat 36.35-43.58, lon -9.39 to -6.00), whose exact
# limits are not documented. The -6.00 edge is a hard cut in that extract, so it is kept here.
LAT = (36.0, 44.0)
LON = (-10.0, -6.0)

SLOTS = [f"{h:02d}{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
WORKERS = 4
RETRIES = 5
TIMEOUT = 60
MAX_FAILED_DAYS = 5   # consecutive days with network errors before giving up


class LoginError(Exception):
    pass


def log(msg):
    print(f"  - {msg}", flush=True)


def fetch(session, url):
    """GET with retries. Returns the body, or None if the path does not exist (404)."""
    for attempt in range(RETRIES):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code in (401, 403):
                raise LoginError(f"HTTP {r.status_code}: login refused, check username and password")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            size = r.headers.get("Content-Length")
            if size is not None and "Content-Encoding" not in r.headers and int(size) != len(r.content):
                raise requests.ConnectionError(f"truncated: {len(r.content)} of {size} bytes")
            return r.content
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def list_day(session, base_url, day):
    """ListProduct file names of one day; empty if the day is not in the archive."""
    url = f"{base_url}/{day:%Y/%m/%d}/"
    body = fetch(session, url)
    if body is None:
        return url, []
    text = body.decode(errors="replace")
    names = sorted(set(re.findall(r'href="[^"]*/(HDF5_[^"/]*ListProduct[^"/]*_\d{12})"', text)))
    if not names and "_h5ai" not in text:   # e.g. a maintenance page: must not count as an empty day
        raise requests.ConnectionError(f"unexpected page instead of a listing: {url}")
    return url, names


def attr(h5, name):
    """Global attribute as text; the array-valued ones are padded with '-' entries."""
    v = h5.attrs.get(name, b"")
    v = v[0] if isinstance(v, np.ndarray) else v
    return v.decode() if isinstance(v, bytes) else str(v)


def read_detections(content, time_utc):
    """Detections inside the box from one file, with scale factors and missing values applied.

    Raises OSError/KeyError/IndexError if the file cannot be read as a FRP-PIXEL product.
    """
    with h5py.File(io.BytesIO(content), "r") as h5:
        info = {k: attr(h5, k) for k in ("SATELLITE", "PRODUCT_ALGORITHM_VERSION", "OVERALL_QUALITY_FLAG")}
        lat, lon = h5["LATITUDE"][()] / 100, h5["LONGITUDE"][()] / 100
        keep = (lat >= LAT[0]) & (lat <= LAT[1]) & (lon >= LON[0]) & (lon <= LON[1])
        df = pd.DataFrame(index=range(int(keep.sum())))
        df["time_utc"] = time_utc
        for name, ds in h5.items():
            raw = ds[()][keep]
            # FIRE_CONFIDENCE is stored as 1-100 although its attribute says scale 100; the
            # 2004-2019 extract also uses 1-100. ACQTIME and the image indices are plain integers.
            scale = 1.0 if name == "FIRE_CONFIDENCE" else float(ds.attrs["SCALING_FACTOR"])
            values = raw.astype("float32") / scale
            values[raw == ds.attrs["MISSING_VALUE"]] = np.nan
            df[name.lower()] = values
    return df, info


def process_file(session, url, name):
    """One slot -> (slot, detections, attributes); detections is None if the file is unusable."""
    slot = name[-12:]
    time_utc = pd.Timestamp(datetime.strptime(slot, "%Y%m%d%H%M"), tz="UTC")
    content = fetch(session, url + name)
    if content is None:
        return slot, None, {}
    try:
        return (slot, *read_detections(content, time_utc))
    except (OSError, KeyError, IndexError):
        return slot, None, {}


def append_inventory(row):
    new = not INVENTORY.exists()
    pd.DataFrame([row]).to_csv(INVENTORY, mode="a", header=new, index=False)


def download_day(sessions, pool, base_url, day):
    url, names = list_day(sessions[0], base_url, day)
    results = list(pool.map(lambda n: process_file(sessions[threading.get_ident()], url, n), names))

    frames = [df for _, df, _ in results if df is not None and len(df)]
    infos = [info for _, df, info in results if df is not None]
    listed = {name[-4:] for name in names}
    empty = pd.DataFrame({"time_utc": pd.Series(dtype="datetime64[ns, UTC]")})
    table = pd.concat(frames, ignore_index=True) if frames else empty

    append_inventory({
        "date": day.isoformat(),
        "files": len(names),
        "rows": len(table),
        "missing_slots": " ".join(s for s in SLOTS if s not in listed),
        "unreadable_slots": " ".join(slot[-4:] for slot, df, _ in results if df is None),
        "satellites": " ".join(sorted({i["SATELLITE"] for i in infos})),
        "versions": " ".join(sorted({i["PRODUCT_ALGORITHM_VERSION"] for i in infos})),
        "quality_flags": " ".join(sorted({i["OVERALL_QUALITY_FLAG"] for i in infos})),
    })

    path = day_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)   # a day file only ever appears complete
    return len(names), len(table)


def day_path(day):
    return DAYS_DIR / f"{day.year}" / f"{day:%Y%m%d}.parquet"


def dates(start, end):
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def make_session(user, password):
    s = requests.Session()
    s.auth = (user, password)
    return s


def download(start, end, base_url):
    todo = [d for d in dates(start, end) if not day_path(d).exists()]
    print(f"MSG FRP-PIXEL {start} -> {end}: {len(todo):,} days to download")
    if not todo:
        return
    user = os.environ.get("LSASAF_USER") or input("LSA SAF username: ")
    password = os.environ.get("LSASAF_PASSWORD") or getpass.getpass("LSA SAF password: ")

    DAYS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = {}   # one connection per thread, reused for all of its requests

    def init_worker():
        sessions[threading.get_ident()] = make_session(user, password)

    failed_in_a_row = 0
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS, initializer=init_worker) as pool:
        sessions[0] = make_session(user, password)
        for k, day in enumerate(todo, 1):
            try:
                n_files, n_rows = download_day(sessions, pool, base_url, day)
            except LoginError as e:
                raise SystemExit(f"Stopped: {e}")
            except requests.RequestException as e:
                failed_in_a_row += 1
                log(f"{day}: failed, retry by running again ({type(e).__name__})")
                if failed_in_a_row >= MAX_FAILED_DAYS:
                    raise SystemExit(f"Stopped: {MAX_FAILED_DAYS} days failed in a row. Rerun to resume.")
                continue
            failed_in_a_row = 0
            left = (time.time() - t0) / k * (len(todo) - k) / 3600
            log(f"{day}: {n_files} files, {n_rows} detections ({k:,}/{len(todo):,}, about {left:.1f} h left)")


def merge(start, end, out):
    """Daily files -> one table with the schema of seviri_frp.parquet (clean_data.py)."""
    print(f"Merge {start} -> {end}")
    days = dates(start, end)
    missing = [d for d in days if not day_path(d).exists()]
    if missing:
        log(f"WARNING: {len(missing):,} of {len(days):,} days not downloaded, first {missing[0]}")
    tables = [pd.read_parquet(day_path(d)) for d in days if day_path(d).exists()]
    tables = [t for t in tables if len(t)]
    if not tables:
        raise SystemExit("Nothing to merge.")
    d = pd.concat(tables, ignore_index=True)

    n = len(d)
    d = d.dropna(subset=["latitude", "longitude", "frp", "fire_confidence"])
    log(f"dropped {n - len(d):,} rows with missing values")
    df = pd.DataFrame({
        "time_utc": d["time_utc"].astype("datetime64[ns, UTC]"),
        "lat": d["latitude"].astype("float64").round(2),
        "lon": d["longitude"].astype("float64").round(2),
        "frp_mw": d["frp"].round(1).astype("float32"),
        "confidence": d["fire_confidence"].astype("int8"),
    })
    n = len(df)
    df = df.drop_duplicates()
    log(f"dropped {n - len(df):,} exact duplicate rows")
    assert not df.duplicated(["time_utc", "lat", "lon"]).any(), "conflicting duplicates"
    df = df.sort_values(["time_utc", "lat", "lon"]).reset_index(drop=True)
    log(f"clean rows: {len(df):,}, {df['time_utc'].nunique():,} timestamps, "
        f"{df['time_utc'].min()} -> {df['time_utc'].max()}")

    if INVENTORY.exists():
        inv = pd.read_csv(INVENTORY, keep_default_na=False).drop_duplicates("date", keep="last")
        inv = inv[inv["date"].between(start.isoformat(), end.isoformat())]
        log(f"inventory: {inv['files'].sum():,} of {len(inv) * len(SLOTS):,} slots present, "
            f"{(inv['unreadable_slots'] != '').sum()} days with unreadable files, "
            f"{(inv['files'] == 0).sum()} days absent from the archive")
        log(f"satellites {sorted(set(' '.join(inv['satellites']).split()))}, "
            f"algorithm {sorted(set(' '.join(inv['versions']).split()))}, "
            f"quality flags {sorted(set(' '.join(inv['quality_flags']).split()))}")
    print(df.groupby(df["time_utc"].dt.year).size().rename("rows").to_string())

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


def main():
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)   # today is still incomplete
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY)
    ap.add_argument("--end", type=date.fromisoformat, default=min(LAST_DAY, yesterday))
    ap.add_argument("--merge", action="store_true", help="only build the merged table from the daily files")
    ap.add_argument("--out", type=Path, default=OUT, help="merged table (with --merge)")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()
    if args.merge:
        merge(args.start, args.end, args.out)
    else:
        download(args.start, args.end, args.base_url.rstrip("/"))


if __name__ == "__main__":
    main()
