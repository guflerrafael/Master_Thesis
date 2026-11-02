"""Download the MSG/SEVIRI FRP-PIXEL Quality Product (LSA SAF) and keep only the model grid.

Source:  https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5/<year>/<month>/<day>/
         one full-disk 3712 x 3712 file per 15-min slot (about 60 MB per day), same login as
         src/download_msg_frp.py. Flag coding: product user manual, Table 7.
Outputs: data/raw/msg_quality_pt/<year>/<yyyymmdd>.npz  flags (96, CROP_H, CROP_W) uint8
         data/raw/msg_quality_pt/inventory.csv          per day: files, missing slots, satellite

Image cell (i, j) of src/grid.py is flags[slot, i + MARGIN, j + MARGIN]. Slots without a readable
file hold MISSING. The full-disk files are read in memory and not stored. A finished day is never
downloaded again, so rerunning the same command resumes.

Before any download the array position is measured: the detections of a ListProduct file must
carry flag 1 or 2 in the Quality file of the same slot. The run stops if they do not line up.

Run from the repository root:
  .venv/bin/python src/download_msg_quality.py --probe-only      # only the position check
  .venv/bin/python src/download_msg_quality.py                   # validation years 2015-2016
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
from datetime import date, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

import grid as G
from download_msg_frp import BASE_URL, MAX_FAILED_DAYS, SLOTS, WORKERS, LoginError, attr, dates, fetch, log, make_session

ROOT = Path(__file__).resolve().parents[1]
DAYS_DIR = ROOT / "data" / "raw" / "msg_quality_pt"
INVENTORY = DAYS_DIR / "inventory.csv"
FIRST_DAY, LAST_DAY = date(2015, 1, 1), date(2016, 12, 31)

DISK = 3712
MARGIN = 4                       # extra cells on each side of the model grid
CROP_H, CROP_W = G.H + 2 * MARGIN, G.W + 2 * MARGIN
MISSING = 253                    # not a product flag: no readable file for this slot
LINE_I0 = 1857 - G.ROW_MAX       # ABS_LINE of image row i = 0 (grid.py: ABS_LINE + row = 1857)
PIXEL_J0 = 1857 + G.COL_MIN      # ABS_PIXEL of image column j = 0 (ABS_PIXEL - col = 1857)
PROBE_SLOTS = ("1200", "1215", "1230", "1300", "1100", "1400")
MIN_AGREEMENT = 0.95


def quality_dataset(h5):
    """The 2D flag array; its name differs within the manual (QUALITYFLAG, QUALITY FLAG)."""
    names = [k for k, ds in h5.items() if ds.shape == (DISK, DISK)]
    if len(names) != 1:
        raise KeyError(f"expected one {DISK} x {DISK} dataset, found {list(h5)}")
    return h5[names[0]]


def list_day(session, base_url, day):
    """Quality file names of one day; empty if the day is not in the archive."""
    url = f"{base_url}/{day:%Y/%m/%d}/"
    body = fetch(session, url)
    if body is None:
        return url, []
    text = body.decode(errors="replace")
    names = sorted(set(re.findall(r'href="[^"]*/(HDF5_[^"/]*QualityProduct[^"/]*_\d{12})"', text)))
    if not names and "_h5ai" not in text:   # e.g. a maintenance page: must not count as an empty day
        raise requests.ConnectionError(f"unexpected page instead of a listing: {url}")
    return url, names


def measure_position(quality, lines, pixels):
    """Offset of the detections (ABS_LINE, ABS_PIXEL) in the flag array -> ((dl, dp), agreement,
    best agreement over the flipped arrays). flag_of(line, pixel) = quality[line + dl, pixel + dp]."""
    fire = (quality == 1) | (quality == 2)
    agree = {(dl, dp): fire[lines + dl, pixels + dp].mean() for dl in range(-2, 3) for dp in range(-2, 3)}
    best = max(agree, key=agree.get)
    flipped = [fire[::a, ::b] for a, b in ((-1, 1), (1, -1), (-1, -1))]
    other = max(f[lines + dl, pixels + dp].mean() for f in flipped for dl in range(-2, 3) for dp in range(-2, 3))
    return best, float(agree[best]), float(other)


def probe(session, base_url, day):
    """Measure the array position on one slot of `day` with detections on the full disk."""
    url = f"{base_url}/{day:%Y/%m/%d}/"
    for slot in PROBE_SLOTS:
        stamp = f"{day:%Y%m%d}{slot}"
        q = fetch(session, f"{url}HDF5_LSASAF_MSG_FRP-PIXEL-QualityProduct_MSG-Disk_{stamp}")
        l = fetch(session, f"{url}HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_MSG-Disk_{stamp}")
        if q is None or l is None:
            continue
        with h5py.File(io.BytesIO(l), "r") as h5:
            lines, pixels = h5["ABS_LINE"][()].astype(int), h5["ABS_PIXEL"][()].astype(int)
        if not len(lines):
            continue
        with h5py.File(io.BytesIO(q), "r") as h5:
            flags = quality_dataset(h5)[()]
            info = {k: attr(h5, k) for k in ("SATELLITE", "PRODUCT_ALGORITHM_VERSION", "CLOUD_COVERAGE")}
        (dl, dp), agreement, flipped = measure_position(flags, lines, pixels)
        crop = flags[LINE_I0 + dl:LINE_I0 + dl + G.H, PIXEL_J0 + dp:PIXEL_J0 + dp + G.W]
        values, counts = np.unique(crop, return_counts=True)
        log(f"{stamp}: {len(lines):,} detections, {agreement:.1%} have flag 1/2 at offset line {dl:+d}, "
            f"pixel {dp:+d} (flipped arrays at most {flipped:.1%}); {info}")
        log("flags on the model grid: " + ", ".join(f"{v}: {c / crop.size:.1%}" for v, c in zip(values, counts)))
        return (dl, dp), agreement
    raise SystemExit(f"Stopped: no slot of {day} with both files and detections to check the array position.")


def read_crop(content, dl, dp):
    with h5py.File(io.BytesIO(content), "r") as h5:
        info = {k: attr(h5, k) for k in ("SATELLITE", "PRODUCT_ALGORITHM_VERSION", "CLOUD_COVERAGE")}
        line, pixel = LINE_I0 + dl - MARGIN, PIXEL_J0 + dp - MARGIN
        return quality_dataset(h5)[line:line + CROP_H, pixel:pixel + CROP_W].astype(np.uint8), info


def process_file(session, url, name, dl, dp):
    """One slot -> (slot HHMM, crop, attributes); crop is None if the file is unusable."""
    content = fetch(session, url + name)
    if content is None:
        return name[-4:], None, {}
    try:
        return (name[-4:], *read_crop(content, dl, dp))
    except (OSError, KeyError, IndexError, ValueError):
        return name[-4:], None, {}


def day_path(day):
    return DAYS_DIR / f"{day.year}" / f"{day:%Y%m%d}.npz"


def download_day(sessions, pool, base_url, day, offset):
    url, names = list_day(sessions[0], base_url, day)
    results = list(pool.map(lambda n: process_file(sessions[threading.get_ident()], url, n, *offset), names))

    flags = np.full((len(SLOTS), CROP_H, CROP_W), MISSING, np.uint8)
    for slot, crop, _ in results:
        if crop is not None:
            flags[SLOTS.index(slot)] = crop
    infos = [info for _, crop, info in results if crop is not None]
    listed = {name[-4:] for name in names}
    row = {
        "date": day.isoformat(),
        "files": len(names),
        "missing_slots": " ".join(s for s in SLOTS if s not in listed),
        "unreadable_slots": " ".join(slot for slot, crop, _ in results if crop is None),
        "satellites": " ".join(sorted({i["SATELLITE"] for i in infos})),
        "versions": " ".join(sorted({i["PRODUCT_ALGORITHM_VERSION"] for i in infos})),
        "cloud_mask": " ".join(sorted({i["CLOUD_COVERAGE"] for i in infos})),
    }
    pd.DataFrame([row]).to_csv(INVENTORY, mode="a", header=not INVENTORY.exists(), index=False)

    path = day_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, flags=flags)
    tmp.replace(path)   # a day file only ever appears complete
    return len(names)


def download(start, end, base_url, probe_only):
    todo = [d for d in dates(start, end) if not day_path(d).exists()]
    print(f"MSG FRP-PIXEL Quality {start} -> {end}: {len(todo):,} days to download")
    if not todo and not probe_only:
        return
    user = os.environ.get("LSASAF_USER") or input("LSA SAF username: ")
    password = os.environ.get("LSASAF_PASSWORD") or getpass.getpass("LSA SAF password: ")

    try:
        offsets = [probe(make_session(user, password), base_url, d) for d in (start, end)]
    except LoginError as e:
        raise SystemExit(f"Stopped: {e}")
    if offsets[0][0] != offsets[1][0] or min(a for _, a in offsets) < MIN_AGREEMENT:
        raise SystemExit(f"Stopped: Quality and List products do not line up ({offsets}); nothing downloaded.")
    offset = offsets[0][0]
    if probe_only:
        print(f"Array position confirmed: offset line {offset[0]:+d}, pixel {offset[1]:+d}")
        return

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
                n_files = download_day(sessions, pool, base_url, day, offset)
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
            log(f"{day}: {n_files} files ({k:,}/{len(todo):,}, about {left:.1f} h left)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=FIRST_DAY)
    ap.add_argument("--end", type=date.fromisoformat, default=LAST_DAY)
    ap.add_argument("--probe-only", action="store_true", help="only check the array position, download nothing")
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()
    download(args.start, args.end, args.base_url.rstrip("/"), args.probe_only)


if __name__ == "__main__":
    main()
