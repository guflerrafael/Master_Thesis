"""Clean the raw ICNF and SEVIRI files into tidy parquet tables.

Inputs (read-only):  data/raw/ignic_ICNF.xlsx, data/raw/FRP_SEVIRI_PT_2004_2019.mat
Outputs:             data/processed/icnf_fires.parquet, data/processed/seviri_frp.parquet

Run from the repository root:  .venv/bin/python src/clean_data.py
Every cleaning step and its row count is documented in notes.md.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Mainland Portugal bounding box (with a small margin). Points outside are invalid.
PT_LAT = (36.8, 42.2)
PT_LON = (-9.6, -6.1)


def log(msg):
    print(f"  - {msg}")


def normalise_text(s: pd.Series) -> pd.Series:
    """Upper case, no accents, trimmed: 'Setúbal' / 'SETUBAL' -> 'SETUBAL'."""
    def f(x):
        if not isinstance(x, str):
            return x
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
        return " ".join(x.upper().split())
    return s.map(f)


def clean_icnf() -> pd.DataFrame:
    print("ICNF")
    # The xlsx is the source: the csv export has timestamps shifted by -1 s (see notes.md).
    df = pd.read_excel(RAW / "ignic_ICNF.xlsx")
    log(f"raw rows: {len(df):,}")

    df = df.drop(columns="Unnamed: 0").rename(columns={
        "Data": "start_local",
        "DataFim": "end_local",
        "Dicofre": "municipality_code",
        "Distrito": "district",
        "Concelho": "municipality",
        "AA": "burned_area_ha",
    })

    for col in ["district", "municipality"]:
        n_raw = df[col].nunique()
        df[col] = normalise_text(df[col])
        log(f"{col}: {n_raw} raw spellings -> {df[col].nunique()} after normalisation")

    n = len(df)
    df = df[df["start_local"].notna()]
    log(f"dropped {n - len(df)} rows without start time")

    n = len(df)
    df = df.drop_duplicates()
    log(f"dropped {n - len(df):,} exact duplicate rows")

    # End time: missing, 1900 sentinel, or before the start -> unusable, set to NaT.
    end_invalid = (
        df["end_local"].isna()
        | (df["end_local"].dt.year < 1980)
        | (df["end_local"] < df["start_local"])
    )
    log(f"end_local set to NaT for {end_invalid.sum():,} rows "
        f"(missing: {df['end_local'].isna().sum()}, "
        f"sentinel <1980: {(df['end_local'].dt.year < 1980).sum()}, "
        f"end < start: {(df['end_local'] < df['start_local']).sum():,})")
    df["end_invalid"] = end_invalid
    df.loc[end_invalid, "end_local"] = pd.NaT
    df["duration_h"] = (df["end_local"] - df["start_local"]).dt.total_seconds() / 3600

    # Coordinates: outside mainland Portugal or exactly zero -> unusable, set to NaN.
    has = df["lat"].notna() & df["lon"].notna()
    inside = df["lat"].between(*PT_LAT) & df["lon"].between(*PT_LON)
    coord_invalid = has & ~inside
    log(f"coordinates set to NaN for {coord_invalid.sum()} rows outside mainland bbox")
    df["coord_invalid"] = coord_invalid
    df.loc[coord_invalid, ["lat", "lon"]] = np.nan

    # Times are Portuguese local time (verified in src/timezone_check.py, notes.md 1.4).
    # Non-existent / ambiguous wall-clock times at the DST switches become NaT.
    for col in ["start", "end"]:
        local = df[f"{col}_local"].dt.tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT")
        df[f"{col}_utc"] = local.dt.tz_convert("UTC")
        lost = (local.isna() & df[f"{col}_local"].notna()).sum()
        log(f"{col}_utc: {lost} rows NaT (time falls in a DST switch hour)")

    df = df.sort_values(["start_local", "municipality_code", "burned_area_ha"], kind="stable")
    df.insert(0, "fire_id", np.arange(len(df), dtype="int32"))
    df["municipality_code"] = df["municipality_code"].astype("int16")
    df["district"] = df["district"].astype("category")
    df["municipality"] = df["municipality"].astype("category")
    df = df.reset_index(drop=True)

    log(f"clean rows: {len(df):,} ({df['lat'].notna().sum():,} with valid coordinates)")
    return df


def clean_seviri() -> pd.DataFrame:
    print("SEVIRI")
    m = loadmat(RAW / "FRP_SEVIRI_PT_2004_2019.mat")
    df = pd.DataFrame({
        # Assumed UTC (standard for SEVIRI products; not documented in the file).
        "time_utc": pd.to_datetime(m["TIME"].ravel().astype(str)).tz_localize("UTC"),
        # Stored with 2-decimal precision; rounding removes float noise (43.579998 -> 43.58).
        "lat": m["LATITUDE"].ravel().round(2),
        "lon": m["LONGITUDE"].ravel().round(2),
        "frp_mw": m["FRP"].ravel().astype("float32"),
        "confidence": m["FIRE_CONFIDENCE"].ravel().astype("int8"),
    })
    log(f"raw rows: {len(df):,}")

    n = len(df)
    df = df.drop_duplicates()
    log(f"dropped {n - len(df):,} exact duplicate rows")
    assert not df.duplicated(["time_utc", "lat", "lon"]).any(), "conflicting duplicates"

    df = df.sort_values(["time_utc", "lat", "lon"]).reset_index(drop=True)
    log(f"clean rows: {len(df):,}, {df['time_utc'].nunique():,} timestamps, "
        f"{df['time_utc'].min()} -> {df['time_utc'].max()}")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean_icnf().to_parquet(OUT / "icnf_fires.parquet", index=False)
    clean_seviri().to_parquet(OUT / "seviri_frp.parquet", index=False)
    print(f"written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
