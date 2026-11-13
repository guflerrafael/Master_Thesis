"""Clean the raw ICNF and SEVIRI files into tidy parquet tables.

Inputs (read-only):  data/raw/icnf/Registos_Incendios_SGIF_*.xlsx   ICNF SGIF export, 2001-2025
                     data/raw/ignic_ICNF.xlsx                       ICNF, used for 1980-2000 only
                     data/raw/FRP_SEVIRI_PT_2004_2019.mat           SEVIRI extract 2004-2019
                     data/processed/seviri_frp_2020_2025.parquet    LSA SAF download (src/download_msg_frp.py --merge)
Outputs:             data/processed/icnf_fires.parquet
                     data/processed/seviri_frp.parquet              2004-2019 (the exploratory scripts read this)
                     data/processed/seviri_frp_2004_2025.parquet    2004-2019 + 2020-2025, input of the pipeline

The SGIF export holds every row of ignic_ICNF.xlsx from 2001 on (notes.md 8.2). With the same cleaning
rules the rows and fire_ids up to 2023 are the same as with the old file; the SGIF export adds 2024-2025
and the columns of SGIF_COLUMNS.

Run from the repository root:  .venv/bin/python src/clean_data.py
Every cleaning step and its row count is documented in notes.md.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# Mainland Portugal bounding box (with a small margin). Points outside are invalid.
PT_LAT = (36.8, 42.2)
PT_LON = (-9.6, -6.1)
SGIF_FIRST_YEAR = 2001            # the SGIF export starts in 2001; earlier rows come from ignic_ICNF.xlsx
FWI_COLUMNS = ["FWI", "ISI", "FFMC", "DMC", "DC", "BUI", "DSR"]   # -80 marks a missing value


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


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def read_old_icnf() -> pd.DataFrame:
    """ignic_ICNF.xlsx in the columns of the cleaned table (before duplicates are dropped)."""
    # The xlsx is the source: the csv export has timestamps shifted by -1 s (see notes.md).
    df = pd.read_excel(RAW / "ignic_ICNF.xlsx")
    log(f"ignic_ICNF.xlsx raw rows: {len(df):,}")
    return df.drop(columns="Unnamed: 0").rename(columns={
        "Data": "start_local",
        "DataFim": "end_local",
        "Dicofre": "municipality_code",
        "Distrito": "district",
        "Concelho": "municipality",
        "AA": "burned_area_ha",
    })


def read_sgif() -> pd.DataFrame:
    """The three SGIF workbooks in the columns of the cleaned table, plus the SGIF-only columns."""
    files = sorted((RAW / "icnf").glob("Registos_Incendios_SGIF_*.xlsx"))
    assert len(files) == 3, f"expected 3 SGIF workbooks, found {len(files)}"
    parts = []
    for f in files:
        d = pd.read_excel(f, sheet_name=f.stem.replace("Registos_Incendios_", ""))
        log(f"{f.name}: {len(d):,} rows, alert years {d['Ano'].min()}-{d['Ano'].max()}")
        parts.append(d)
    s = pd.concat(parts, ignore_index=True)
    # types differ between the workbooks (notes.md 8.1): end time is text in 2001-2010 (ISO format)
    end = pd.to_datetime(s["DataHora_Extincao"].astype(str).replace({"None": None, "NaT": None, "nan": None}), format="ISO8601")
    num = lambda c: pd.to_numeric(s[c], errors="coerce")
    parish = num("DTCCFR")
    # lat/lon against the PT-TM06/ETRS89 coordinates of the same row (a quality flag)
    lon_e, lat_e = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True).transform(
        num("X_ETRS89").values, num("Y_ETRS89").values)
    cause_known = s["CodCausa"].notna()
    out = pd.DataFrame({
        "start_local": pd.to_datetime(s["DataHoraAlerta"]),
        "end_local": end.dt.floor("min"),             # the old file stored minutes only
        "municipality_code": parish // 100,           # DTCCFR = INE parish code
        "district": s["Distrito"],
        "municipality": s["Concelho"],
        "burned_area_ha": s["AreaTotal_ha"].astype(float),
        "lat": s["Latitude"].astype(float),
        "lon": s["Longitude"].astype(float),
        "sgif_code": s["Codigo_SGIF"].astype(str),
        "anepc_code": num("Codigo_ANEPC").astype("Int64"),
        "parish_code": parish.astype("Int32"),
        "area_forest_ha": s["AreaPov_ha"].astype(float),
        "area_shrub_ha": s["AreaMato_ha"].astype(float),
        "area_agric_ha": s["AreaAgric_ha"].astype(float),
        "first_intervention_local": pd.to_datetime(s["DataHora_PrimeiraIntervencao"]),
        "cause_code": num("CodCausa").astype("Int32"),
        "cause_type": s["TipoCausa"],
        "cause_group": s["GrupoCausa"],
        "rekindle": (s["TipoCausa"] == "Reacendimento").astype("boolean").where(cause_known, pd.NA),
        "alert_source": s["FonteAlerta"].map(lambda v: None if pd.isna(v) else str(v)),
        **{c.lower(): num(c).where(num(c) >= 0) for c in FWI_COLUMNS},
        "coord_check_km": haversine_km(s["Latitude"].values, s["Longitude"].values, lat_e, lon_e),
    })
    log(f"SGIF rows: {len(out):,}; alert times {out['start_local'].min()} -> {out['start_local'].max()}")
    return out


SGIF_COLUMNS = ["sgif_code", "anepc_code", "parish_code", "area_forest_ha", "area_shrub_ha", "area_agric_ha",
                "first_intervention_local", "first_intervention_utc", "cause_code", "cause_type", "cause_group",
                "rekindle", "alert_source", "fwi", "isi", "ffmc", "dmc", "dc", "bui", "dsr", "coord_check_km"]


def clean_icnf() -> pd.DataFrame:
    print("ICNF")
    old = read_old_icnf()
    old = old[old["start_local"].isna() | (old["start_local"].dt.year < SGIF_FIRST_YEAR)]
    log(f"ignic_ICNF.xlsx rows kept (before {SGIF_FIRST_YEAR} or without start): {len(old):,}")
    sgif = read_sgif()
    assert sgif["start_local"].dt.year.min() >= SGIF_FIRST_YEAR
    df = pd.concat([old.assign(source="ignic_xlsx"), sgif.assign(source="sgif")], ignore_index=True)
    df = df.astype({c: t for c, t in sgif.dtypes.items() if str(t) in ("Int64", "Int32", "Int16", "boolean")})

    for col in ["district", "municipality"]:
        n_raw = df[col].nunique()
        df[col] = normalise_text(df[col])
        log(f"{col}: {n_raw} raw spellings -> {df[col].nunique()} after normalisation")

    n = len(df)
    df = df[df["start_local"].notna()]
    log(f"dropped {n - len(df)} rows without start time")

    # Exact duplicates over the columns of the old file (SGIF: double entries with different SGIF codes,
    # notes.md 8.2). The first row in file order is kept, as before.
    n = len(df)
    df = df.drop_duplicates(subset=["start_local", "end_local", "municipality_code", "district", "municipality",
                                    "burned_area_ha", "lat", "lon"])
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

    # Times are Portuguese local time (verified in src/timezone_check.py, notes.md 1.4; 2020-2025 in 7.2, 8.3).
    # Non-existent / ambiguous wall-clock times at the DST switches become NaT.
    for col in ["start", "end", "first_intervention"]:
        local = df[f"{col}_local"].dt.tz_localize("Europe/Lisbon", ambiguous="NaT", nonexistent="NaT")
        df[f"{col}_utc"] = local.dt.tz_convert("UTC")
        lost = (local.isna() & df[f"{col}_local"].notna()).sum()
        log(f"{col}_utc: {lost} rows NaT (time falls in a DST switch hour)")

    df = df.sort_values(["start_local", "municipality_code", "burned_area_ha"], kind="stable")
    df.insert(0, "fire_id", np.arange(len(df), dtype="int32"))
    missing_mun = df["municipality_code"].isna().sum()
    log(f"rows without municipality: {missing_mun} (municipality_code kept as missing)")
    df["municipality_code"] = df["municipality_code"].astype("Int16" if missing_mun else "int16")
    for col in ["district", "municipality", "cause_type", "cause_group", "source"]:
        df[col] = df[col].astype("category")
    cols = ["fire_id", "start_local", "end_local", "municipality_code", "district", "municipality", "burned_area_ha",
            "lat", "lon", "end_invalid", "duration_h", "coord_invalid", "start_utc", "end_utc", "source"] + SGIF_COLUMNS
    df = df[cols].reset_index(drop=True)

    sg = df["source"] == "sgif"
    log(f"clean rows: {len(df):,} ({df['lat'].notna().sum():,} with valid coordinates; {sg.sum():,} from SGIF, "
        f"{df.loc[sg, 'start_local'].min():%Y-%m-%d} -> {df.loc[sg, 'start_local'].max():%Y-%m-%d})")
    log(f"SGIF rows with a known cause: {df.loc[sg, 'cause_code'].notna().mean():.1%}; rekindles: {int(df['rekindle'].sum()):,}")
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


def merge_seviri(old: pd.DataFrame) -> pd.DataFrame:
    """The 2004-2019 extract and the LSA SAF download 2020-2025 in one table (notes.md 7.1-7.2: same grid,
    same time label, no new geo-referencing offset)."""
    print("SEVIRI 2004-2025")
    new = pd.read_parquet(OUT / "seviri_frp_2020_2025.parquet")
    assert (old.dtypes == new.dtypes).all(), "schemas differ"
    assert old["time_utc"].max() < new["time_utc"].min(), "the two sources overlap in time"
    df = pd.concat([old, new], ignore_index=True)
    assert not df.duplicated(["time_utc", "lat", "lon"]).any(), "conflicting duplicates"
    df = df.sort_values(["time_utc", "lat", "lon"]).reset_index(drop=True)
    log(f"rows: {len(old):,} + {len(new):,} = {len(df):,}, {df['time_utc'].min()} -> {df['time_utc'].max()}")
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean_icnf().to_parquet(OUT / "icnf_fires.parquet", index=False)
    sev = clean_seviri()
    sev.to_parquet(OUT / "seviri_frp.parquet", index=False)
    merge_seviri(sev).to_parquet(OUT / "seviri_frp_2004_2025.parquet", index=False)
    print(f"written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
