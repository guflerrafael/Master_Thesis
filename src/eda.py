"""Exploratory data analysis (EDA) of the raw and the cleaned datasets.

The script generates profiling reports for both the ICNF wildfire data and the SEVIRI FRP data,
before and after cleaning, which can be used to understand the distributions, correlations,
and potential issues in the datasets, and to check the effect of the cleaning.

Inputs (read-only):  data/raw/ignic_ICNF.xlsx, data/raw/FRP_SEVIRI_PT_2004_2019.mat,
                     data/processed/icnf_fires.parquet, data/processed/seviri_frp.parquet
Outputs:             outputs/profiling/{raw,processed}/{icnf,seviri}_eda_report.html

Run from the repository root:  .venv/bin/python src/eda.py
"""

from pathlib import Path

from data_profiling import ProfileReport
import pandas as pd
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "profiling"


def load_raw_icnf() -> pd.DataFrame:
    # As delivered, including the leftover index column `Unnamed: 0`.
    return pd.read_excel(RAW / "ignic_ICNF.xlsx")


def load_raw_seviri() -> pd.DataFrame:
    # As delivered; only flattened into a table and TIME parsed so it is profiled as a date.
    m = loadmat(RAW / "FRP_SEVIRI_PT_2004_2019.mat")
    return pd.DataFrame({
        "TIME": pd.to_datetime(m["TIME"].ravel().astype(str)),
        "LATITUDE": m["LATITUDE"].ravel(),
        "LONGITUDE": m["LONGITUDE"].ravel(),
        "FRP": m["FRP"].ravel(),
        "FIRE_CONFIDENCE": m["FIRE_CONFIDENCE"].ravel(),
    })


def main():
    datasets = {
        ("icnf", "raw"): (load_raw_icnf, "ICNF Data Profiling (raw)"),
        ("icnf", "processed"): (lambda: pd.read_parquet(PROC / "icnf_fires.parquet"), "ICNF Data Profiling (processed)"),
        ("seviri", "raw"): (load_raw_seviri, "SEVIRI FRP Data Profiling (raw)"),
        ("seviri", "processed"): (lambda: pd.read_parquet(PROC / "seviri_frp.parquet"), "SEVIRI FRP Data Profiling (processed)"),
    }
    for (name, stage), (load, title) in datasets.items():
        path = OUT / stage / f"{name}_eda_report.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        ProfileReport(load(), title=title, explorative=True).to_file(path)
        print(f"Report generated: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
