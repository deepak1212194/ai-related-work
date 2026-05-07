"""
data_prep.py — Data Loading, Cleaning, and Preparation
=======================================================
Energy Forecasting Service — Module 1

Reads the UCI household power consumption dataset, cleans it,
resamples to hourly resolution, engineers features, detects anomalies,
and splits into train/test sets.

Every decision (resolution, cleaning, splitting, anomaly definition)
is documented inline with rationale.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "artifacts"

RAW_DATA_PATH = DATA_DIR / "household_power_consumption.txt"
SAMPLE_DATA_PATH = DATA_DIR / "sample_hourly.csv"

CLEANED_CSV = OUTPUT_DIR / "cleaned_hourly_data.csv"
TRAIN_CSV = OUTPUT_DIR / "train.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
ANOMALY_BOUNDS_JSON = OUTPUT_DIR / "anomaly_bounds.json"
DATA_QUALITY_REPORT = OUTPUT_DIR / "data_quality_report.json"

RESAMPLE_FREQ = "1h"
TEST_START_DATE = "2010-10-01"
IQR_MULTIPLIER = 3.0
TARGET_COLUMN = "Global_active_power"


# ──────────────────────────────────────────────────────────────────────
# Module 1: Data Loading
# ──────────────────────────────────────────────────────────────────────
def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the UCI household power consumption dataset.

    Falls back to sample data if full dataset is not present.
    """
    path = filepath or RAW_DATA_PATH

    if path.exists() and path.suffix == ".txt":
        print(f"[LOAD] Reading full dataset from {path}")
        df = pd.read_csv(path, sep=";", low_memory=False, na_values=["?", ""])
        df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
        df.drop(columns=["Date", "Time"], inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
    elif SAMPLE_DATA_PATH.exists():
        print(f"[LOAD] Full dataset not found. Using sample: {SAMPLE_DATA_PATH}")
        df = pd.read_csv(SAMPLE_DATA_PATH, index_col="datetime", parse_dates=True)
    else:
        raise FileNotFoundError(
            f"No data found. Download the UCI dataset to {RAW_DATA_PATH} "
            f"or ensure {SAMPLE_DATA_PATH} exists."
        )

    print(f"[LOAD] {len(df):,} rows | {df.index.min()} → {df.index.max()}")
    return df


# ──────────────────────────────────────────────────────────────────────
# Module 2: Data Quality Assessment
# ──────────────────────────────────────────────────────────────────────
def assess_data_quality(df: pd.DataFrame) -> Dict:
    """Quantify data quality issues before cleaning."""
    report = {}

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(4)
    report["missing_values"] = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns
    }

    total_missing = int(missing.sum())
    total_cells = len(df) * len(df.columns)
    report["total_missing_pct"] = round(total_missing / total_cells * 100, 4)

    dup_count = int(df.index.duplicated().sum())
    report["duplicate_timestamps"] = dup_count

    if TARGET_COLUMN in df.columns:
        zero_power = int((df[TARGET_COLUMN] == 0).sum())
        report["zero_power_readings"] = zero_power
        report["zero_power_pct"] = round(zero_power / len(df) * 100, 4)

        report["basic_stats"] = {
            col: {
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
            for col in df.select_dtypes(include=[np.number]).columns
        }

    report["date_range"] = {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "total_rows": len(df),
    }

    print(f"[QUALITY] Missing: {report['total_missing_pct']}% | Duplicates: {dup_count}")
    return report


# ──────────────────────────────────────────────────────────────────────
# Module 3: Data Cleaning
# ──────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    1. Remove duplicate timestamps
    2. Reindex to continuous range
    3. Forward-fill short gaps (≤5 points)
    4. Interpolate medium gaps (≤60 points)
    5. Drop remaining NaN
    """
    print("[CLEAN] Starting...")

    # Remove duplicates
    dup_mask = df.index.duplicated(keep="first")
    if dup_mask.sum() > 0:
        df = df[~dup_mask].copy()
        print(f"[CLEAN] Removed {dup_mask.sum()} duplicate timestamps")

    # Determine frequency
    freq = pd.infer_freq(df.index[:100])
    if freq is None:
        freq = "1min" if len(df) > 50000 else "1h"

    # Reindex
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    n_gaps = len(full_idx) - len(df)
    df = df.reindex(full_idx)
    df.index.name = "datetime"
    print(f"[CLEAN] Reindexed: {n_gaps:,} gaps filled as NaN")

    # Track missing
    if TARGET_COLUMN in df.columns:
        df["was_missing"] = df[TARGET_COLUMN].isnull().astype(int)

    # Fill gaps
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == "was_missing":
            continue
        df[col] = df[col].ffill(limit=5)
        df[col] = df[col].interpolate(method="linear", limit=60)

    if TARGET_COLUMN in df.columns:
        remaining = df[TARGET_COLUMN].isnull().sum()
        print(f"[CLEAN] After fill: {remaining:,} NaN remain")
        df = df.dropna(subset=[TARGET_COLUMN])

    print(f"[CLEAN] Final: {len(df):,} rows")
    return df


# ──────────────────────────────────────────────────────────────────────
# Module 4: Resampling
# ──────────────────────────────────────────────────────────────────────
def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to hourly resolution.

    Rationale: 24-hour forecast → 24 steps. Hourly matches
    how grid operators plan load dispatch.
    """
    # Check if already hourly
    if hasattr(df.index, 'freq') and df.index.freq and df.index.freq.name == 'h':
        print("[RESAMPLE] Already hourly — skipping")
        return df

    print(f"[RESAMPLE] Resampling to {RESAMPLE_FREQ}")
    agg_rules = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if "sub_metering" in col.lower() or "Sub_metering" in col:
            agg_rules[col] = "sum"
        elif col == "was_missing":
            agg_rules[col] = "max"
        else:
            agg_rules[col] = "mean"

    df_hourly = df.resample(RESAMPLE_FREQ).agg(agg_rules)
    df_hourly = df_hourly.dropna(subset=[TARGET_COLUMN] if TARGET_COLUMN in df_hourly.columns else [])

    print(f"[RESAMPLE] Hourly: {len(df_hourly):,} rows")
    return df_hourly


# ──────────────────────────────────────────────────────────────────────
# Module 5: Feature Engineering
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for the ML model:
    1. Calendar (hour, day, month, weekend) with cyclical encoding
    2. Lag features (1h, 24h, 168h)
    3. Rolling statistics (24h, 168h)
    """
    print("[FEATURES] Engineering...")
    df = df.copy()

    # Calendar
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lags
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)

    # Rolling stats
    df["rolling_mean_24h"] = df[TARGET_COLUMN].shift(1).rolling(24, min_periods=12).mean()
    df["rolling_std_24h"] = df[TARGET_COLUMN].shift(1).rolling(24, min_periods=12).std()
    df["rolling_mean_168h"] = df[TARGET_COLUMN].shift(1).rolling(168, min_periods=84).mean()

    n_feat = len([c for c in df.columns if c != TARGET_COLUMN])
    print(f"[FEATURES] {n_feat} features created")
    return df


# ──────────────────────────────────────────────────────────────────────
# Module 6: Anomaly Detection
# ──────────────────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame, train_end: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Flag anomalous readings using 3×IQR bounds computed on training data only.

    3×IQR chosen over 1.5× (Tukey's default) because power consumption is
    right-skewed — 1.5× flags ~7% of data (too many normal evening peaks).
    3× flags ~0.5-1%, catching genuine sensor faults only.
    """
    print("[ANOMALY] Detecting...")
    df = df.copy()

    train_data = df.loc[:train_end, TARGET_COLUMN].dropna()
    Q1 = float(train_data.quantile(0.25))
    Q3 = float(train_data.quantile(0.75))
    IQR = Q3 - Q1
    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR

    df["is_anomaly"] = ((df[TARGET_COLUMN] < lower) | (df[TARGET_COLUMN] > upper)).astype(int)

    bounds = {
        "Q1": round(Q1, 4), "Q3": round(Q3, 4), "IQR": round(IQR, 4),
        "lower_bound": round(lower, 4), "upper_bound": round(upper, 4),
        "n_anomalies": int(df["is_anomaly"].sum()),
        "anomaly_pct": round(df["is_anomaly"].mean() * 100, 4),
    }
    print(f"[ANOMALY] Bounds: [{bounds['lower_bound']}, {bounds['upper_bound']}] → {bounds['n_anomalies']} flagged")
    return df, bounds


# ──────────────────────────────────────────────────────────────────────
# Module 7: Train/Test Split
# ──────────────────────────────────────────────────────────────────────
def split_train_test(df: pd.DataFrame, test_start: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split — no data leakage.

    Random/k-fold splits leak future patterns into training.
    """
    ts = test_start or TEST_START_DATE
    print(f"[SPLIT] At {ts}")

    train = df.loc[:pd.Timestamp(ts) - pd.Timedelta("1h")].copy()
    test = df.loc[ts:].copy()

    print(f"[SPLIT] Train: {len(train):,} | Test: {len(test):,} | "
          f"Ratio: {len(train)/len(df)*100:.1f}/{len(test)/len(df)*100:.1f}")
    return train, test


# ──────────────────────────────────────────────────────────────────────
# Module 8: Save Artifacts
# ──────────────────────────────────────────────────────────────────────
def save_artifacts(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                   quality_report: Dict, anomaly_bounds: Dict) -> None:
    """Persist all artifacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEANED_CSV)
    train.to_csv(TRAIN_CSV)
    test.to_csv(TEST_CSV)

    with open(ANOMALY_BOUNDS_JSON, "w") as f:
        json.dump(anomaly_bounds, f, indent=2)
    with open(DATA_QUALITY_REPORT, "w") as f:
        json.dump(quality_report, f, indent=2, default=str)

    print(f"[SAVE] Artifacts → {OUTPUT_DIR}/")


# ──────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────
def main() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Execute the full data preparation pipeline."""
    print("=" * 60)
    print("  ENERGY FORECASTING — DATA PREPARATION")
    print("=" * 60)

    df_raw = load_raw_data()
    quality_report = assess_data_quality(df_raw)
    df_clean = clean_data(df_raw)
    df_hourly = resample_to_hourly(df_clean)
    df_hourly, anomaly_bounds = detect_anomalies(df_hourly, TEST_START_DATE)
    df_featured = engineer_features(df_hourly)
    train, test = split_train_test(df_featured)
    save_artifacts(df_featured, train, test, quality_report, anomaly_bounds)

    print("\n" + "=" * 60)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 60)
    return df_featured, train, test, quality_report, anomaly_bounds


if __name__ == "__main__":
    main()
