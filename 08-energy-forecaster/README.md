# 08 · Energy Forecasting Service

**SARIMA vs XGBoost** comparison for 24-hour-ahead electricity demand forecasting, exposed as a FastAPI service with an interactive dashboard.

Based on the [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) dataset (2M+ minute-level readings over 4 years).

## What it demonstrates

- **Full ML pipeline**: data quality → cleaning → resampling → feature engineering → anomaly detection → model training → evaluation
- **Model comparison**: Classical (SARIMA) vs ML (XGBoost) with MAE, RMSE, MAPE metrics
- **Feature engineering**: Cyclical calendar encoding, lag features, rolling statistics — all with proper data leak prevention
- **Anomaly detection**: 3×IQR bounds computed on training data only
- **Chronological split**: No random/k-fold — strict temporal integrity

## Data Setup

The full UCI dataset is **not included** due to size (~130MB). To use it:

```bash
# Download from UCI
wget https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip
unzip individual+household+electric+power+consumption.zip -d data/
```

A **48-hour sample** (`data/sample_hourly.csv`) is included for quick testing.

## Quick start

```bash
cd 08-energy-forecaster
pip install -r requirements.txt

# Using sample data
uvicorn app.main:app --reload --port 8008
# Open http://localhost:8008
# Click "Prepare Data" → "Train & Compare"

# Using full dataset (after downloading)
python -m src.data_prep
python -m src.model
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status + data/model availability |
| `POST` | `/api/prepare` | Run full data preparation pipeline |
| `POST` | `/api/train` | Train SARIMA + XGBoost, return comparison |
| `GET` | `/api/results` | Get latest comparison results |

## Architecture

```
Raw Data → Quality Check → Clean → Resample (Hourly) → Feature Engineering
                                                              ↓
                                                    Anomaly Detection (3×IQR)
                                                              ↓
                                                    Chronological Split
                                                     ↓              ↓
                                                  SARIMA         XGBoost
                                                     ↓              ↓
                                                    MAE/RMSE/MAPE Comparison
```

## Stack

FastAPI · XGBoost · statsmodels · pandas · numpy · Pydantic · Docker
