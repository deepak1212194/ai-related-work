"""
model.py — SARIMA vs XGBoost Model Training & Evaluation
==========================================================
Energy Forecasting Service — Module 2

Trains two forecasting approaches:
  1. SARIMA — classical statistical method capturing seasonality
  2. XGBoost — gradient-boosted trees using engineered features

Compares both using MAE, RMSE, MAPE with proper data leak prevention.
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
TARGET = "Global_active_power"


@dataclass
class ModelResult:
    """Training/evaluation result for a single model."""
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE — handle zeros
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
    }


# ──────────────────────────────────────────────────────────────────────
# SARIMA
# ──────────────────────────────────────────────────────────────────────
def train_sarima(train: pd.DataFrame, test: pd.DataFrame,
                 order: Tuple = (1, 1, 1),
                 seasonal_order: Tuple = (1, 1, 1, 24)) -> ModelResult:
    """
    Train SARIMA model.

    Order (1,1,1)(1,1,1,24):
    - AR(1), I(1), MA(1) for non-seasonal component
    - Seasonal period = 24 hours (daily cycle)
    """
    print("[SARIMA] Training...")
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            train[TARGET],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)

        # Forecast
        n_test = len(test)
        forecast = fitted.forecast(steps=n_test)
        predictions = forecast.values

        metrics = compute_metrics(test[TARGET].values, predictions)
        print(f"[SARIMA] MAE={metrics['mae']} RMSE={metrics['rmse']} MAPE={metrics['mape']}%")

        return ModelResult(
            model_name="SARIMA",
            metrics=metrics,
            predictions=predictions,
        )
    except ImportError:
        print("[SARIMA] statsmodels not installed — skipping")
        return ModelResult(model_name="SARIMA", metrics={"error": "statsmodels not installed"})
    except Exception as e:
        print(f"[SARIMA] Error: {e}")
        return ModelResult(model_name="SARIMA", metrics={"error": str(e)})


# ──────────────────────────────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────────────────────────────
def _get_feature_cols(df: pd.DataFrame) -> list:
    """Get feature columns (exclude target, metadata, and non-numeric)."""
    exclude = {TARGET, "is_anomaly", "was_missing", "is_statistical_anomaly", "is_imputed"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def train_xgboost(train: pd.DataFrame, test: pd.DataFrame) -> ModelResult:
    """
    Train XGBoost regressor using engineered features.

    Data leak prevention: features use only lagged values
    (shift(1) and higher), so no future information leaks in.
    """
    print("[XGBOOST] Training...")
    try:
        import xgboost as xgb

        feature_cols = _get_feature_cols(train)

        # Drop rows with NaN in features (from lagged features)
        train_clean = train[feature_cols + [TARGET]].dropna()
        test_clean = test[feature_cols + [TARGET]].dropna()

        if len(train_clean) == 0 or len(test_clean) == 0:
            return ModelResult(model_name="XGBoost", metrics={"error": "Insufficient data after NaN removal"})

        X_train = train_clean[feature_cols]
        y_train = train_clean[TARGET]
        X_test = test_clean[feature_cols]
        y_test = test_clean[TARGET]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        predictions = model.predict(X_test)
        metrics = compute_metrics(y_test.values, predictions)

        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])

        print(f"[XGBOOST] MAE={metrics['mae']} RMSE={metrics['rmse']} MAPE={metrics['mape']}%")

        return ModelResult(
            model_name="XGBoost",
            metrics=metrics,
            predictions=predictions,
            feature_importance=top_features,
        )
    except ImportError:
        print("[XGBOOST] xgboost not installed — skipping")
        return ModelResult(model_name="XGBoost", metrics={"error": "xgboost not installed"})
    except Exception as e:
        print(f"[XGBOOST] Error: {e}")
        return ModelResult(model_name="XGBoost", metrics={"error": str(e)})


# ──────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────
def compare_models(train: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """Train both models and return comparison results."""
    print("=" * 60)
    print("  MODEL COMPARISON: SARIMA vs XGBoost")
    print("=" * 60)

    sarima = train_sarima(train, test)
    xgboost = train_xgboost(train, test)

    comparison = {
        "sarima": {
            "metrics": sarima.metrics,
            "has_predictions": sarima.predictions is not None,
        },
        "xgboost": {
            "metrics": xgboost.metrics,
            "has_predictions": xgboost.predictions is not None,
            "feature_importance": xgboost.feature_importance,
        },
    }

    # Determine winner (by MAE)
    s_mae = sarima.metrics.get("mae", float("inf"))
    x_mae = xgboost.metrics.get("mae", float("inf"))
    if isinstance(s_mae, (int, float)) and isinstance(x_mae, (int, float)):
        comparison["winner"] = "XGBoost" if x_mae < s_mae else "SARIMA"
        comparison["mae_improvement"] = round(abs(s_mae - x_mae) / max(s_mae, 0.001) * 100, 1)
    else:
        comparison["winner"] = "unknown"

    # Save comparison
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n[RESULT] Winner: {comparison.get('winner', 'N/A')}")
    return comparison


if __name__ == "__main__":
    from .data_prep import main as prep_main
    _, train, test, _, _ = prep_main()
    compare_models(train, test)
