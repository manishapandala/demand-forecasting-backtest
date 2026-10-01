"""Leakage-aware one-day-ahead demand forecasting and rolling backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


RANDOM_STATE = 52
FEATURE_COLUMNS = [
    "promotion",
    "price_index",
    "store_code",
    "product_code",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    wape: float
    bias: float

    def to_dict(self) -> dict:
        return asdict(self)


def generate_demand(days: int = 900, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate deterministic daily demand for four store-product series."""
    if days < 120:
        raise ValueError("days must be at least 120")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    rows = []
    for store_index, store in enumerate(("North", "South")):
        for product_index, product in enumerate(("Core", "Plus")):
            base = 72 + store_index * 13 + product_index * 18
            for offset in range(days):
                date = start + pd.Timedelta(days=offset)
                weekly = 13 * np.sin(2 * np.pi * date.dayofweek / 7)
                yearly = 9 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                trend = 0.025 * offset
                promotion = int((offset + store_index * 3 + product_index * 5) % 29 in {0, 1, 2})
                price_index = 1 + 0.05 * np.sin(2 * np.pi * offset / 61) + 0.025 * product_index
                promo_lift = promotion * (24 + 7 * product_index)
                price_effect = -38 * (price_index - 1)
                noise = rng.normal(0, 4.5)
                demand = max(1, base + weekly + yearly + trend + promo_lift + price_effect + noise)
                rows.append(
                    (date, store, product, promotion, round(float(price_index), 4), round(float(demand), 2))
                )
    return pd.DataFrame(
        rows,
        columns=["date", "store", "product", "promotion", "price_index", "demand"],
    ).sort_values(["date", "store", "product"], ignore_index=True)


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and lag features using only earlier target values."""
    required = {"date", "store", "product", "promotion", "price_index", "demand"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    working = frame.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise")
    working = working.sort_values(["store", "product", "date"]).reset_index(drop=True)
    groups = working.groupby(["store", "product"], sort=False)["demand"]
    for lag in (1, 7, 14, 28):
        working[f"lag_{lag}"] = groups.shift(lag)
    shifted = groups.shift(1)
    working["rolling_mean_7"] = shifted.groupby(
        [working["store"], working["product"]]
    ).transform(lambda values: values.rolling(7, min_periods=7).mean())
    working["rolling_mean_28"] = shifted.groupby(
        [working["store"], working["product"]]
    ).transform(lambda values: values.rolling(28, min_periods=28).mean())
    working["store_code"] = working["store"].map({"North": 0, "South": 1})
    working["product_code"] = working["product"].map({"Core": 0, "Plus": 1})
    day_of_week = working["date"].dt.dayofweek
    day_of_year = working["date"].dt.dayofyear
    working["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    working["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    working["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    working["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return working.dropna(subset=FEATURE_COLUMNS).sort_values(
        ["date", "store", "product"], ignore_index=True
    )


def chronological_split(
    featured: pd.DataFrame,
    validation_days: int = 84,
    test_days: int = 84,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days and test_days must be positive")
    maximum_date = featured["date"].max()
    test_start = maximum_date - pd.Timedelta(days=test_days - 1)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    train = featured[featured["date"] < validation_start].copy()
    validation = featured[(featured["date"] >= validation_start) & (featured["date"] < test_start)].copy()
    test = featured[featured["date"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("not enough history for the requested split")
    return train, validation, test


def build_model(random_state: int = RANDOM_STATE) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=23,
        min_samples_leaf=18,
        l2_regularization=0.8,
        random_state=random_state,
    )


def evaluate(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> ForecastMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = predicted_values - actual_values
    denominator = max(float(np.abs(actual_values).sum()), 1e-9)
    return ForecastMetrics(
        mae=round(float(mean_absolute_error(actual_values, predicted_values)), 3),
        rmse=round(float(np.sqrt(mean_squared_error(actual_values, predicted_values))), 3),
        wape=round(float(np.abs(errors).sum() / denominator), 4),
        bias=round(float(errors.mean()), 3),
    )


def fit_and_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> tuple[dict, pd.DataFrame]:
    model = build_model(random_state)
    model.fit(train[FEATURE_COLUMNS], train["demand"])
    validation_prediction = model.predict(validation[FEATURE_COLUMNS])
    residual_radius = float(np.quantile(np.abs(validation["demand"] - validation_prediction), 0.90))

    prediction = model.predict(test[FEATURE_COLUMNS])
    baseline = test["lag_7"].to_numpy()
    model_metrics = evaluate(test["demand"], prediction)
    baseline_metrics = evaluate(test["demand"], baseline)
    lower = np.maximum(0, prediction - residual_radius)
    upper = prediction + residual_radius
    coverage = float(((test["demand"].to_numpy() >= lower) & (test["demand"].to_numpy() <= upper)).mean())

    output = test[["date", "store", "product", "promotion", "demand"]].copy()
    output["seasonal_naive"] = np.round(baseline, 3)
    output["forecast"] = np.round(prediction, 3)
    output["lower_90"] = np.round(lower, 3)
    output["upper_90"] = np.round(upper, 3)
    metrics = {
        "model": model_metrics.to_dict(),
        "seasonal_naive": baseline_metrics.to_dict(),
        "wape_improvement": round(
            float((baseline_metrics.wape - model_metrics.wape) / baseline_metrics.wape), 4
        ),
        "interval_radius": round(residual_radius, 3),
        "interval_coverage": round(coverage, 4),
    }
    return metrics, output


def rolling_backtest(featured: pd.DataFrame, folds: int = 3, horizon_days: int = 28) -> list[dict]:
    """Run expanding-window folds ending immediately before the final holdout."""
    if folds < 1 or horizon_days < 1:
        raise ValueError("folds and horizon_days must be positive")
    final_date = featured["date"].max() - pd.Timedelta(days=84)
    results = []
    for fold in range(folds):
        test_end = final_date - pd.Timedelta(days=(folds - fold - 1) * horizon_days)
        test_start = test_end - pd.Timedelta(days=horizon_days - 1)
        validation_start = test_start - pd.Timedelta(days=horizon_days)
        train = featured[featured["date"] < validation_start]
        validation = featured[
            (featured["date"] >= validation_start) & (featured["date"] < test_start)
        ]
        test = featured[(featured["date"] >= test_start) & (featured["date"] <= test_end)]
        metrics, _ = fit_and_score(train, validation, test, RANDOM_STATE + fold)
        results.append(
            {
                "fold": fold + 1,
                "train_end": str(train["date"].max().date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "model_wape": metrics["model"]["wape"],
                "baseline_wape": metrics["seasonal_naive"]["wape"],
            }
        )
    return results


def run_pipeline(seed: int = RANDOM_STATE) -> tuple[dict, pd.DataFrame]:
    raw = generate_demand(seed=seed)
    featured = make_features(raw)
    train, validation, test = chronological_split(featured)
    holdout_metrics, predictions = fit_and_score(train, validation, test, seed)
    folds = rolling_backtest(featured)
    summary = {
        "scope": "deterministic generated-demand one-day-ahead forecasting",
        "raw_rows": len(raw),
        "series": int(raw.groupby(["store", "product"]).ngroups),
        "date_range": {
            "start": str(raw["date"].min().date()),
            "end": str(raw["date"].max().date()),
        },
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "holdout": holdout_metrics,
        "backtest_folds": folds,
        "mean_backtest_model_wape": round(float(np.mean([fold["model_wape"] for fold in folds])), 4),
        "mean_backtest_baseline_wape": round(
            float(np.mean([fold["baseline_wape"] for fold in folds])), 4
        ),
    }
    return summary, predictions
