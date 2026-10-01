# Demand Forecasting Backtest

A leakage-aware, one-day-ahead demand forecasting project with chronological splits, lag and rolling features, an explicit seven-day seasonal baseline, expanding-window backtests, validation-calibrated prediction intervals, and decision-ready error metrics.

The public dataset is deterministic and generated across four store-product series. That keeps the repository reproducible and makes every demand driver visible.

## What it demonstrates

- time-aware feature engineering with shifted lags and rolling statistics;
- separate training, validation, and final 84-day holdout periods;
- histogram gradient boosting compared with a seasonal naive forecast;
- three expanding-window backtest folds;
- MAE, RMSE, WAPE, bias, relative improvement, and interval coverage;
- publishable JSON and CSV artifacts consumed by the project page.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m ruff check .
python -m unittest discover -s tests -v
python run_demo.py
```

Outputs:

- `demo_summary.json` - holdout and rolling-backtest evidence;
- `artifacts/forecast_report.json` - machine-readable evaluation report;
- `artifacts/holdout_predictions.csv` - forecast, baseline, and interval by series/date.

## Honest scope

This generated-data project validates methodology, leakage controls, and reporting. It does not claim forecast accuracy for a real retailer or estimate financial impact.

See [ENGINEERING.md](ENGINEERING.md) for the architecture, leakage review, verification contract, and safe-change checklist.
