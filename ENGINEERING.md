# Engineering Review

## Architecture

`forecasting.py` separates deterministic data generation, lag construction, chronological partitioning, model fitting, evaluation, prediction intervals, and rolling backtests. `run_demo.py` writes artifacts. `index.html` presents only values loaded from the generated summary.

Lag and rolling features are shifted before aggregation, so the target row cannot appear in its own predictors. Validation residuals set interval width; the final 84 days are held out from training and validation.

## Verification contract

```bash
python -m ruff check .
python -m compileall -q .
python -m unittest discover -s tests -v
python run_demo.py
```

## Change-safety checklist

- Reject random train/test splits for time-dependent evaluation.
- Require every target-derived feature to be shifted before rolling operations.
- Compare against the seven-day seasonal naive baseline in every fold.
- Report WAPE, MAE, RMSE, and bias; do not rely on one metric.
- Calibrate interval width on validation residuals, never final holdout residuals.
- Re-run the full rolling backtest after feature or model changes.
- Keep generated data labeled as generated and avoid operational savings claims.

## Boundaries

This is a deterministic one-day-ahead simulation. It is not a multi-step recursive forecast, inventory optimizer, or production demand estimate. Promotion and price effects are intentionally generated so the evaluation logic remains reproducible.
