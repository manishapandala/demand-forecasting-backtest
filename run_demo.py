"""Generate the forecast report and publishable prediction artifact."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from forecasting import run_pipeline


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    started = perf_counter()
    summary, predictions = run_pipeline()
    summary["runtime_seconds"] = round(perf_counter() - started, 3)
    ARTIFACTS.mkdir(exist_ok=True)
    (ROOT / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (ARTIFACTS / "forecast_report.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_csv(ARTIFACTS / "holdout_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
