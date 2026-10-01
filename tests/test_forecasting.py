"""Tests for deterministic generation, leakage controls, and forecasting evidence."""

import unittest

import pandas as pd

from forecasting import (
    FEATURE_COLUMNS,
    chronological_split,
    generate_demand,
    make_features,
    rolling_backtest,
    run_pipeline,
)


class ForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate_demand()
        cls.featured = make_features(cls.raw)
        cls.summary, cls.predictions = run_pipeline()

    def test_generation_is_deterministic(self):
        pd.testing.assert_frame_equal(generate_demand(), generate_demand())

    def test_invalid_short_history_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_demand(days=30)

    def test_feature_contract_has_no_missing_values(self):
        self.assertFalse(self.featured[FEATURE_COLUMNS].isna().any().any())

    def test_lag_1_uses_the_previous_observation(self):
        series = self.featured.query("store == 'North' and product == 'Core'").reset_index(drop=True)
        raw_series = self.raw.query("store == 'North' and product == 'Core'").set_index("date")
        row = series.iloc[12]
        prior_date = row["date"] - pd.Timedelta(days=1)
        self.assertAlmostEqual(row["lag_1"], raw_series.loc[prior_date, "demand"])

    def test_splits_are_strictly_chronological(self):
        train, validation, test = chronological_split(self.featured)
        self.assertLess(train["date"].max(), validation["date"].min())
        self.assertLess(validation["date"].max(), test["date"].min())

    def test_rolling_folds_have_ordered_dates(self):
        folds = rolling_backtest(self.featured)
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(fold["train_end"] < fold["test_start"] for fold in folds))

    def test_model_improves_on_seasonal_naive_holdout(self):
        self.assertLess(
            self.summary["holdout"]["model"]["wape"],
            self.summary["holdout"]["seasonal_naive"]["wape"],
        )

    def test_pipeline_is_reproducible(self):
        second, second_predictions = run_pipeline()
        self.assertEqual(self.summary, second)
        pd.testing.assert_frame_equal(self.predictions, second_predictions)


if __name__ == "__main__":
    unittest.main()
