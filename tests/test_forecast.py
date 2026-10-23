"""Unit and integration tests for backend.forecast."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backend.forecast import (
    CROPS,
    MIN_ARIMA_POINTS,
    arima_forecast,
    evaluate,
    load_series,
    naive_forecast,
    validate_crop,
    validate_horizon,
)


class TestForecastValidation(unittest.TestCase):
    """Test input validation for crops and forecast horizons."""

    def test_validate_crop_success(self):
        for crop in CROPS:
            self.assertEqual(validate_crop(crop), crop)
            self.assertEqual(validate_crop(crop.upper()), crop)
            self.assertEqual(validate_crop(f"  {crop}  "), crop)

    def test_validate_crop_invalid(self):
        with self.assertRaises(ValueError):
            validate_crop("unknown_grain")
        with self.assertRaises(ValueError):
            validate_crop("")
        with self.assertRaises(ValueError):
            validate_crop("   ")

    def test_validate_horizon_success(self):
        self.assertEqual(validate_horizon(7), 7)
        self.assertEqual(validate_horizon(14), 14)
        self.assertEqual(validate_horizon(30), 30)
        self.assertEqual(validate_horizon(90), 90)

    def test_validate_horizon_bounds(self):
        with self.assertRaises(ValueError):
            validate_horizon(0)
        with self.assertRaises(ValueError):
            validate_horizon(-5)
        with self.assertRaises(ValueError):
            validate_horizon(100)

    def test_validate_horizon_types(self):
        with self.assertRaises(TypeError):
            validate_horizon("14")  # type: ignore
        with self.assertRaises(TypeError):
            validate_horizon(14.5)  # type: ignore


class TestDataLoading(unittest.TestCase):
    """Test CSV loading, zero price handling, date filling, and error scenarios."""

    def test_load_all_standard_crops(self):
        for crop in CROPS:
            s = load_series(crop)
            self.assertIsInstance(s, pd.Series)
            self.assertFalse(s.empty)
            self.assertIsInstance(s.index, pd.DatetimeIndex)
            # Ensure series is contiguous daily
            diffs = s.index.to_series().diff().dropna()
            self.assertTrue((diffs == pd.Timedelta(days=1)).all())
            # Ensure strictly positive prices
            self.assertTrue((s > 0).all())
            # Ensure no NaNs
            self.assertFalse(s.isna().any())

    def test_load_missing_crop_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                load_series("garlic", raw_dir=Path(tmp_dir))

    def test_zero_price_and_gap_handling(self):
        """Synthetic CSV with zeroes and weekend gaps must be cleanly filled."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "garlic.csv"
            dates = ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-06"]
            data = {
                "date": dates,
                "product name": ["Garlic"] * 4,
                "measurement": ["kg"] * 4,
                "wholesale price minimum": [50.0, 52.0, 0.0, 55.0],
                "wholesale price maximum": [60.0, 62.0, 0.0, 65.0],
                "retail price minimum": [55.0, 56.0, 0.0, 58.0],  # 0 on 2024-01-05
                "retail price maximum": [65.0, 66.0, 0.0, 68.0],
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)

            series = load_series("garlic", raw_dir=Path(tmp_dir))
            # 6 days from 2024-01-01 to 2024-01-06
            self.assertEqual(len(series), 6)
            # Ensure zero was replaced and all values are positive
            self.assertTrue((series > 0).all())
            self.assertFalse(series.isna().any())

    def test_malformed_csv_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "garlic.csv"
            pd.DataFrame({"wrong_col": [1, 2]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                load_series("garlic", raw_dir=Path(tmp_dir))


class TestForecastingModels(unittest.TestCase):
    """Test ARIMA and Naive forecasting, confidence interval shapes, and fallbacks."""

    def setUp(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)
        prices = 50.0 + np.cumsum(np.random.normal(0, 0.5, 100))
        self.synthetic_series = pd.Series(prices, index=dates)

    def test_naive_forecast_properties(self):
        horizon = 14
        fc, lower, upper = naive_forecast(self.synthetic_series, horizon)
        self.assertEqual(len(fc), horizon)
        self.assertEqual(len(lower), horizon)
        self.assertEqual(len(upper), horizon)
        # Persistence: forecast equals last observed value
        self.assertTrue(np.allclose(fc, self.synthetic_series.iloc[-1]))
        # Confidence interval order: lower <= forecast <= upper
        self.assertTrue(np.all(lower <= fc))
        self.assertTrue(np.all(fc <= upper))

    def test_arima_forecast_properties(self):
        horizon = 10
        fc, lower, upper, meta = arima_forecast(self.synthetic_series, horizon)
        self.assertEqual(len(fc), horizon)
        self.assertEqual(len(lower), horizon)
        self.assertEqual(len(upper), horizon)
        self.assertFalse(meta["is_fallback"])
        self.assertIn("ARIMA", meta["model_name"])
        # Interval bounds
        self.assertTrue(np.all(lower <= fc + 1e-4))
        self.assertTrue(np.all(fc - 1e-4 <= upper))

    def test_arima_short_series_fallback(self):
        """Series with fewer than MIN_ARIMA_POINTS must cleanly fall back to naive."""
        short_series = self.synthetic_series.iloc[:15]
        self.assertLess(len(short_series), MIN_ARIMA_POINTS)

        fc, lower, upper, meta = arima_forecast(short_series, horizon=7)
        self.assertEqual(len(fc), 7)
        self.assertTrue(meta["is_fallback"])
        self.assertIn("Naive", meta["model_name"])
        self.assertTrue(np.allclose(fc, short_series.iloc[-1]))

    def test_evaluate_walk_forward(self):
        """Walk-forward evaluation returns structured accuracy comparison."""
        res = evaluate("garlic", horizon=7, n_origins=5)
        self.assertEqual(res["crop"], "garlic")
        self.assertIn("naive_mae", res)
        self.assertIn("arima_mae", res)
        self.assertIn("recommended_model", res)
        self.assertIn(res["recommended_model"], ["ARIMA", "Naive"])
        self.assertIn("summary_bn", res)


if __name__ == "__main__":
    unittest.main()
