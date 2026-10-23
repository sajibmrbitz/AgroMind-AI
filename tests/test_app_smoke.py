"""Smoke test for app.py functions and execution."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from app import get_cached_series, get_cached_evaluation
from backend.forecast import CROPS


class TestAppSmoke(unittest.TestCase):
    """Smoke tests verifying that app functions run cleanly for all crops."""

    def test_get_cached_series_all_crops(self):
        for crop in CROPS:
            s = get_cached_series(crop)
            self.assertIsInstance(s, pd.Series)
            self.assertGreater(len(s), 100)

    def test_get_cached_evaluation_garlic(self):
        eval_metrics = get_cached_evaluation("garlic", 14)
        self.assertIn("recommended_model", eval_metrics)
        self.assertIn("arima_mae", eval_metrics)
        self.assertIn("naive_mae", eval_metrics)


if __name__ == "__main__":
    unittest.main()
