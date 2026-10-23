"""Unit tests for backend.market."""

import unittest
import pandas as pd
from backend.market import (
    CROP_METADATA,
    calculate_volatility_metrics,
    get_all_crops_summary,
    get_crop_display_label,
    get_crop_info,
    list_available_crops,
)


class TestMarketMetadata(unittest.TestCase):
    """Test crop metadata, translations, volatility calculations, and 5-crop overview."""

    def test_all_crops_have_required_fields(self):
        crops = list_available_crops()
        self.assertGreaterEqual(len(crops), 5)
        required_keys = {"en_name", "bn_name", "category", "season_bn", "volatility", "unit", "icon"}
        for crop in crops:
            info = get_crop_info(crop)
            for k in required_keys:
                self.assertIn(k, info)
                self.assertTrue(len(str(info[k])) > 0)

    def test_get_crop_display_label_bilingual(self):
        label_bn = get_crop_display_label("garlic", lang="bn")
        self.assertIn("রসুন", label_bn)
        self.assertIn("Garlic", label_bn)

        label_en = get_crop_display_label("garlic", lang="en")
        self.assertIn("Garlic", label_en)
        self.assertIn("রসুন", label_en)

    def test_calculate_volatility_metrics(self):
        # Synthetic constant series (0 volatility)
        const_s = pd.Series([100.0] * 10)
        v_const = calculate_volatility_metrics(const_s)
        self.assertEqual(v_const["cv_pct"], 0.0)
        self.assertIn("স্থিতিশীল", v_const["risk_label_bn"])

        # Synthetic volatile series
        vol_s = pd.Series([50.0, 150.0, 40.0, 200.0])
        v_vol = calculate_volatility_metrics(vol_s)
        self.assertGreater(v_vol["cv_pct"], 35.0)
        self.assertIn("উচ্চ", v_vol["risk_label_bn"])

    def test_get_all_crops_summary(self):
        summary = get_all_crops_summary(horizon=7)
        self.assertEqual(len(summary), 5)
        for item in summary:
            self.assertIn("crop", item)
            self.assertIn("latest_price", item)
            self.assertIn("cv_pct", item)
            self.assertIn("recommended_model", item)

    def test_invalid_crop_raises_error(self):
        with self.assertRaises(ValueError):
            get_crop_info("invalid_crop_name")


if __name__ == "__main__":
    unittest.main()
