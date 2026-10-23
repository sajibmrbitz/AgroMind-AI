"""Unit tests for backend.market."""

import unittest
from backend.market import (
    CROP_METADATA,
    get_crop_display_label,
    get_crop_info,
    list_available_crops,
)


class TestMarketMetadata(unittest.TestCase):
    """Test crop metadata, translations, and agronomic helpers."""

    def test_all_crops_have_required_fields(self):
        crops = list_available_crops()
        self.assertGreaterEqual(len(crops), 5)
        required_keys = {"en_name", "bn_name", "category", "season_bn", "volatility", "unit"}
        for crop in crops:
            info = get_crop_info(crop)
            for k in required_keys:
                self.assertIn(k, info)
                self.assertTrue(len(str(info[k])) > 0)

    def test_get_crop_display_label(self):
        label = get_crop_display_label("garlic")
        self.assertIn("রসুন", label)
        self.assertIn("Garlic", label)

    def test_invalid_crop_raises_error(self):
        with self.assertRaises(ValueError):
            get_crop_info("invalid_crop_name")


if __name__ == "__main__":
    unittest.main()
