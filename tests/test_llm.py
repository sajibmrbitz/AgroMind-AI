"""Unit and mocked integration tests for backend.llm."""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pandas as pd

from backend.llm import (
    MONTH_NAMES_BN,
    MONTH_NAMES_EN,
    fallback_advice,
    llm_advice,
    seasonal_analysis,
)


class TestSeasonalAnalysis(unittest.TestCase):
    """Test seasonal analysis with synthetic time series with known peaks and troughs."""

    def setUp(self):
        # Construct 2 full years (2022-2023) where July is explicitly peak and January is low
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        prices = []
        for d in dates:
            if d.month == 7:  # July = Peak
                base = 100.0
            elif d.month == 1:  # January = Low
                base = 20.0
            else:
                base = 50.0
            prices.append(base)

        self.synthetic_series = pd.Series(prices, index=dates)

    def test_known_peak_and_trough_detection(self):
        summary = seasonal_analysis(self.synthetic_series)

        self.assertEqual(summary["peak_month"], "July")
        self.assertEqual(summary["peak_month_bn"], "জুলাই")
        self.assertEqual(summary["peak_month_number"], 7)
        self.assertEqual(summary["peak_average"], 100.0)

        self.assertEqual(summary["low_month"], "January")
        self.assertEqual(summary["low_month_bn"], "জানুয়ারি")
        self.assertEqual(summary["low_month_number"], 1)
        self.assertEqual(summary["low_average"], 20.0)

        # 12 monthly rows present
        self.assertEqual(len(summary["monthly"]), 12)

        # Peak seasonal index must be > 1.0, low must be < 1.0
        july_row = [r for r in summary["monthly"] if r["month"] == 7][0]
        jan_row = [r for r in summary["monthly"] if r["month"] == 1][0]
        self.assertGreater(july_row["seasonal_index"], 1.0)
        self.assertLess(jan_row["seasonal_index"], 1.0)

    def test_empty_or_zero_price_series(self):
        empty_series = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            seasonal_analysis(empty_series)

        zero_series = pd.Series([0.0, 0.0], index=pd.date_range("2024-01-01", periods=2))
        with self.assertRaises(ValueError):
            seasonal_analysis(zero_series)


class TestDeterministicFallbackAdvice(unittest.TestCase):
    """Verify local Bengali advice produces expected guidance without network activity."""

    def setUp(self):
        self.mock_summary = {
            "overall_average": 55.0,
            "peak_month": "July",
            "peak_month_bn": "জুলাই",
            "peak_average": 90.0,
            "low_month": "January",
            "low_month_bn": "জানুয়ারি",
            "low_average": 30.0,
            "recent_trend_pct": 5.4,
            "data_start": "2020-01-01",
            "data_end": "2024-12-31",
        }

    @patch("backend.llm.urlopen")
    def test_fallback_makes_no_network_calls(self, mock_urlopen):
        advice = fallback_advice("garlic", self.mock_summary)
        mock_urlopen.assert_not_called()
        self.assertIn("Garlic", advice)
        self.assertIn("জুলাই", advice)
        self.assertIn("জানুয়ারি", advice)
        self.assertIn("BDT/kg", advice)
        # Check disclaimer presence
        self.assertIn("পরিসংখ্যানগত দিকনির্দেশনা", advice)

    @patch("backend.llm.urlopen")
    def test_fallback_english(self, mock_urlopen):
        advice_en = fallback_advice("garlic", self.mock_summary, lang="en")
        mock_urlopen.assert_not_called()
        self.assertIn("Garlic", advice_en)
        self.assertIn("July", advice_en)
        self.assertIn("January", advice_en)
        self.assertIn("Seasonality Profile", advice_en)

    @patch("backend.llm.urlopen")
    def test_llm_advice_defaults_to_fallback_when_no_api_key(self, mock_urlopen):
        """When no key is configured, llm_advice returns deterministic advice without network call."""
        advice = llm_advice("chickpea", self.mock_summary, api_key_override="")
        mock_urlopen.assert_not_called()
        self.assertIn("Chickpea", advice)


class TestGeminiIntegrationMocked(unittest.TestCase):
    """Test Gemini API interactions under mocked success, HTTP errors, and timeouts."""

    def setUp(self):
        self.mock_summary = {
            "overall_average": 60.0,
            "peak_month": "August",
            "peak_month_bn": "আগস্ট",
            "peak_average": 85.0,
            "low_month": "February",
            "low_month_bn": "ফেব্রুয়ারি",
            "low_average": 40.0,
            "recent_trend_pct": -3.2,
            "data_start": "2021-01-01",
            "data_end": "2024-12-31",
        }

    @patch("backend.llm.urlopen")
    def test_gemini_success_and_header_security(self, mock_urlopen):
        """Verify successful Gemini response and ensure API key is in headers, NOT the URL."""
        response_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "কাঁচামরিচের বাজারদর আগস্টে ঊর্ধ্বমুখী থাকবে। ধাপে ধাপে বিক্রি করুন।"}
                        ]
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        advice = llm_advice("green_chilli", self.mock_summary, api_key_override="test_mock_secret_key")
        self.assertIn("কাঁচামরিচের বাজারদর", advice)

        # Inspect the Request object passed to urlopen
        called_req = mock_urlopen.call_args[0][0]
        # Crucial security check: API key MUST NOT appear in URL query string
        self.assertNotIn("test_mock_secret_key", called_req.full_url)
        # API key MUST be in header
        self.assertEqual(called_req.headers.get("X-goog-api-key"), "test_mock_secret_key")

    @patch("backend.llm.urlopen")
    def test_gemini_empty_response_fallback(self, mock_urlopen):
        """Empty response from Gemini triggers fallback rather than crashing."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"candidates": []}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        advice = llm_advice("cucumber", self.mock_summary, api_key_override="test_mock_secret_key")
        self.assertIn("Cucumber", advice)
        self.assertIn("ব্যাকআপ অ্যালগরিদম", advice)

    @patch("backend.llm.urlopen")
    def test_gemini_http_429_rate_limit_fallback(self, mock_urlopen):
        """HTTP 429 triggers retry and clean fallback."""
        mock_urlopen.side_effect = HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(b"Rate limit exceeded"))

        advice = llm_advice("garlic", self.mock_summary, max_retries=1, api_key_override="test_mock_secret_key")
        self.assertIn("Garlic", advice)
        self.assertIn("ব্যাকআপ অ্যালগরিদম", advice)

    @patch("backend.llm.urlopen")
    def test_gemini_timeout_fallback(self, mock_urlopen):
        """Network timeout triggers clean fallback."""
        mock_urlopen.side_effect = TimeoutError("Connection timed out")

        advice = llm_advice("sweet_pumpkin", self.mock_summary, max_retries=1, api_key_override="test_mock_secret_key")
        self.assertIn("Sweet Pumpkin", advice)
        self.assertIn("ব্যাকআপ অ্যালগরিদম", advice)


if __name__ == "__main__":
    unittest.main()
