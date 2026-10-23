"""Seasonality analysis and resilient Gemini-powered Bengali advisory engine.

Provides calendar-month price seasonality calculations, safe Gemini 2.0 Flash integration
with header-based authentication, exponential backoff for transient errors, and a deterministic
local Bengali advisory fallback that functions completely offline without external network calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
from typing import Dict, Any, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

MONTH_NAMES_EN = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

MONTH_NAMES_BN = {
    1: "জানুয়ারি", 2: "ফেব্রুয়ারি", 3: "মার্চ", 4: "এপ্রিল",
    5: "মে", 6: "জুন", 7: "জুলাই", 8: "আগস্ট",
    9: "সেপ্টেম্বর", 10: "অক্টোবর", 11: "নভেম্বর", 12: "ডিসেম্বর",
}

# Supported Gemini models in preferred priority order
DEFAULT_MODEL = "gemini-2.0-flash"


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs without overwriting explicit environment values.

    Never prints or logs secret contents.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except Exception:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        # Only set if key does not already exist in the environment
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    """Safely retrieve Gemini API key from environment, Streamlit secrets, or .env."""
    _load_dotenv()
    key = os.getenv("GEMINI_API_KEY") or os.getenv("gemini_api_key") or ""
    if not key:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                if "GEMINI_API_KEY" in st.secrets:
                    key = str(st.secrets["GEMINI_API_KEY"])
                elif "gemini_api_key" in st.secrets:
                    key = str(st.secrets["gemini_api_key"])
        except Exception:
            pass
    return key.strip()


def seasonal_analysis(series: pd.Series) -> Dict[str, Any]:
    """Return calendar-month price statistics and actionable seasonality facts."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean.index)
    clean = clean[clean > 0].sort_index()

    if clean.empty:
        raise ValueError("Seasonal analysis requires at least one positive price.")

    monthly = (
        clean.to_frame("price")
        .assign(month=lambda frame: frame.index.month)
        .groupby("month")["price"]
        .agg(["mean", "min", "max", "count"])
        .reindex(range(1, 13))
    )
    overall_mean = float(clean.mean())
    monthly["seasonal_index"] = monthly["mean"] / overall_mean

    available = monthly.dropna(subset=["mean"]).copy()
    if available.empty:
        raise ValueError("No valid monthly data available for seasonal calculation.")

    peak_month = int(available["mean"].idxmax())
    low_month = int(available["mean"].idxmin())

    recent = clean.tail(min(90, len(clean)))
    earlier = clean.iloc[-min(180, len(clean)) : -len(recent)]
    trend_pct = 0.0 if earlier.empty or earlier.mean() == 0 else float((recent.mean() / earlier.mean() - 1) * 100)

    rows: List[Dict[str, Any]] = []
    for month, row in monthly.iterrows():
        rows.append({
            "month": month,
            "month_name": MONTH_NAMES_EN[month],
            "month_name_bn": MONTH_NAMES_BN[month],
            "average_price": None if pd.isna(row["mean"]) else round(float(row["mean"]), 2),
            "minimum_price": None if pd.isna(row["min"]) else round(float(row["min"]), 2),
            "maximum_price": None if pd.isna(row["max"]) else round(float(row["max"]), 2),
            "observations": int(row["count"]) if not pd.isna(row["count"]) else 0,
            "seasonal_index": None if pd.isna(row["seasonal_index"]) else round(float(row["seasonal_index"]), 3),
        })

    return {
        "overall_average": round(overall_mean, 2),
        "peak_month": MONTH_NAMES_EN[peak_month],
        "peak_month_bn": MONTH_NAMES_BN[peak_month],
        "peak_month_number": peak_month,
        "peak_average": round(float(available.loc[peak_month, "mean"]), 2),
        "low_month": MONTH_NAMES_EN[low_month],
        "low_month_bn": MONTH_NAMES_BN[low_month],
        "low_month_number": low_month,
        "low_average": round(float(available.loc[low_month, "mean"]), 2),
        "recent_trend_pct": round(trend_pct, 2),
        "monthly": rows,
        "data_start": clean.index.min().date().isoformat(),
        "data_end": clean.index.max().date().isoformat(),
    }


def fallback_advice(crop: str, summary: Dict[str, Any]) -> str:
    """Generate deterministic, expert-validated Bengali advisory without network calls."""
    trend = summary.get("recent_trend_pct", 0.0)
    peak_bn = summary.get("peak_month_bn", summary.get("peak_month", ""))
    low_bn = summary.get("low_month_bn", summary.get("low_month", ""))
    crop_display = crop.replace("_", " ").title()

    if trend > 2.0:
        trend_text = f"সাম্প্রতিক সময়ে বাজারদর ঊর্ধ্বমুখী (প্রায় +{abs(trend):.1f}% বৃদ্ধি পেয়েছে)"
        action_text = "বর্তমান বাড়তি দরে আপনার মজুদ ফসলের একটি অংশ ধাপে ধাপে বিক্রি করতে পারেন।"
    elif trend < -2.0:
        trend_text = f"সাম্প্রতিক সময়ে বাজারদর নিম্নমুখী (প্রায় -{abs(trend):.1f}% হ্রাস পেয়েছে)"
        action_text = "সংরক্ষণ সুবিধা থাকলে এখনই তাড়াহুড়ো করে সব বিক্রি না করে বাজার স্থিতিশীল হওয়ার অপেক্ষা করুন।"
    else:
        trend_text = "সাম্প্রতিক বাজারদর মোটামুটি স্থিতিশীল অবস্থায় রয়েছে"
        action_text = "নিয়মিত স্থানীয় মোকামের দাম যাচাই করে স্বাভাবিক গতিতে পণ্য সরবরাহ বজায় রাখুন।"

    return (
        f"🌾 **{crop_display}-এর বাজার পরামর্শ (অফলাইন অ্যালগরিদম ভিত্তিক):**\n\n"
        f"১. **মৌসুমি বিশ্লেষণ:** ঐতিহাসিকভাবে এই ফসলের সর্বোচ্চ দাম পাওয়া যায় **{peak_bn}** মাসে "
        f"(গড় {summary['peak_average']:.2f} BDT/kg), এবং সর্বনিম্ন দাম দেখা যায় **{low_bn}** মাসে "
        f"(গড় {summary['low_average']:.2f} BDT/kg)।\n"
        f"২. **বর্তমান প্রবণতা:** {trend_text}।\n"
        f"৩. **ব্যবহারিক সুপারিশ:** {action_text}\n"
        f"৪. **সতর্কতা:** ফসল তোলার খরচ, পরিবহন ও আবহাওয়া পরিস্থিতির সাথে স্থানীয় পাইকারি আড়তের দর যাচাই করে চূড়ান্ত সিদ্ধান্ত নিন। "
        f"এটি অতীত ডেটাভিত্তিক পরিসংখ্যানগত দিকনির্দেশনা, নিশ্চিত আর্থিক প্রতিশ্রুতি নয়।"
    )


def llm_advice(
    crop: str,
    summary: Dict[str, Any],
    max_retries: int = 2,
    timeout: float = 15.0,
    api_key_override: Optional[str] = None,
) -> str:
    """Generate Bengali advice using Gemini API with safe header auth and deterministic fallback.

    - Protects secrets: Uses `x-goog-api-key` header to avoid exposing API keys in URLs or logs.
    - Transient retry: Retries on 429 (rate-limit) and 503 (service unavailable) with exponential backoff.
    - Fallback guarantee: Never crashes the user interface; smoothly falls back to deterministic local advice.
    """
    api_key = api_key_override if api_key_override is not None else get_api_key()

    if not api_key or not api_key.strip():
        # Fallback cleanly when API key is not configured
        return fallback_advice(crop, summary)

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    # Use clean endpoint without query parameters; key is passed safely in header
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"

    # Minimal, safe payload with aggregate summary statistics only
    compact_stats = {
        "crop": crop,
        "overall_average_bdt_per_kg": summary.get("overall_average"),
        "peak_month": summary.get("peak_month"),
        "peak_average_bdt_per_kg": summary.get("peak_average"),
        "low_month": summary.get("low_month"),
        "low_average_bdt_per_kg": summary.get("low_average"),
        "recent_trend_percentage": summary.get("recent_trend_pct"),
        "data_period": f"{summary.get('data_start')} to {summary.get('data_end')}",
    }

    prompt = (
        "You are an agricultural economist and market advisor for Bangladeshi smallholder farmers. "
        "Analyze the provided aggregate historical price statistics and produce a concise, practical recommendation in Bengali (বাংলা). "
        "Strict rules:\n"
        "1. Answer strictly in Bengali with supportive tone.\n"
        "2. Do not guarantee future prices; emphasize that this is historical seasonal guidance.\n"
        "3. Provide exactly two actionable recommendations (e.g. storage timing, phased selling, or market monitoring).\n"
        "4. Mention price units clearly as টাকা/কেজি (BDT/kg).\n\n"
        f"Input Data:\n{json.dumps(compact_stats, ensure_ascii=False)}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }

    req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip(),
    }

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        request = Request(endpoint, data=req_data, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                resp_bytes = response.read()
                resp_json = json.loads(resp_bytes.decode("utf-8"))

            candidates = resp_json.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned an empty candidate list.")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini returned an empty content body.")

            text = parts[0].get("text", "").strip()
            if not text:
                raise RuntimeError("Gemini returned blank advisory text.")

            return f"🤖 **AgroMind AI পরামর্শ (Gemini 2.0 Flash প্রস্তুতকৃত):**\n\n{text}"

        except HTTPError as exc:
            last_error = exc
            # Transient HTTP status codes: 429 (Too Many Requests), 500, 502, 503, 504
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                backoff_time = (2 ** attempt) * 1.0 + random.uniform(0.1, 0.5)
                time.sleep(backoff_time)
                continue
            # For 400, 401, 403, 404 or exhausted retries, exit retry loop
            break
        except (URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, RuntimeError) as exc:
            last_error = exc
            if attempt < max_retries and isinstance(exc, (URLError, TimeoutError)):
                backoff_time = (2 ** attempt) * 0.8 + random.uniform(0.1, 0.4)
                time.sleep(backoff_time)
                continue
            break

    # If Gemini failed due to network/API error, smoothly fall back to deterministic advice
    # without breaking the application or leaking secrets
    deterministic = fallback_advice(crop, summary)
    error_note = (
        f"\n\n*(বিজ্ঞপ্তি: Gemini সার্ভার সংযোগে সমস্যা হওয়ায় স্থানীয় ব্যাকআপ অ্যালগরিদম "
        f"দিয়ে পরামর্শ প্রদর্শন করা হয়েছে।)*"
    )
    return deterministic + error_note
