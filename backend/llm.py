"""Seasonality analysis and resilient Gemini-powered Bengali/English advisory engine.

Provides calendar-month price seasonality calculations, safe Gemini integration
with header-based authentication, dynamic model resolution with automated fallback,
exponential backoff for transient errors, and bilingual local advisory fallbacks
that function completely offline without external network calls.
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

DEFAULT_MODEL = "gemini-3-flash-preview"
FALLBACK_MODELS = ["gemini-3-flash-preview", "gemini-2.5-pro", "gemini-3.8-flash"]


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


def fallback_advice(crop: str, summary: Dict[str, Any], lang: str = "bn") -> str:
    """Generate professional, institutional agro-economic advisory in standard Bengali or English."""
    trend = summary.get("recent_trend_pct", 0.0)
    crop_display = crop.replace("_", " ").title()

    if lang == "en":
        peak_en = summary.get("peak_month", "")
        low_en = summary.get("low_month", "")

        if trend > 2.0:
            trend_text = f"Recent 90-day market dynamics indicate an upward trend (+{abs(trend):.1f}%)."
            action_text = (
                "Capitalize on favorable prevailing prices by releasing stored inventories in planned tranches, "
                "while maintaining phased distribution to hedge against potential price pullbacks."
            )
        elif trend < -2.0:
            trend_text = f"Recent 90-day market dynamics show a downward price momentum (-{abs(trend):.1f}%)."
            action_text = (
                "Producers and rural traders are advised to avoid distress selling; leverage ventilated storage "
                "or temporary holding facilities until market prices stabilize."
            )
        else:
            trend_text = "Recent market prices remain relatively stable within historical seasonal boundaries."
            action_text = (
                "Maintain steady, regular supplies to local wholesale trading centers while balancing demand."
            )

        return (
            f"📋 **{crop_display} — Agro-Economic Market Intelligence & Strategic Advisory:**\n\n"
            f"• **Seasonality Profile:** Historical time-series data shows that the highest seasonal price occurs in **{peak_en}** "
            f"(avg {summary['peak_average']:.2f} BDT/kg), while the lowest price level occurs in **{low_en}** (avg {summary['low_average']:.2f} BDT/kg).\n\n"
            f"• **Short-Term Price Momentum:** {trend_text}\n\n"
            f"• **Strategic Marketing Guidance:** {action_text}\n\n"
            f"• **Risk Mitigation & Governance:** Commercial decisions should incorporate local farmgate deductions, rural freight charges, "
            f"and regional weather forecasts. This advisory represents statistical econometric likelihoods based on historical observations, "
            f"not guaranteed financial returns."
        )

    # Default: Professional Bengali
    peak_bn = summary.get("peak_month_bn", summary.get("peak_month", ""))
    low_bn = summary.get("low_month_bn", summary.get("low_month", ""))

    if trend > 2.0:
        trend_analysis = f"বিগত ৯০ দিনের বাজার গতিশীলতায় ঊর্ধ্বমুখী প্রবণতা (+{abs(trend):.1f}%) পরিলক্ষিত হচ্ছে।"
        action_strategy = (
            "বর্তমান অনুকূল বাজারমূল্যের সদ্ব্যবহার করতে সংরক্ষিত বা উৎপাদিত ফসলের একাংশ পরিকল্পিতভাবে "
            "বিপণন করা যেতে পারে। একই সাথে সম্ভাব্য বাজার সংশোধনের ঝুঁকি এড়াতে ধাপে ধাপে সরবরাহ বজায় রাখা যুক্তিসঙ্গত।"
        )
    elif trend < -2.0:
        trend_analysis = f"বিগত ৯০ দিনের বাজার গতিশীলতায় নিম্নমুখী প্রবণতা (-{abs(trend):.1f}%) দৃশ্যমান।"
        action_strategy = (
            "উৎপাদনকারী কৃষক ও ব্যবসায়ীদের তাৎক্ষণিক লোকসানে এককালীন বিক্রয় পরিহার করে উপযুক্ত গুদামজাতকরণ "
            "বা মানসম্মত সাময়িক সংরক্ষণের মাধ্যমে বাজার স্থিতিশীল হওয়া পর্যন্ত অপেক্ষমাণ থাকার পরামর্শ দেওয়া হচ্ছে।"
        )
    else:
        trend_analysis = "সাম্প্রতিক বাজারমূল্য তুলনামূলকভাবে স্থিতিশীল এবং স্বাভাবিক ঐতিহাসিক সীমার মধ্যে অবস্থান করছে।"
        action_strategy = (
            "চাহিদা ও জোগানের ভারসাম্য বজায় রেখে নিয়মিত বিরতিতে স্থানীয় আড়ত ও পাইকারি বাজারে সরবরাহ অব্যাহত রাখা বাঞ্ছনীয়।"
        )

    return (
        f"📋 **{crop_display} — কৃষি-অর্থনৈতিক বাজার বিশ্লেষণ ও কৌশলগত সুপারিশ:**\n\n"
        f"• **মৌসুমি চক্র (Seasonality Profile):** ঐতিহাসিক উপাত্ত পর্যালোচনায় দেখা যায়, এই পণ্যের সর্বোচ্চ মূল্য অর্জিত হয় **{peak_bn}** মাসে "
        f"(গড় {summary['peak_average']:.2f} BDT/kg) এবং সর্বনিম্ন মূল্য পরিলক্ষিত হয় **{low_bn}** মাসে (গড় {summary['low_average']:.2f} BDT/kg)।\n\n"
        f"• **স্বল্পমেয়াদি মূল্য প্রবণতা:** {trend_analysis}\n\n"
        f"• **বাণিজ্যিক বিপণন কৌশল:** {action_strategy}\n\n"
        f"• **ঝুঁকি ব্যবস্থাপনা ও সতর্কতা:** খামার প্রান্তিক বিক্রয়মূল্য, স্থানীয় পরিবহন খরচ এবং জলবায়ু পূর্বাভাস বিবেচনায় রেখে "
        f"চূড়ান্ত বাণিজ্যিক সিদ্ধান্ত গ্রহণ করুন। এটি ঐতিহাসিক সময়ক্রমিক পরিসংখ্যানগত দিকনির্দেশনা, কোনো নিশ্চয়তামূলক আর্থিক প্রতিশ্রুতি নয়।"
    )


def llm_advice(
    crop: str,
    summary: Dict[str, Any],
    max_retries: int = 2,
    timeout: float = 20.0,
    api_key_override: Optional[str] = None,
    lang: str = "bn",
) -> str:
    """Generate Bengali/English advice using Gemini API with safe header auth, multi-model fallback, and offline backup."""
    api_key = api_key_override if api_key_override is not None else get_api_key()

    if not api_key or not api_key.strip():
        return fallback_advice(crop, summary, lang=lang)

    configured_model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    candidate_models: List[str] = []
    for candidate in [configured_model] + FALLBACK_MODELS:
        if candidate and candidate not in candidate_models:
            candidate_models.append(candidate)

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

    if lang == "en":
        prompt = (
            "You are an expert senior agricultural economist and market intelligence specialist in Bangladesh. "
            "Analyze the provided aggregate historical price statistics and formulate a complete, rigorous, professional market advisory in clear executive English. "
            "Formatting & Style Rules:\n"
            "1. Do NOT write letters, sender/receiver headers ('Subject:', 'From:'), or pleasantries.\n"
            "2. Directly start with structured bullet points using bold headers:\n"
            "   * **Current Market Status & Momentum:** Detailed analysis of recent price movements and current level.\n"
            "   * **Seasonality Dynamics:** Explanation of historical peak and trough months with average benchmark prices.\n"
            "   * **Procurement & Storage Strategy:** Practical timing for phased procurement, ventilated storage, or staged selling.\n"
            "   * **Risk Assessment & Governance:** Risk considerations including farmgate-retail margin gap (15-30%) and volatility.\n"
            "3. Tone: Highly professional, objective, and authoritative.\n"
            "4. Price units must be explicitly stated as BDT/kg.\n"
            "5. Conclude cleanly without getting cut off.\n\n"
            f"Input Data:\n{json.dumps(compact_stats, ensure_ascii=False)}"
        )
    else:
        prompt = (
            "You are an expert senior agricultural economist and agricultural market intelligence specialist in Bangladesh. "
            "Analyze the provided aggregate historical price statistics and formulate a complete, rigorous, professional market advisory in formal standard Bengali (প্রমিত বাংলা). "
            "Formatting & Style Rules:\n"
            "1. Do NOT write letters, sender/receiver headers ('বিষয়:', 'প্রেরক:'), or pleasantries.\n"
            "2. Directly start with structured bullet points using clear bold headers:\n"
            "   * **বাজারের বর্তমান অবস্থা ও প্রবণতা:** Analysis of recent price changes and current level.\n"
            "   * **মৌসুমি মূল্য চক্র (Seasonality Dynamics):** Explanation of historical peak and low months with average prices.\n"
            "   * **সংগ্রহ ও মজুদ কৌশল:** Practical timing for phased procurement, storage, or phased selling.\n"
            "   * **ঝুঁকি মূল্যায়ন ও সতর্কতা:** Risk management considerations (e.g., farmgate vs retail margins, perishability).\n"
            "3. Tone: Highly professional, objective, and authoritative.\n"
            "4. Price units must be explicitly stated as BDT/kg (টাকা/কেজি).\n"
            "5. Conclude cleanly without getting cut off.\n\n"
            f"Input Data:\n{json.dumps(compact_stats, ensure_ascii=False)}"
        )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2500,
        },
    }

    req_data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key.strip(),
    }

    for model in candidate_models:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:generateContent"

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

                display_model = "Gemini AI" if "preview" in model else model
                header_title = (
                    f"🤖 **Agro-Economic Market Intelligence (Prepared by {display_model}):**"
                    if lang == "en"
                    else f"🤖 **কৃষি-অর্থনৈতিক বাজার বুদ্ধিমত্তা ({display_model} প্রস্তুতকৃত):**"
                )
                return f"{header_title}\n\n{text}"

            except HTTPError as exc:
                if exc.code == 404:
                    break
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    backoff_time = (2 ** attempt) * 1.0 + random.uniform(0.1, 0.4)
                    time.sleep(backoff_time)
                    continue
                break
            except (URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError, RuntimeError) as exc:
                if attempt < max_retries and isinstance(exc, (URLError, TimeoutError)):
                    backoff_time = (2 ** attempt) * 0.8 + random.uniform(0.1, 0.3)
                    time.sleep(backoff_time)
                    continue
                break

    deterministic = fallback_advice(crop, summary, lang=lang)
    error_note = (
        (
            f"\n\n*(Notice: Unable to connect to Gemini AI servers; "
            f"displaying verified deterministic offline analytical advisory.)*"
        )
        if lang == "en"
        else (
            f"\n\n*(বিজ্ঞপ্তি: Gemini এআই সার্ভারের সাথে সংযোগ সম্ভব না হওয়ায় "
            f"বিকল্প ব্যাকআপ অ্যালগরিদম ভিত্তিক বিশ্লেষণ প্রদর্শিত হয়েছে।)*"
        )
    )
    return deterministic + error_note
