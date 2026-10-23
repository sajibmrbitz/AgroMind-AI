"""Streamlit frontend dashboard for AgroMind AI.

International AI Congress 2026 Submission Edition.
Provides bilingual (Bengali & English) agricultural price intelligence, interactive
commodity selection pop-up dialog, ARIMA forecasting with confidence intervals,
calendar seasonality decomposition, Gemini AI / local deterministic advisory, and CSV data export.
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from backend.forecast import (
    CROPS,
    arima_forecast,
    evaluate,
    load_series,
    validate_crop,
    validate_horizon,
)
from backend.llm import fallback_advice, get_api_key, llm_advice, seasonal_analysis
from backend.market import get_crop_display_label, get_crop_info

# Configure page layout and aesthetics
st.set_page_config(
    page_title="AgroMind AI — Agricultural Price Intelligence & Seasonality",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Theme-Independent Styling (Dark & Light Mode Compatible)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Hind+Siliguri:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Hind Siliguri', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Executive Badge */
    .badge-congress {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        vertical-align: middle;
        margin-left: 10px;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }

    /* Custom KPI Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin: 1rem 0 1.5rem 0;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.5px;
    }
    .metric-delta-pos {
        font-size: 0.82rem;
        font-weight: 600;
        color: #34d399;
        margin-top: 4px;
    }
    .metric-delta-neg {
        font-size: 0.82rem;
        font-weight: 600;
        color: #f87171;
        margin-top: 4px;
    }
    .metric-delta-neutral {
        font-size: 0.82rem;
        font-weight: 600;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Model Governance Banner */
    .governance-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .governance-text {
        font-size: 0.92rem;
        color: #e2e8f0;
        line-height: 1.5;
    }

    /* Executive Advisory Card */
    .advisory-container {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #f1f5f9;
        font-size: 0.95rem;
        line-height: 1.65;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .advisory-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Data Governance & Limitations Disclaimer Box */
    .disclaimer-container {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(245, 159, 0, 0.35);
        border-left: 5px solid #f59f00;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 2rem;
        color: #e2e8f0;
        font-size: 0.90rem;
        line-height: 1.65;
        backdrop-filter: blur(6px);
    }
    .disclaimer-title {
        color: #fbbf24;
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .badge-date {
        background: rgba(15, 23, 42, 0.85);
        color: #38bdf8;
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* Sidebar Agronomic Card */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px;
        margin-top: 1rem;
    }
    .sidebar-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #38bdf8;
        margin-bottom: 8px;
    }
    .sidebar-item {
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-bottom: 6px;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Bilingual Text Repository
TEXTS = {
    "bn": {
        "badge": "Congress 2026 Edition",
        "title": "🌾 AgroMind AI",
        "subtitle": "বাংলাদেশ কৃষি বাজার কাঠামোর সময়ক্রমিক উপাত্ত বিশ্লেষণ, ARIMA পূর্বাভাস ও প্রাতিষ্ঠানিক সিদ্ধান্ত সহায়তা প্ল্যাটফর্ম",
        "control_panel": "⚙️ নিয়ন্ত্রণ প্যানেল (Control Panel)",
        "language_label": "🌐 ভাষা পরিবর্তন (Language):",
        "commodity_selection": "কৃষিপণ্য নির্বাচন (Commodity):",
        "forecast_horizon": "পূর্বাভাস সময়সীমা (Forecast Horizon — দিন):",
        "change_crop_btn": "🌾 ফসল পরিবর্তন করুন (Change Commodity)",
        "agronomic_title": "📊 কৃষি-অর্থনৈতিক বৈশিষ্ট্য (Profile)",
        "category": "শ্রেণি:",
        "season": "উৎপাদন মৌসুম:",
        "volatility": "বাজার অস্থিরতা:",
        "districts": "প্রধান বাণিজ্যিক অঞ্চল:",
        "storage": "💡 সংরক্ষণ ও বিপণন নির্দেশিকা:",
        "ai_connected": "🟢 Gemini AI ইঞ্জিন সংযুক্ত",
        "offline_active": "🟡 অফলাইন পরিসংখ্যানিক পরামর্শক সক্রিয়",
        "latest_price": "সর্বশেষ রেকর্ডকৃত দর (Latest Mid Price)",
        "daily_change": "দৈনিক পরিবর্তন",
        "historical_mean": "ঐতিহাসিক গড় মূল্য (Historical Mean)",
        "all_data_mean": "সমগ্র উপাত্ত সময়কালের গড়",
        "peak_season": "বার্ষিক সর্বোচ্চ মূল্য মৌসুম (Peak Season)",
        "historical_avg": "ঐতিহাসিক গড়:",
        "momentum_90d": "ত্রৈমাসিক মূল্য প্রবণতা (90-Day Momentum)",
        "momentum_sub": "পূর্ববর্তী ত্রৈমাসিকের তুলনায় পরিবর্তন",
        "chart_title": "দৈনিক বাজারদর ও আগামী {horizon} দিনের পূর্বাভাস",
        "chart_hist": "ঐতিহাসিক দর (Historical)",
        "chart_forecast": "প্রাক্কলিত দর (ARIMA Forecast)",
        "chart_lower": "৮০% নিম্ন আস্থার সীমা (Lower Bound)",
        "chart_upper": "৮০% উচ্চ আস্থার সীমা (Upper Bound)",
        "expander_table": "🔍 পূর্বাভাসের পূর্ণাঙ্গ উপাত্ত সারণি ও রফতানি (Forecast Dataset & Table)",
        "download_forecast": "📥 পূর্বাভাসের উপাত্ত ডাউনলোড করুন (Download Forecast CSV)",
        "seasonality_header": "📅 মাসভিত্তিক মৌসুমি মূল্য চক্র (Monthly Seasonality Index)",
        "download_seasonality": "📥 মৌসুমি সূচক ডেটাসেট ডাউনলোড (Download Seasonality CSV)",
        "advisory_header": "💡 প্রাতিষ্ঠানিক কৃষি-অর্থনৈতিক পরামর্শ (Market Advisory)",
        "generate_advisory_btn": "🔄 নতুন বাজার পরামর্শ তৈরি করুন (Generate Advisory)",
        "advisory_box_title": "📋 প্রাতিষ্ঠানিক বাজার গোয়েন্দা ও সিদ্ধান্ত সুপারিশ",
        "disclaimer_title": "⚖️ ডেটা গভর্নেন্স ও প্রাতিষ্ঠানিক ব্যবহারের সীমাবদ্ধতা (Data Governance & Limitations Notice)",
        "disclaimer_1": "১. উপাত্তের উৎস ও সময়কাল: প্রদর্শিত উপাত্ত Mendeley AgriPriceBD (Muhammad et al., 2026, CC BY 4.0) থেকে সংগৃহীত।",
        "disclaimer_2": "২. খুচরা বনাম খামার প্রান্তিক মূল্য ব্যবধান: বিশ্লেষিত দর বিভাগীয় ও প্রধান নগর বাজারসমূহের খুচরা মধ্যবর্তী দর (BDT/kg)। মধ্যস্বত্বভোগী কমিশন, পরিবহন ব্যয় ও অপচয়ের কারণে স্থানীয় খামার প্রান্তিক মূল্য (Farmgate Price) সাধারণত ১৫% থেকে ৩০% কম হতে পারে।",
        "disclaimer_3": "৩. অপ্রত্যাশিত বাজার ঝুঁকি: আকস্মিক বন্যা, অতিবৃষ্টি, আমদানি নীতি পরিবর্তন কিংবা সরবরাহ বিঘ্নের মতো বাহ্যিক ঝুঁকি ঐতিহাসিক মডেলে প্রক্ষেপণযোগ্য নয়। অতএব ৮০% আস্থার সীমার (Confidence Interval) আলোকেই সতর্কতা অবলম্বনপূর্বক বাণিজ্যিক সিদ্ধান্ত গ্রহণ বাঞ্ছনীয়।",
    },
    "en": {
        "badge": "Congress 2026 Edition",
        "title": "🌾 AgroMind AI",
        "subtitle": "Agricultural Market Time-Series Intelligence, ARIMA Forecasting & Institutional Decision Support Platform for Bangladesh",
        "control_panel": "⚙️ Control Panel",
        "language_label": "🌐 Language Selection:",
        "commodity_selection": "Commodity Selection:",
        "forecast_horizon": "Forecast Horizon (Days):",
        "change_crop_btn": "🌾 Change Commodity Selection",
        "agronomic_title": "📊 Agronomic & Market Profile",
        "category": "Category:",
        "season": "Production Season:",
        "volatility": "Price Volatility:",
        "districts": "Major Producing Districts:",
        "storage": "💡 Post-Harvest & Storage Advisory:",
        "ai_connected": "🟢 Gemini AI Engine Connected",
        "offline_active": "🟡 Offline Algorithmic Advisory Active",
        "latest_price": "Latest Recorded Mid Price",
        "daily_change": "Daily Change",
        "historical_mean": "Historical Mean Price",
        "all_data_mean": "Multi-year Dataset Mean",
        "peak_season": "Annual Peak Price Season",
        "historical_avg": "Historical Mean:",
        "momentum_90d": "Quarterly Price Momentum (90d)",
        "momentum_sub": "Relative to Previous Quarter",
        "chart_title": "Daily Price Dynamics & Next {horizon}-Day Forecast",
        "chart_hist": "Historical Price",
        "chart_forecast": "Forecast (ARIMA)",
        "chart_lower": "Lower Bound (80% CI)",
        "chart_upper": "Upper Bound (80% CI)",
        "expander_table": "🔍 Detailed Forecast Dataset & Export",
        "download_forecast": "📥 Download Forecast CSV",
        "seasonality_header": "📅 Monthly Seasonality Cycle & Index",
        "download_seasonality": "📥 Download Seasonality CSV",
        "advisory_header": "💡 Institutional Agro-Economic Advisory",
        "generate_advisory_btn": "🔄 Generate Market Advisory",
        "advisory_box_title": "📋 Agro-Economic Intelligence & Strategic Recommendations",
        "disclaimer_title": "⚖️ Data Governance & Limitations Notice",
        "disclaimer_1": "1. Data Provenance & Scope: Data sourced from Mendeley AgriPriceBD (Muhammad et al., 2026, CC BY 4.0).",
        "disclaimer_2": "2. Retail vs. Farmgate Price Gap: Reported mid-prices represent urban and divisional retail averages (BDT/kg). Farmgate prices received by smallholder farmers are typically 15% to 30% lower due to intermediary margins and transport logistics.",
        "disclaimer_3": "3. Unpredictable External Shocks: Unseasonal floods, flash droughts, sudden import tariff modifications, or border trade disruptions cannot be predicted from historical time series alone. Use the 80% confidence corridor as a risk indicator.",
    },
}


@st.cache_data(show_spinner=False)
def get_cached_series(crop: str) -> pd.Series:
    """Load and cache contiguous retail mid-price series for a crop."""
    return load_series(crop)


@st.cache_data(show_spinner=False)
def get_cached_evaluation(crop: str, horizon: int) -> Dict[str, Any]:
    """Cache walk-forward model evaluation metrics."""
    return evaluate(crop, horizon=horizon, n_origins=15)


# Landing Commodity Selection Popup Dialog (Streamlit native @st.dialog)
@st.dialog("🌾 কৃষিপণ্য নির্বাচন / Commodity Selection", width="large")
def show_crop_selection_dialog(lang: str = "bn") -> None:
    """Popup modal prompting user to select which agricultural commodity to explore."""
    if lang == "en":
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.2rem;">
                <h3 style="color: #38bdf8; margin-bottom: 4px;">Which commodity's market status would you like to explore?</h3>
                <p style="color: #94a3b8; font-size: 0.95rem;">Select one of Bangladesh's key agricultural staples to generate real-time forecasts and AI insights:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 1.2rem;">
                <h3 style="color: #38bdf8; margin-bottom: 4px;">আপনি কোন কৃষিপণ্যের বাজার অবস্থা জানতে চান?</h3>
                <p style="color: #94a3b8; font-size: 0.95rem;">পূর্বাভাস ও প্রাতিষ্ঠানিক বিশ্লেষণ দেখার জন্য নিচের যেকোনো একটি ফসল নির্বাচন করুন:</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5 Interactive Crop Selection Cards
    cols = st.columns(len(CROPS))
    for i, c in enumerate(CROPS):
        info = get_crop_info(c)
        icon = info.get("icon", "🌾")
        bn_title = info["bn_name"]
        en_title = info["en_name"]
        label = f"{icon}\n\n**{bn_title}**\n\n*{en_title}*" if lang == "bn" else f"{icon}\n\n**{en_title}**\n\n*{bn_title}*"

        with cols[i]:
            if st.button(label, key=f"popup_crop_{c}", use_container_width=True):
                st.session_state["selected_crop"] = c
                st.session_state["has_chosen_crop"] = True
                st.rerun()


def main() -> None:
    # Initialize Session States
    if "lang" not in st.session_state:
        st.session_state["lang"] = "bn"
    if "selected_crop" not in st.session_state:
        st.session_state["selected_crop"] = "garlic"
    if "has_chosen_crop" not in st.session_state:
        st.session_state["has_chosen_crop"] = False

    # Check if first-landing popup needs to be triggered
    if not st.session_state["has_chosen_crop"]:
        show_crop_selection_dialog(lang=st.session_state["lang"])

    # Language Switcher in Sidebar Top
    with st.sidebar:
        st.markdown(f"### {TEXTS[st.session_state['lang']]['control_panel']}")
        chosen_lang = st.radio(
            TEXTS[st.session_state["lang"]]["language_label"],
            options=["বাংলা", "English"],
            index=0 if st.session_state["lang"] == "bn" else 1,
            horizontal=True,
        )
        new_lang = "bn" if chosen_lang == "বাংলা" else "en"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            st.rerun()

        T = TEXTS[st.session_state["lang"]]

        # Commodity selector
        selected_crop = st.selectbox(
            T["commodity_selection"],
            CROPS,
            index=CROPS.index(st.session_state["selected_crop"]),
            format_func=lambda c: get_crop_display_label(c, lang=st.session_state["lang"]),
            help="Select one of the 5 tracked staple commodities.",
        )
        if selected_crop != st.session_state["selected_crop"]:
            st.session_state["selected_crop"] = selected_crop

        # Button to re-open the crop selection popup modal
        if st.button(T["change_crop_btn"], use_container_width=True):
            show_crop_selection_dialog(lang=st.session_state["lang"])

        horizon = st.slider(
            T["forecast_horizon"],
            min_value=7,
            max_value=30,
            value=14,
            step=1,
            help="Determine the prediction horizon (7 to 30 days).",
        )

        # Agronomic profile card (bilingual)
        crop_info = get_crop_info(st.session_state["selected_crop"])
        cat_text = crop_info["category_bn"] if st.session_state["lang"] == "bn" else crop_info["category_en"]
        season_text = crop_info["season_bn"] if st.session_state["lang"] == "bn" else crop_info["season_en"]
        vol_text = crop_info["volatility_bn"] if st.session_state["lang"] == "bn" else crop_info["volatility_en"]
        dist_text = crop_info["key_districts_bn"] if st.session_state["lang"] == "bn" else crop_info["key_districts_en"]
        storage_text = crop_info["storage_tip_bn"] if st.session_state["lang"] == "bn" else crop_info["storage_tip_en"]

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-title">{T['agronomic_title']}</div>
                <div class="sidebar-item"><strong>{T['category']}</strong> {cat_text}</div>
                <div class="sidebar-item"><strong>{T['season']}</strong> {season_text}</div>
                <div class="sidebar-item"><strong>{T['volatility']}</strong> {vol_text}</div>
                <div class="sidebar-item"><strong>{T['districts']}</strong> {dist_text}</div>
                <div class="sidebar-item" style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <strong style="color: #34d399;">{T['storage']}</strong><br>{storage_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        api_configured = bool(get_api_key())
        if api_configured:
            st.success(T["ai_connected"])
        else:
            st.info(T["offline_active"])

    # Active Language Repository
    T = TEXTS[st.session_state["lang"]]
    cur_crop = st.session_state["selected_crop"]
    crop_info = get_crop_info(cur_crop)

    # Main Dashboard Header
    st.markdown(
        f"""
        <div style="margin-bottom: 1.2rem;">
            <h1 style="margin: 0; padding: 0; font-size: 2.1rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center;">
                {T['title']} <span class="badge-congress">{T['badge']}</span>
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.02rem;">
                {T['subtitle']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ingestion & Model Computation
    with st.spinner("Analyzing time-series price data and fitting models..." if st.session_state["lang"] == "en" else "উপাত্ত বিশ্লেষণ এবং পূর্বাভাস মডেল পরিচালিত হচ্ছে..."):
        try:
            history = get_cached_series(cur_crop)
            summary = seasonal_analysis(history)
            forecast, lower, upper, meta = arima_forecast(history, horizon)
            eval_metrics = get_cached_evaluation(cur_crop, horizon)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error(f"⚠️ Error during data modeling: {exc}")
            st.stop()

    # KPI Summary Cards (Custom styled theme-resilient HTML grid)
    delta_daily = history.iloc[-1] - history.iloc[-2]
    delta_class = "metric-delta-pos" if delta_daily > 0 else ("metric-delta-neg" if delta_daily < 0 else "metric-delta-neutral")
    delta_sign = "+" if delta_daily > 0 else ""

    trend_pct = summary["recent_trend_pct"]
    trend_class = "metric-delta-pos" if trend_pct > 0 else ("metric-delta-neg" if trend_pct < 0 else "metric-delta-neutral")
    trend_sign = "+" if trend_pct > 0 else ""

    peak_display = (
        f"{summary['peak_month']} ({summary['peak_month_bn']})"
        if st.session_state["lang"] == "en"
        else f"{summary['peak_month_bn']} ({summary['peak_month'][:3]})"
    )

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">{T['latest_price']}</div>
                <div class="metric-value">{history.iloc[-1]:.2f} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">BDT/kg</span></div>
                <div class="{delta_class}">{T['daily_change']}: {delta_sign}{delta_daily:.2f} BDT</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{T['historical_mean']}</div>
                <div class="metric-value">{summary['overall_average']:.2f} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">BDT/kg</span></div>
                <div class="metric-delta-neutral">{T['all_data_mean']}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{T['peak_season']}</div>
                <div class="metric-value">{peak_display}</div>
                <div class="metric-delta-pos">{T['historical_avg']} {summary['peak_average']:.1f} BDT/kg</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">{T['momentum_90d']}</div>
                <div class="metric-value">{trend_sign}{trend_pct:.1f}%</div>
                <div class="{trend_class}">{T['momentum_sub']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Model Governance & Baseline Comparison Banner
    rec_model = eval_metrics["recommended_model"]
    if st.session_state["lang"] == "en":
        if rec_model == "ARIMA":
            gov_icon = "📈"
            gov_message = (
                f"<strong>Model Governance & Empirical Evaluation:</strong> Across {eval_metrics['n_origins']} walk-forward test origins, "
                f"the <strong>ARIMA(1,1,1) model outperforms</strong> the persistence baseline by "
                f"<strong>{eval_metrics['diff_pct']}% lower error</strong> (MAE: <code>{eval_metrics['arima_mae']} BDT/kg</code> vs. <code>{eval_metrics['naive_mae']} BDT/kg</code>)."
            )
        else:
            gov_icon = "📊"
            gov_message = (
                f"<strong>Model Governance & Empirical Evaluation:</strong> Due to sharp short-term price fluctuations in this commodity, "
                f"the <strong>Naive Persistence Baseline</strong> yielded <strong>{eval_metrics['diff_pct']}% lower error</strong> "
                f"(MAE: <code>{eval_metrics['naive_mae']} BDT/kg</code> vs. <code>{eval_metrics['arima_mae']} BDT/kg</code>) and is presented as the primary benchmark."
            )
    else:
        if rec_model == "ARIMA":
            gov_icon = "📈"
            gov_message = (
                f"<strong>মডেল মূল্যায়ন ও নির্ভরযোগ্যতা যাচাই (Model Governance):</strong> "
                f"বিগত {eval_metrics['n_origins']}টি ওয়াক-ফরোয়ার্ড মূল্যায়ন বিন্দুতে ARIMA(1,1,1) মডেলটি "
                f"সরল নাইভ বেসলাইনের তুলনায় <strong>{eval_metrics['diff_pct']}% উচ্চতর নির্ভুলতা</strong> প্রদর্শন করেছে "
                f"(গড় ত্রুটি MAE: <code>{eval_metrics['arima_mae']} BDT/kg</code> বনাম <code>{eval_metrics['naive_mae']} BDT/kg</code>)।"
            )
        else:
            gov_icon = "📊"
            gov_message = (
                f"<strong>মডেল মূল্যায়ন ও নির্ভরযোগ্যতা যাচাই (Model Governance):</strong> "
                f"উচ্চ স্বল্পমেয়াদি বাজার অস্থিরতার কারণে এই ফসলে সরল নাইভ বেসলাইন পদ্ধতি "
                f"ARIMA-এর চেয়ে <strong>{eval_metrics['diff_pct']}% কম গড় ত্রুটি</strong> "
                f"(MAE: <code>{eval_metrics['naive_mae']} BDT/kg</code> বনাম <code>{eval_metrics['arima_mae']} BDT/kg</code>) বজায় রেখেছে।"
            )

    st.markdown(
        f"""
        <div class="governance-card">
            <span style="font-size: 1.4rem;">{gov_icon}</span>
            <div class="governance-text">{gov_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Forecast Section Title
    crop_header_name = f"{crop_info['en_name']} ({crop_info['bn_name']})" if st.session_state["lang"] == "en" else f"{crop_info['bn_name']} ({crop_info['en_name']})"
    st.subheader(f"📈 {crop_header_name} — {T['chart_title'].format(horizon=horizon)}")

    forecast_index = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    # Prepare DataFrame for Line Chart
    chart_history = history.tail(365)
    chart_df = pd.DataFrame(
        {
            T["chart_hist"]: chart_history,
            T["chart_forecast"]: pd.Series(forecast, index=forecast_index),
            T["chart_lower"]: pd.Series(lower, index=forecast_index),
            T["chart_upper"]: pd.Series(upper, index=forecast_index),
        }
    )
    st.line_chart(chart_df, color=["#38bdf8", "#34d399", "#f59e0b", "#ef4444"])

    # Forecast Details Table & CSV Export
    date_col = "Date" if st.session_state["lang"] == "en" else "তারিখ (Date)"
    fc_col = "Forecast (BDT/kg)" if st.session_state["lang"] == "en" else "প্রাক্কলিত দর (Forecast BDT/kg)"
    lower_col = "Lower 80% CI" if st.session_state["lang"] == "en" else "৮০% নিম্ন আস্থার সীমা (Lower CI)"
    upper_col = "Upper 80% CI" if st.session_state["lang"] == "en" else "৮০% উচ্চ আস্থার সীমা (Upper CI)"
    model_col = "Algorithm" if st.session_state["lang"] == "en" else "ব্যবহৃত মডেল (Algorithm)"

    forecast_table = pd.DataFrame(
        {
            date_col: forecast_index.strftime("%Y-%m-%d"),
            fc_col: np.round(forecast, 2),
            lower_col: np.round(lower, 2),
            upper_col: np.round(upper, 2),
            model_col: meta["model_name"],
        }
    )

    with st.expander(T["expander_table"]):
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        csv_forecast = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=T["download_forecast"],
            data=csv_forecast,
            file_name=f"{cur_crop}_forecast_{horizon}d.csv",
            mime="text/csv",
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Seasonality Analysis & Executive Advisory
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader(T["seasonality_header"])
        seasonal_df = pd.DataFrame(summary["monthly"])

        if st.session_state["lang"] == "en":
            seasonal_display = seasonal_df.set_index("month_name")[
                ["month_name_bn", "average_price", "minimum_price", "maximum_price", "seasonal_index", "observations"]
            ].rename(
                columns={
                    "month_name_bn": "Bengali Month",
                    "average_price": "Mean Price (BDT/kg)",
                    "minimum_price": "Min Price",
                    "maximum_price": "Max Price",
                    "seasonal_index": "Seasonality Index",
                    "observations": "Observations",
                }
            )
        else:
            seasonal_display = seasonal_df.set_index("month_name_bn")[
                ["month_name", "average_price", "minimum_price", "maximum_price", "seasonal_index", "observations"]
            ].rename(
                columns={
                    "month_name": "মাস (ইংরেজি)",
                    "average_price": "গড় দর (BDT/kg)",
                    "minimum_price": "সর্বনিম্ন দর",
                    "maximum_price": "সর্বোচ্চ দর",
                    "seasonal_index": "মৌসুমি সূচক (Index)",
                    "observations": "পর্যবেক্ষণ সংখ্যা",
                }
            )

        st.dataframe(seasonal_display, use_container_width=True)

        csv_seasonal = seasonal_display.to_csv(encoding="utf-8").encode("utf-8")
        st.download_button(
            label=T["download_seasonality"],
            data=csv_seasonal,
            file_name=f"{cur_crop}_seasonal_index.csv",
            mime="text/csv",
        )

    with right_col:
        st.subheader(T["advisory_header"])
        advice_key = f"advice_{cur_crop}_{st.session_state['lang']}"

        if st.button(T["generate_advisory_btn"], type="primary"):
            with st.spinner("Formulating market advisory..." if st.session_state["lang"] == "en" else "কৃষি অর্থনীতি ও বাজার উপাত্ত বিশ্লেষণ করা হচ্ছে..."):
                try:
                    advice_text = llm_advice(cur_crop, summary, lang=st.session_state["lang"])
                    st.session_state[advice_key] = advice_text
                except Exception:
                    st.session_state[advice_key] = fallback_advice(cur_crop, summary, lang=st.session_state["lang"])

        if advice_key not in st.session_state:
            st.session_state[advice_key] = fallback_advice(cur_crop, summary, lang=st.session_state["lang"])

        current_advice = st.session_state[advice_key]
        st.markdown(
            f"""
            <div class="advisory-container">
                <div class="advisory-header">{T['advisory_box_title']}</div>
                <div style="white-space: pre-line;">{current_advice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Responsible AI & Historical Data Limitations Notice
    st.markdown(
        f"""
        <div class="disclaimer-container">
            <div class="disclaimer-title">{T['disclaimer_title']}</div>
            <div style="margin-top: 6px;">
                <strong>{T['disclaimer_1']}</strong> Scope: <span class="badge-date">{summary['data_start']}</span> to <span class="badge-date">{summary['data_end']}</span>.<br>
                <strong>{T['disclaimer_2']}</strong><br>
                <strong>{T['disclaimer_3']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()