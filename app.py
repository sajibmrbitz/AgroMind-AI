"""Streamlit frontend dashboard for AgroMind AI.

International AI Congress 2026 Submission Edition.
Provides professional Bengali agricultural price intelligence, ARIMA forecasting with
confidence intervals benchmarked against persistence baselines, calendar seasonality,
Gemini 2.0 Flash / local deterministic advisory, and CSV data export.
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
    page_title="AgroMind AI — কৃষিপণ্যের বাজারদর ও মৌসুমি বিশ্লেষণ",
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


@st.cache_data(show_spinner=False)
def get_cached_series(crop: str) -> pd.Series:
    """Load and cache contiguous retail mid-price series for a crop."""
    return load_series(crop)


@st.cache_data(show_spinner=False)
def get_cached_evaluation(crop: str, horizon: int) -> Dict[str, Any]:
    """Cache walk-forward model evaluation metrics."""
    return evaluate(crop, horizon=horizon, n_origins=15)


def main() -> None:
    # Sidebar: Parameter controls and agronomic context
    with st.sidebar:
        st.markdown("### ⚙️ নিয়ন্ত্রণ প্যানেল (Control Panel)")
        selected_crop = st.selectbox(
            "কৃষিপণ্য নির্বাচন (Commodity Selection):",
            CROPS,
            format_func=get_crop_display_label,
            index=0,
            help="বিশ্লেষণের জন্য বাংলাদেশ বাজারের ৫টি প্রধান কৃষিপণ্যের একটি নির্বাচন করুন।",
        )

        horizon = st.slider(
            "পূর্বাভাস সময়সীমা (Forecast Horizon — দিন):",
            min_value=7,
            max_value=30,
            value=14,
            step=1,
            help="আগামী কত দিনের মূল্য প্রাক্কলন ও আস্থার সীমা দেখতে চান তা নির্ধারণ করুন।",
        )

        # Agronomic profile card
        crop_info = get_crop_info(selected_crop)
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-title">📊 কৃষি-অর্থনৈতিক বৈশিষ্ট্য (Agronomic Profile)</div>
                <div class="sidebar-item"><strong>শ্রেণি:</strong> {crop_info['category']}</div>
                <div class="sidebar-item"><strong>উৎপাদন মৌসুম:</strong> {crop_info['season_bn']}</div>
                <div class="sidebar-item"><strong>বাজার অস্থিরতা:</strong> {crop_info['volatility']}</div>
                <div class="sidebar-item"><strong>প্রধান বাণিজ্যিক অঞ্চল:</strong> {crop_info['key_districts']}</div>
                <div class="sidebar-item" style="margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                    <strong style="color: #34d399;">💡 সংরক্ষণ ও বিপণন নির্দেশিকা:</strong><br>{crop_info['storage_tip']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        api_configured = bool(get_api_key())
        if api_configured:
            st.success("🟢 Gemini 2.0 AI ইঞ্জিন সংযুক্ত")
        else:
            st.info("🟡 অফলাইন পরিসংখ্যানিক পরামর্শক সক্রিয়")

    # Main Dashboard Header
    st.markdown(
        """
        <div style="margin-bottom: 1.2rem;">
            <h1 style="margin: 0; padding: 0; font-size: 2.1rem; font-weight: 700; color: #f8fafc; display: flex; align-items: center;">
                🌾 AgroMind AI <span class="badge-congress">Congress 2026 Edition</span>
            </h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.02rem;">
                বাংলাদেশ কৃষি বাজার কাঠামোর সময়ক্রমিক উপাত্ত বিশ্লেষণ, ARIMA পূর্বাভাস ও প্রাতিষ্ঠানিক সিদ্ধান্ত সহায়তা প্ল্যাটফর্ম
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ingestion & Model Computation
    with st.spinner("উপাত্ত বিশ্লেষণ এবং পূর্বাভাস মডেল পরিচালিত হচ্ছে..."):
        try:
            history = get_cached_series(selected_crop)
            summary = seasonal_analysis(history)
            forecast, lower, upper, meta = arima_forecast(history, horizon)
            eval_metrics = get_cached_evaluation(selected_crop, horizon)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error(f"⚠️ উপাত্ত বিশ্লেষণ প্রক্রিয়ায় ত্রুটি দেখা দিয়েছে: {exc}")
            st.stop()

    # KPI Summary Cards (Custom styled theme-resilient HTML grid)
    delta_daily = history.iloc[-1] - history.iloc[-2]
    delta_class = "metric-delta-pos" if delta_daily > 0 else ("metric-delta-neg" if delta_daily < 0 else "metric-delta-neutral")
    delta_sign = "+" if delta_daily > 0 else ""

    trend_pct = summary["recent_trend_pct"]
    trend_class = "metric-delta-pos" if trend_pct > 0 else ("metric-delta-neg" if trend_pct < 0 else "metric-delta-neutral")
    trend_sign = "+" if trend_pct > 0 else ""

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">সর্বশেষ রেকর্ডকৃত দর (Latest Mid Price)</div>
                <div class="metric-value">{history.iloc[-1]:.2f} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">BDT/kg</span></div>
                <div class="{delta_class}">দৈনিক পরিবর্তন: {delta_sign}{delta_daily:.2f} BDT</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">ঐতিহাসিক গড় মূল্য (Historical Mean)</div>
                <div class="metric-value">{summary['overall_average']:.2f} <span style="font-size: 0.95rem; font-weight: 500; color: #94a3b8;">BDT/kg</span></div>
                <div class="metric-delta-neutral">সমগ্র উপাত্ত সময়কালের গড়</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">বার্ষিক সর্বোচ্চ মূল্য মৌসুম (Peak Season)</div>
                <div class="metric-value">{summary['peak_month_bn']} <span style="font-size: 0.9rem; font-weight: 500; color: #94a3b8;">({summary['peak_month'][:3]})</span></div>
                <div class="metric-delta-pos">ঐতিহাসিক গড়: {summary['peak_average']:.1f} BDT/kg</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">ত্রৈমাসিক মূল্য প্রবণতা (90-Day Momentum)</div>
                <div class="metric-value">{trend_sign}{trend_pct:.1f}%</div>
                <div class="{trend_class}">পূর্ববর্তী ত্রৈমাসিকের তুলনায় পরিবর্তন</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scientific Model Governance & Baseline Comparison Banner
    rec_model = eval_metrics["recommended_model"]
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

    # Forecast Section
    st.subheader(f"📈 {crop_info['bn_name']} ({crop_info['en_name']})-এর দৈনিক বাজারদর ও আগামী {horizon} দিনের পূর্বাভাস")

    forecast_index = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    # Prepare DataFrame for Line Chart
    chart_history = history.tail(365)
    chart_df = pd.DataFrame(
        {
            "ঐতিহাসিক দর (Historical)": chart_history,
            "প্রাক্কলিত দর (ARIMA Forecast)": pd.Series(forecast, index=forecast_index),
            "৮০% নিম্ন আস্থার সীমা (Lower Bound)": pd.Series(lower, index=forecast_index),
            "৮০% উচ্চ আস্থার সীমা (Upper Bound)": pd.Series(upper, index=forecast_index),
        }
    )
    st.line_chart(chart_df, color=["#38bdf8", "#34d399", "#f59e0b", "#ef4444"])

    # Forecast Details Table & CSV Export
    forecast_table = pd.DataFrame(
        {
            "তারিখ (Date)": forecast_index.strftime("%Y-%m-%d"),
            "প্রাক্কলিত দর (Forecast BDT/kg)": np.round(forecast, 2),
            "৮০% নিম্ন আস্থার সীমা (Lower CI)": np.round(lower, 2),
            "৮০% উচ্চ আস্থার সীমা (Upper CI)": np.round(upper, 2),
            "ব্যবহৃত মডেল (Algorithm)": meta["model_name"],
        }
    )

    with st.expander("🔍 পূর্বাভাসের পূর্ণাঙ্গ উপাত্ত সারণি ও রফতানি (Forecast Dataset & Table)"):
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        csv_forecast = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 পূর্বাভাসের উপাত্ত ডাউনলোড করুন (Download Forecast CSV)",
            data=csv_forecast,
            file_name=f"{selected_crop}_forecast_{horizon}d.csv",
            mime="text/csv",
        )

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Seasonality Analysis & Executive Advisory
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📅 মাসভিত্তিক মৌসুমি মূল্য চক্র (Monthly Seasonality Index)")
        seasonal_df = pd.DataFrame(summary["monthly"]).set_index("month_name_bn")
        display_seasonal = seasonal_df[
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
        st.dataframe(display_seasonal, use_container_width=True)

        csv_seasonal = display_seasonal.to_csv(encoding="utf-8").encode("utf-8")
        st.download_button(
            label="📥 মৌসুমি সূচক ডেটাসেট ডাউনলোড (Download Seasonality CSV)",
            data=csv_seasonal,
            file_name=f"{selected_crop}_seasonal_index.csv",
            mime="text/csv",
        )

    with right_col:
        st.subheader("💡 প্রাতিষ্ঠানিক কৃষি-অর্থনৈতিক পরামর্শ (Market Advisory)")
        advice_key = f"advice_{selected_crop}"

        if st.button("🔄 নতুন বাজার পরামর্শ তৈরি করুন (Generate Advisory)", type="primary"):
            with st.spinner("কৃষি অর্থনীতি ও বাজার উপাত্ত বিশ্লেষণ করা হচ্ছে..."):
                try:
                    advice_text = llm_advice(selected_crop, summary)
                    st.session_state[advice_key] = advice_text
                except Exception:
                    st.session_state[advice_key] = fallback_advice(selected_crop, summary)

        if advice_key not in st.session_state:
            st.session_state[advice_key] = fallback_advice(selected_crop, summary)

        # Render custom executive advisory card instead of generic st.info
        current_advice = st.session_state[advice_key]
        st.markdown(
            f"""
            <div class="advisory-container">
                <div class="advisory-header">📋 প্রাতিষ্ঠানিক বাজার গোয়েন্দা ও সিদ্ধান্ত সুপারিশ</div>
                <div style="white-space: pre-line;">{current_advice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Responsible AI & Historical Data Limitations Notice (Dark-theme-resilient styling)
    st.markdown(
        f"""
        <div class="disclaimer-container">
            <div class="disclaimer-title">⚖️ ডেটা গভর্নেন্স ও প্রাতিষ্ঠানিক ব্যবহারের সীমাবদ্ধতা (Data Governance & Limitations Notice)</div>
            <div style="margin-top: 6px;">
                <strong>১. উপাত্তের উৎস ও সময়কাল:</strong> প্রদর্শিত উপাত্ত Mendeley AgriPriceBD (Muhammad et al., 2026, CC BY 4.0) থেকে সংগৃহীত। উপাত্তের ব্যাপ্তি: <span class="badge-date">{summary['data_start']}</span> থেকে <span class="badge-date">{summary['data_end']}</span>।<br>
                <strong>২. খুচরা বনাম খামার প্রান্তিক মূল্য ব্যবধান:</strong> বিশ্লেষিত দর বিভাগীয় ও প্রধান নগর বাজারসমূহের খুচরা মধ্যবর্তী দর (BDT/kg)। মধ্যস্বত্বভোগী ও আড়তদারি কমিশন, পরিবহন ব্যয় এবং বাছাইকরণ অপচয়ের কারণে স্থানীয় খামার প্রান্তিক মূল্য (Farmgate Price) সাধারণত ১৫% থেকে ৩০% কম হতে পারে।<br>
                <strong>৩. অপ্রত্যাশিত বাজার ঝুঁকি:</strong> আকস্মিক বন্যা, অতিবৃষ্টি, আমদানি শুল্কের আকস্মিক পরিবর্তন কিংবা আন্তর্জাতিক সরবরাহ বিধিনিষেধের মতো অপ্রত্যাশিত বাহ্যিক ঝুঁকি ঐতিহাসিক পরিসংখ্যানিক মডেলে প্রক্ষেপণযোগ্য নয়। অতএব ৮০% আস্থার সীমার (Confidence Interval) আলোকেই সতর্কতা অবলম্বনপূর্বক বাণিজ্যিক সিদ্ধান্ত গ্রহণ বাঞ্ছনীয়।
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()