"""Streamlit frontend dashboard for AgroMind AI.

International AI Congress 2026 Submission Edition.
Provides Bengali-first agricultural price intelligence, ARIMA forecasting with
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

# Custom responsive CSS for mobile compatibility and professional typography
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    .disclaimer-box {
        background-color: #fff9db;
        border-left: 5px solid #f59f00;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-top: 1.5rem;
    }
    .badge-congress {
        background-color: #2b8a3e;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        vertical-align: middle;
        margin-left: 8px;
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
        st.markdown("### 🌾 নিয়ন্ত্রণ প্যানেল (Control Panel)")
        selected_crop = st.selectbox(
            "কৃষিপণ্য নির্বাচন করুন (Select Crop):",
            CROPS,
            format_func=get_crop_display_label,
            index=0,
            help="বিশ্লেষণের জন্য বাংলাদেশ বাজারের ৫টি প্রধান পণ্যের একটি নির্বাচন করুন।",
        )

        horizon = st.slider(
            "পূর্বাভাস দিগন্ত (Forecast Horizon in Days):",
            min_value=7,
            max_value=30,
            value=14,
            step=1,
            help="আগামী কত দিনের মূল্য পূর্বাভাস ও আস্থার সীমা দেখতে চান তা নির্ধারণ করুন।",
        )

        # Agronomic profile card
        crop_info = get_crop_info(selected_crop)
        st.markdown("---")
        st.markdown(f"**📊 ফসলের পরিচয় ও বৈশিষ্ট্য:**")
        st.caption(f"**শ্রেণি:** {crop_info['category']}")
        st.caption(f"**চাষের মৌসুম:** {crop_info['season_bn']}")
        st.caption(f"**মূল্য অস্থিরতা:** {crop_info['volatility']}")
        st.caption(f"**প্রধান জেলা:** {crop_info['key_districts']}")
        st.info(f"💡 **সংরক্ষণ টিপস:** {crop_info['storage_tip']}")

        st.markdown("---")
        api_configured = bool(get_api_key())
        if api_configured:
            st.success("🟢 Gemini 2.0 AI ইঞ্জিন সক্রিয়")
        else:
            st.warning("🟡 অফলাইন পরিসংখ্যানগত অ্যালগরিদম সক্রিয়")

    # Main Dashboard Header
    st.markdown(
        '<h1>🌾 AgroMind AI <span class="badge-congress">Congress 2026 Edition</span></h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "বাংলাদেশের ক্ষুদ্র কৃষকদের জন্য ডেটা-চালিত খুচরা মূল্য বুদ্ধিমত্তা, মৌসুমি সূচক এবং স্বল্পমেয়াদি পূর্বাভাস"
    )

    # Ingestion & Model Computation
    with st.spinner("ডেটা বিশ্লেষণ এবং পূর্বাভাস মডেল পরিচালিত হচ্ছে..."):
        try:
            history = get_cached_series(selected_crop)
            summary = seasonal_analysis(history)
            forecast, lower, upper, meta = arima_forecast(history, horizon)
            eval_metrics = get_cached_evaluation(selected_crop, horizon)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            st.error(f"⚠️ ডেটা বিশ্লেষণ প্রক্রিয়ায় ত্রুটি দেখা দিয়েছে: {exc}")
            st.stop()

    # KPI Summary Cards (4-column layout)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="সর্বশেষ খুচরা দর",
            value=f"{history.iloc[-1]:.2f} BDT/kg",
            delta=f"{history.iloc[-1] - history.iloc[-2]:+.2f} BDT",
            help="উৎস ডেটাসেটের সর্বশেষ জানা খুচরা মধ্যবর্তী দর (টাকা/কেজি)।",
        )
    with col2:
        st.metric(
            label="ঐতিহাসিক গড় দর",
            value=f"{summary['overall_average']:.2f} BDT/kg",
            help="সমগ্র ডেটা সময়কালের গড় খুচরা মূল্য।",
        )
    with col3:
        st.metric(
            label="সর্বোচ্চ দামের মাস",
            value=f"{summary['peak_month_bn']} ({summary['peak_month'][:3]})",
            delta=f"গড় {summary['peak_average']:.1f} BDT",
            help="যে মাসে ঐতিহাসিকভাবে খুচরা মূল্য সর্বোচ্চ শিখরে পৌঁছায়।",
        )
    with col4:
        st.metric(
            label="সাম্প্রতিক ৯০ দিনের প্রবণতা",
            value=f"{summary['recent_trend_pct']:+.1f}%",
            delta_color="normal" if abs(summary["recent_trend_pct"]) <= 5 else "inverse",
            help="পূর্ববর্তী ৯০ দিনের তুলনায় সাম্প্রতিক ৯০ দিনের গড়ের শতকরা পরিবর্তন।",
        )

    # Scientific Model Transparency & Baseline Comparison Banner
    with st.container():
        rec_model = eval_metrics["recommended_model"]
        if rec_model == "ARIMA":
            st.success(
                f"📈 **মডেল মূল্যায়ন (Model Benchmark):** এই ফসলে {eval_metrics['n_origins']}টি ওয়াক-ফরোয়ার্ড টেস্টে "
                f"ARIMA(1,1,1) মডেলটি সরল নাইভ বেসলাইনের চেয়ে বেশি নির্ভুল (MAE: {eval_metrics['arima_mae']} BDT/kg বনাম {eval_metrics['naive_mae']} BDT/kg)।"
            )
        else:
            st.info(
                f"📊 **মডেল মূল্যায়ন (Model Benchmark):** উচ্চ বাজার অস্থিরতার কারণে এই ফসলে সরল নাইভ বেসলাইন "
                f"ARIMA-এর চেয়ে {eval_metrics['diff_pct']}% কম গড় ত্রুটি (MAE: {eval_metrics['naive_mae']} BDT/kg বনাম {eval_metrics['arima_mae']} BDT/kg) বজায় রেখেছে।"
            )

    # Forecast Section
    st.subheader(f"📈 {crop_info['bn_name']} ({crop_info['en_name']})-এর দৈনিক দর ও আগামী {horizon} দিনের পূর্বাভাস")

    forecast_index = pd.date_range(
        history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
    )

    # Prepare DataFrame for Line Chart
    chart_history = history.tail(365)
    chart_df = pd.DataFrame(
        {
            "ঐতিহাসিক দাম (Historical)": chart_history,
            "পূর্বাভাস (Forecast)": pd.Series(forecast, index=forecast_index),
            "৮০% নিম্ন সীমা (Lower Bound)": pd.Series(lower, index=forecast_index),
            "৮০% উচ্চ সীমা (Upper Bound)": pd.Series(upper, index=forecast_index),
        }
    )
    st.line_chart(chart_df, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])

    # Forecast Details Table & CSV Export
    forecast_table = pd.DataFrame(
        {
            "তারিখ (Date)": forecast_index.strftime("%Y-%m-%d"),
            "প্রত্যাশিত দর (Forecast BDT/kg)": np.round(forecast, 2),
            "৮০% নিম্ন সীমা (Lower CI)": np.round(lower, 2),
            "৮০% উচ্চ সীমা (Upper CI)": np.round(upper, 2),
            "মডেল (Algorithm)": meta["model_name"],
        }
    )

    with st.expander("🔍 পূর্বাভাসের বিস্তারিত পরিসংখ্যান ও ডাউনলোড (Forecast Data Table)"):
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        csv_forecast = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 পূর্বাভাসের ডেটা ডাউনলোড করুন (Download Forecast CSV)",
            data=csv_forecast,
            file_name=f"{selected_crop}_forecast_{horizon}d.csv",
            mime="text/csv",
        )

    # Seasonality Analysis & Bengali Advisory
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📅 মাসভিত্তিক মৌসুমি সূচক (Calendar Seasonality)")
        seasonal_df = pd.DataFrame(summary["monthly"]).set_index("month_name_bn")
        display_seasonal = seasonal_df[
            ["month_name", "average_price", "minimum_price", "maximum_price", "seasonal_index", "observations"]
        ].rename(
            columns={
                "month_name": "ইংরেজি মাস",
                "average_price": "গড় দর (BDT/kg)",
                "minimum_price": "সর্বনিম্ন দর",
                "maximum_price": "সর্বোচ্চ দর",
                "seasonal_index": "মৌসুমি সূচক",
                "observations": "নমুনার সংখ্যা",
            }
        )
        st.dataframe(display_seasonal, use_container_width=True)

        csv_seasonal = display_seasonal.to_csv(encoding="utf-8").encode("utf-8")
        st.download_button(
            label="📥 মৌসুমি সূচক ডেটা ডাউনলোড করুন (Download Seasonality CSV)",
            data=csv_seasonal,
            file_name=f"{selected_crop}_seasonal_index.csv",
            mime="text/csv",
        )

    with right_col:
        st.subheader("🌾 কৃষক-বান্ধব সিদ্ধান্ত পরামর্শ (Farmer Advisory)")
        advice_key = f"advice_{selected_crop}"

        if st.button("💬 নতুন পরামর্শ তৈরি করুন (Generate Advisory)", type="primary"):
            with st.spinner("কৃষি অর্থনীতি ও পরিসংখ্যানগত বিশ্লেষণ সমন্বয় করা হচ্ছে..."):
                try:
                    advice_text = llm_advice(selected_crop, summary)
                    st.session_state[advice_key] = advice_text
                except Exception as exc:
                    st.session_state[advice_key] = fallback_advice(selected_crop, summary)

        if advice_key in st.session_state:
            st.info(st.session_state[advice_key])
        else:
            default_preview = fallback_advice(selected_crop, summary)
            st.session_state[advice_key] = default_preview
            st.info(default_preview)

    # Responsible AI & Historical Data Limitations Notice
    st.markdown(
        f"""
        <div class="disclaimer-box">
            <strong>⚠️ ঐতিহাসিক ডেটা ও পূর্বাভাসের সীমাবদ্ধতা (Data Governance & Limitations Notice):</strong><br>
            ১. <strong>উৎস ও কভারেজ:</strong> প্রদর্শিত তথ্য AgriPriceBD (Muhammad et al., 2026, Mendeley Data, CC BY 4.0) থেকে সংগৃহীত। সময়কাল: <code>{summary['data_start']}</code> থেকে <code>{summary['data_end']}</code>।<br>
            ২. <strong>খুচরা বনাম খামার মূল্য:</strong> প্রদর্শিত দর বিভাগীয়/শহুরে খুচরা মধ্যবর্তী দর (BDT/kg)। মধ্যস্বত্বভোগী ও পরিবহন খরচের কারণে কৃষক পর্যায়ে খামার প্রান্তের (Farmgate) দর সাধারণ ১৫-৩০% পর্যন্ত কম হতে পারে।<br>
            ৩. <strong>অনিশ্চয়তা:</strong> খরা, অকাল বন্যা কিংবা আমদানি শুল্ক পরিবর্তনের মতো আকস্মিক শক ঐতিহাসিক মডেলে পূর্বাভাসযোগ্য নয়। অতএব ৮০% আস্থার সীমার (Confidence Interval) আলোকে সতর্কতার সাথে বাণিজ্যিক সিদ্ধান্ত গ্রহণ করুন।
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()