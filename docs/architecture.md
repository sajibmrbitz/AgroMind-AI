# AgroMind AI — Architectural Specification

> **Submission for:** International AI Congress 2026  
> **System Name:** AgroMind AI (অ্যাগ্রোমাইন্ড এআই)  
> **Primary Audience:** Smallholder farmers, rural cooperatives, agricultural extension officers in Bangladesh.

---

## 1. System Overview

AgroMind AI is structured around a **hybrid statistical-AI architecture** designed for high reliability, explainability, and deployment in developing economies where internet access and computational resources may be constrained.

```
                      +---------------------------------------+
                      |       User Interface (Streamlit)      |
                      |   Mobile & Desktop Responsive View    |
                      +-------------------+-------------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
+-------------------------------+               +-----------------------------------+
|     Forecasting Engine        |               |       Intelligence Layer          |
|    (backend/forecast.py)      |               |        (backend/llm.py)           |
|-------------------------------|               |-----------------------------------|
| - Daily Contiguous Series     |               | - Calendar Seasonality Index      |
| - ARIMA(1,1,1) + 80% CI       |               | - Deterministic Local Fallback    |
| - Naive Persistence Baseline  |               | - Google Gemini 2.0 Flash (Auth)  |
| - Walk-Forward Evaluation MAE |               | - Transient Retry & Backoff       |
+---------------+---------------+               +-----------------+-----------------+
                |                                                 |
                +-------------------------+-----------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         Agronomic & Market Data       |
                      |         (backend/market.py &          |
                      |        data/raw/ AgriPriceBD)         |
                      +---------------------------------------+
```

---

## 2. Core Modules

### 2.1 Frontend Layer (`app.py`)
- **Philosophy:** Thin UI orchestrator strictly consuming backend abstractions.
- **Key Capabilities:**
  - Dynamic bilingual parameter controls (Crop selection with Bengali phonetic labels, Horizon slider 7–30 days).
  - Responsive layout tailored for mobile field smartphones as well as desktop displays.
  - Interactive price time series plotting (historical prices vs. forecast + 80% CI).
  - Scientific Model Governance Badge (displays whether ARIMA or Naive baseline demonstrated higher historical accuracy for the selected crop).
  - One-click CSV export for both forecast trajectories and calendar seasonality tables.

### 2.2 Forecasting & Evaluation Engine (`backend/forecast.py`)
- **Data Pipeline:**
  - Cleans raw AgriPriceBD data by averaging daily retail minimum and maximum prices.
  - Identifies reporting gaps (portal zeroes) and treats them as missing observations.
  - Reindexes onto contiguous daily calendar dates with forward-fill and back-fill.
- **Mathematical Formulations:**
  - **ARIMA(p, d, q):** Fits an $\text{ARIMA}(1,1,1)$ process on the most recent 730 days (2 years):
    $$\Delta y_t = c + \phi_1 \Delta y_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
    where $\Delta y_t = y_t - y_{t-1}$.
  - **80% Confidence Interval:** Computed using normal critical value $z_{0.80} \approx 1.28155$ scaling with model forecast variance $\sigma_h^2$.
  - **Naive Persistence Baseline:**
    $$\hat{y}_{t+h|t} = y_t$$
    Confidence interval computed empirically from historical difference standard deviation:
    $$\text{CI}_{80\%} = y_t \pm 1.28155 \cdot \sigma_{\Delta} \sqrt{h}$$
  - **Model Governance:** Evaluates out-of-sample Mean Absolute Error (MAE) across 15–20 rolling walk-forward test origins to recommend the optimal model transparently.
  - **Fail-Safe Fallback:** If series length is below 30 days or ARIMA convergence fails, the system automatically defaults to the Naive baseline with diagnostic metadata.

### 2.3 Hybrid Intelligence Layer (`backend/llm.py`)
- **Dual-Mode Operation:**
  - **Mode 1 (Online):** Google Gemini 2.0 Flash generates customized, context-aware Bengali recommendations based on structured aggregate statistics.
  - **Mode 2 (Offline):** Deterministic algorithmic advisory generating expert-validated Bengali farmer recommendations completely without internet or API keys.
- **Security & Privacy:**
  - Authentication via `x-goog-api-key` header (protects API keys from URL and proxy logs).
  - Zero raw CSV row transmission; only compact summary statistics are sent.
  - Transient retry mechanism with exponential backoff and jitter for HTTP 429 and 5xx errors.
  - Bounded request timeout (15s).

### 2.4 Market & Agronomic Metadata (`backend/market.py`)
- Stores verified agronomic classifications:
  - Scientific & Bengali vernacular names.
  - Sowing and harvesting seasons (Rabi, Kharif-1, Kharif-2) in Bangladesh.
  - Volatility index and storage advisories.
  - Key cultivation district mappings.

---

## 3. Data Governance & Ethical Guardrails

1. **Farmgate vs. Retail Margin Disclaimer:** AgriPriceBD records urban/divisional retail mid-prices. Middlemen margins, grading losses, and rural freight mean farmgate prices are typically 15–30% lower. The app explicitly highlights this difference to prevent farmer miscalculations.
2. **No Guaranteed Predictions:** The platform enforces strict disclaimers that forecasts represent historical econometric likelihoods, not guaranteed market outcomes.
3. **Data Provenance:** Mendeley AgriPriceBD (Muhammad et al., 2026), CC BY 4.0 license.
