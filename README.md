# 🌾 AgroMind AI (অ্যাগ্রোমাইন্ড এআই)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.42+-FF4B4B.svg)](https://streamlit.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()
[![Congress 2026](https://img.shields.io/badge/International%20AI%20Congress-2026%20Candidate-green.svg)]()

> **AgroMind AI** is a resilient, farmer-facing agricultural price intelligence and decision-support system built for Bangladesh. Developed for the **International AI Congress 2026**, the platform translates multi-year econometric retail price data into actionable seasonal insights, statistical forecasts with 80% confidence bands, and culturally grounded Bengali market advisories powered by Google Gemini 2.0 Flash (with a 100% offline deterministic fallback).

---

## 📌 Key Capabilities

- **Bilingual & Farmer-Centric UI:** Tailored specifically for Bangladeshi smallholders and extension workers with authentic Bengali terminology (`টাকা/কেজি`, `মৌসুমি সূচক`, `রবি/খরিফ ফসল`).
- **Data-Driven Forecast Engine (`ARIMA(1,1,1)`):** Forecasts 7 to 30 days ahead with empirical 80% confidence corridors.
- **Scientific Model Governance & Benchmarking:** Automatically tests ARIMA against a persistence Naive baseline across walk-forward historical origins, transparently displaying out-of-sample MAE scores.
- **Calendar Seasonality Decomposition:** Identifies historical peak and trough months, seasonal index ratios, and recent 90-day momentum.
- **Hybrid Intelligence Architecture:** Integrates Google Gemini 2.0 Flash via secure HTTP headers (`x-goog-api-key`) with transient retry logic, alongside a local deterministic Bengali fallback that works completely without internet or API keys.
- **Field Data Export:** One-click CSV export for both forecast intervals and monthly seasonal indicators.
- **Fully Tested & Containerized:** 26 automated unit and smoke tests; Docker and Streamlit Cloud ready.

---

## 📊 Data Source & Provenance

- **Dataset:** **AgriPriceBD** (Muhammad et al., 2026), Mendeley Data.
- **DOI:** [10.17632/bkmxnrn3hn.1](https://doi.org/10.17632/bkmxnrn3hn.1)
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Crops Analyzed:**
  - 🧄 Garlic (দেশি রসুন) — Spices
  - 🧆 Chickpea (ছোলা / বুট) — Pulses
  - 🌶️ Green Chilli (কাঁচামরিচ) — Vegetable/Spice
  - 🥒 Cucumber (শসা ও খিরা) — Salad Vegetable
  - 🎃 Sweet Pumpkin (মিষ্টি কুমড়া) — Cucurbits
- **Location:** Source CSVs are located in `data/raw/` and treated as immutable reference datasets.

---

## 🧮 Methodology & Forecast Interpretation

### 1. Data Ingestion & Cleansing
- Daily mid-prices are computed as:
  $$\text{Mid Price} = \frac{\text{Retail Price Minimum} + \text{Retail Price Maximum}}{2}$$
- Reporting portal zero-price artifacts are treated as missing values and imputed via daily reindexing with forward and backward fill (`ffill().bfill()`).

### 2. Time-Series Forecasting
- Fits an autoregressive integrated moving average model $\text{ARIMA}(1,1,1)$ on the most recent 730 days (2 years):
  $$\Delta y_t = c + \phi_1 \Delta y_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
- Produces 80% confidence bounds based on forecast standard error ($z_{0.80} \approx 1.282$).
- **Fail-Safe Fallback:** If series length is below 30 days or ARIMA fails to converge, the system automatically falls back to a persistence Naive baseline without crashing.

### 3. Model Evaluation (Walk-Forward MAE)
- Evaluates out-of-sample performance over rolling origins:
  $$\text{MAE} = \frac{1}{H} \sum_{h=1}^H |\hat{y}_{t+h} - y_{t+h}|$$
- Real validation results on AgriPriceBD:
  - **Garlic:** ARIMA MAE 11.42 vs Naive 11.81 (ARIMA wins)
  - **Chickpea:** ARIMA MAE 1.68 vs Naive 1.73 (ARIMA wins)
  - **Green Chilli:** ARIMA MAE 23.81 vs Naive 25.42 (ARIMA wins)
  - **Cucumber:** Naive MAE 7.33 vs ARIMA 7.87 (Naive wins due to sharp seasonal volatility)
  - **Sweet Pumpkin:** Naive MAE 1.99 vs ARIMA 2.88 (Naive wins)

---

## ⚠️ Important Limitations & Ethical Notice

1. **Retail vs. Farmgate Price Gap:** The AgriPriceBD dataset tracks urban and divisional retail market prices. Due to transportation costs, intermediary fees, and grading differences, prices received by smallholder farmers at the farmgate are typically **15% to 30% lower** than retail figures.
2. **Unpredictable Shocks:** Statistical models cannot predict unseasonal flooding, flash droughts, border trade embargoes, or sudden import tariff adjustments. Farmers should use the 80% confidence intervals as an indicator of potential price variance rather than a deterministic guarantee.

---

## 🚀 Quickstart & Installation

### Option 1: Local Virtual Environment

```powershell
# 1. Clone repository and navigate to folder
cd "c:\Users\sajib\OneDrive\Desktop\AgroMind AI"

# 2. (Optional) Set up virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Gemini API Key
copy .env.example .env
# Edit .env and enter GEMINI_API_KEY=your_key_here

# 5. Run test suite
python -m unittest discover -s tests -p "test_*.py"

# 6. Start AgroMind AI
streamlit run app.py
```

### Option 2: Docker Container

```bash
docker build -t agromind-ai .
docker run -p 8501:8501 -e GEMINI_API_KEY="your_key" agromind-ai
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing & Verification

AgroMind AI includes a comprehensive automated test suite with 26 tests covering validation, ARIMA forecasting, synthetic seasonality detection, deterministic advisory, and mocked Gemini API scenarios (success, 429 rate limit, 500 server error, timeout):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

All 26 tests run in under 4 seconds.

---

## 📂 Repository Structure

```
AgroMind AI/
├── app.py                  # Responsive Streamlit frontend dashboard
├── backend/
│   ├── __init__.py         # Package initialization
│   ├── forecast.py         # CSV cleaning, ARIMA(1,1,1), Naive baseline, evaluation
│   ├── llm.py              # Seasonality engine, Gemini 2.0 Flash & fallback advisory
│   └── market.py           # Crop metadata, Bengali translations, agronomic profiles
├── data/
│   ├── raw/                # AgriPriceBD CSV files (Garlic, Chickpea, Chilli, etc.)
│   └── processed/          # Processed outputs directory
├── docs/
│   ├── architecture.md     # System design & mathematical formulation
│   ├── deployment.md       # Multi-platform deployment instructions
│   └── congress_presentation.md # 5-minute presentation script & slides
├── tests/
│   ├── __init__.py         # Test package initialization
│   ├── test_forecast.py    # Forecast validation, ARIMA & persistence tests
│   ├── test_llm.py         # Seasonality math & mocked Gemini LLM tests
│   ├── test_market.py      # Crop metadata and label tests
│   └── test_app_smoke.py   # Streamlit smoke & caching tests
├── checks.md               # International AI Congress 2026 Checklist & Dossier
├── requirements.txt        # Pinned production dependencies
├── Dockerfile              # Container deployment definition
├── .dockerignore           # Container build ignore rules
├── .env.example            # Safe environment template
├── .gitignore              # Git ignore rules
└── README.md               # User guide and project documentation
```

---

## 🔒 Security & Privacy

- **Header-Based Authentication:** Gemini API keys are passed strictly via HTTP `x-goog-api-key` headers, preventing credential exposure in proxy logs and URL histories.
- **Zero Data Leakage:** No personal farmer data or raw CSV rows are ever transmitted to external APIs; only anonymous aggregate statistics are provided to the LLM.
- **Fail-Safe Offline Mode:** If no API key is provided or network is down, the system continues operating with 100% functionality via deterministic rule-based advice.