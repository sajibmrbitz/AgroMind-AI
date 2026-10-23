# AgroMind AI — International AI Congress 2026 Presentation Dossier

> **Track:** AI for Agricultural Transformation & Developing Economies  
> **Topic:** Democratizing Price Intelligence for Smallholder Farmers in Bangladesh  
> **Speaker/Presenter Guide:** 5-Minute Pitch & Slide Structure

---

## 🎯 1. The Problem (Minute 1)
- **The Reality:** 60%+ of Bangladesh's rural population depends on agriculture, yet smallholders lose up to 30% of potential earnings due to extreme seasonal price volatility.
- **The Information Chasm:** Perishables like green chilli fluctuate between 40 BDT/kg and 300+ BDT/kg. Farmers lack analytical tools and are at the mercy of middlemen.
- **The Language Barrier:** Existing ML tools speak complex statistical English. Farmers need localized Bengali intelligence they can trust.

---

## ⚙️ 2. The Solution: AgroMind AI (Minute 2)
- **What it is:** A dual-engine agricultural intelligence platform combining rigorous econometric time-series modeling with Google Gemini 2.0 Flash and offline fallback heuristics.
- **Key Innovation 1 — Model Governance:** Rather than blindly displaying ARIMA forecasts, AgroMind AI benchmarks ARIMA against a persistence Naive baseline using walk-forward out-of-sample MAE. If a crop is too volatile, the system transparently explains why the persistence baseline is safer.
- **Key Innovation 2 — Hybrid Advisory Engine:** When online, Gemini 2.0 generates structured, respectful Bengali guidance. If the user is in a remote union parishad with zero connectivity or no API key, the deterministic fallback delivers instant offline advice.

---

## 🔬 3. Technical Depth & Scientific Methodology (Minute 3)
- **Dataset:** AgriPriceBD (Muhammad et al., 2026, Mendeley Data, CC BY 4.0).
- **Zero-Price Imputation:** Treats portal zero-value artifacts as missing values, reindexing onto a daily calendar with forward and backward filling.
- **ARIMA Modeling:** Recent 730-day window, order (1,1,1), providing 80% confidence interval bands ($z \approx 1.282$).
- **Seasonality Index:** Decomposes monthly deviations against multi-year overall means to establish calendar peaks and lows.
- **Security & Privacy:** Header-based auth (`x-goog-api-key`), bounded timeout, zero raw row transmission, zero personally identifiable information (PII).

---

## 📱 4. Live Demonstration Walkthrough (Minute 4)
- **Screen 1 — KPI Dashboard:** Latest Retail Price, Overall Average, Peak Month, 90-day Trend.
- **Screen 2 — Forecast Visualizer:** Interactive 365-day trend with 14-day projection and 80% confidence corridor.
- **Screen 3 — Model Governance Badge:** Live walk-forward benchmark (e.g. Garlic: ARIMA MAE 11.42 vs Naive 11.81).
- **Screen 4 — Farmer Advisory:** One-click actionable advice in native Bengali.
- **Screen 5 — Offline CSV Export:** Empowering extension officers to download data for village field work.

---

## 🌍 5. Scalability & Congress Impact (Minute 5)
- **Reproducibility:** 100% automated test suite (26 unit & smoke tests), Dockerized, container-ready.
- **Low Footprint:** Runs on lightweight compute (single vCPU / 512MB RAM).
- **Direct Alignment with UN SDGs:**
  - SDG 1: No Poverty (protecting farmer earnings).
  - SDG 2: Zero Hunger (stabilizing food supply chains).
  - SDG 8: Decent Work and Economic Growth.
  - SDG 10: Reduced Inequalities (eradicating information asymmetry).
