# AgroMind AI Project Guidelines

## Project Goal

Complete a reliable farmer-facing price intelligence app for Bangladesh:

- Load the supplied AgriPriceBD CSV files from `data/raw/`.
- Show historical retail prices and short-term forecasts.
- Explain calendar-month price seasonality.
- Generate cautious Bengali recommendations using Gemini when configured.
- Remain useful with deterministic local advice when Gemini is unavailable.

## Current Architecture

- `app.py` is the Streamlit frontend and should remain thin.
- `backend/forecast.py` owns CSV loading, price-series preparation, ARIMA forecasts,
  and forecast evaluation.
- `backend/llm.py` owns seasonality calculations, `.env` loading, Gemini requests,
  and the local advice fallback.
- `data/raw/` is the source dataset. Do not mutate source CSVs from the app.
- `README.md` contains the user-facing run instructions.

## What Remains To Complete The Project

Work through these items in order:

1. **Verify the runtime locally**
   - Run `.\.venv\Scripts\python.exe -m py_compile app.py backend\*.py`.
   - Start Streamlit with
     `.\.venv\Scripts\python.exe -m streamlit run app.py`.
   - Test every crop in the sidebar and confirm the chart, table, forecast,
     and advice button work.
   - If the environment is recreated, install from `requirements.txt`.

2. **Add automated tests**
   - Test CSV loading, zero-price handling, date filling, and invalid crop errors.
   - Test seasonal analysis with a small synthetic series whose peak and low
     months are known.
   - Test local fallback advice without making a network request.
   - Mock Gemini responses for success, empty responses, HTTP errors, and timeouts.
   - Test forecast output length and confidence-interval shape.

3. **Improve forecast reliability**
   - Add input validation for crop names and positive forecast horizons.
   - Handle short or insufficient time series with a clear fallback to the naive
     forecast rather than allowing ARIMA to fail ambiguously.
   - Compare ARIMA and naive performance before presenting ARIMA as the primary
     forecast.
   - Display a clear historical-data limitation notice in the UI.

4. **Harden the Gemini integration**
   - Keep all API keys in `.env` or the runtime environment; never commit or
     print them.
   - Confirm the configured Gemini model is available for the selected API.
   - Add retry/backoff only for transient failures and keep a bounded timeout.
   - Do not send raw CSV rows or unnecessary personal data to the API.
   - Keep the deterministic Bengali fallback usable when the API is disabled.

5. **Finish the user experience**
   - Add loading/error states for forecast generation.
   - Make labels and recommendations consistently Bengali and explain BDT/kg.
   - Show forecast dates and confidence intervals clearly.
   - Add a download button for the seasonal table and forecast data.
   - Make the layout usable on a narrow screen.

6. **Document and release**
   - Document the data source, license, methodology, limitations, and forecast
     interpretation in `README.md`.
   - Add a safe `.env.example` containing variable names only.
   - Pin or constrain dependency versions after validating the chosen Python
     version.
   - Run compilation, tests, and a clean Streamlit smoke test before release.

## Security Rules

- Never read, expose, log, or commit the value of `.env` secrets.
- Keep `.env`, `.venv/`, caches, and Streamlit secrets ignored by Git.
- Avoid putting API keys in URLs if the provider supports safer headers; if the
  Gemini API requires a query parameter, do not include it in logs or errors.
- Treat all model output as untrusted text and display it without executing it.

## Change Conventions

- Prefer small, typed, testable functions in `backend/`.
- Reuse existing data-loading and forecast helpers instead of duplicating CSV
  parsing in the frontend.
- Preserve the local fallback path whenever changing external API behavior.
- Surface actionable errors; do not silently return success-shaped defaults.
- Update `README.md` whenever run commands, environment variables, or user-visible
  behavior changes.
