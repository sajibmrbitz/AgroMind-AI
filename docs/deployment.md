# AgroMind AI — Deployment & Operations Guide

> **Prepared for:** International AI Congress 2026 Evaluation Committee  
> **Deployment Targets:** Local Workstation, Docker Container, Streamlit Community Cloud, Hugging Face Spaces, Google Cloud Run

---

## 1. Local Development Deployment

### Prerequisites
- Python 3.10 – 3.12 installed.
- Git (optional).

### Setup Steps
```powershell
# 1. Clone repository (if applicable) and enter directory
cd "c:\Users\sajib\OneDrive\Desktop\AgroMind AI"

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install or verify dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Gemini API Key
copy .env.example .env
# Edit .env and insert GEMINI_API_KEY=your_key_here

# 5. Run automated test suite
python -m unittest discover -s tests -p "test_*.py"

# 6. Launch AgroMind AI dashboard
streamlit run app.py
```

---

## 2. Docker Container Deployment

### Build Image
```bash
docker build -t agromind-ai:congress2026 .
```

### Run Container
```bash
docker run -d -p 8501:8501 --name agromind-app \
  -e GEMINI_API_KEY="your_api_key_here" \
  agromind-ai:congress2026
```
Access the application at `http://localhost:8501`.

---

## 3. Cloud Deployment Options

### Streamlit Community Cloud (Zero-Cost Hosting)
1. Push repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, select branch `main`, and specify `app.py` as the entry point.
4. Under "Advanced settings", add your `GEMINI_API_KEY` as a secret (optional).
5. Click **Deploy**.

### Hugging Face Spaces (Docker or Streamlit SDK)
1. Create a new Space on Hugging Face.
2. Select **Streamlit** SDK.
3. Push files to the Hugging Face git remote.
4. Add `GEMINI_API_KEY` under Space Settings -> Repository Secrets.

### Google Cloud Run (Serverless Production)
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/agromind-ai
gcloud run deploy agromind-ai \
  --image gcr.io/PROJECT_ID/agromind-ai \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY="your_key"
```

---

## 4. Verification & Health Monitoring

To verify service status:
```bash
curl -I http://localhost:8501/_stcore/health
```
Expected output: `HTTP/1.1 200 OK`.
