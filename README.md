# cryp2me-backend

FastAPI backend for cryp2me.ai — crypto data pipeline, indicator computation, and DNN inference.

## Stack

- **FastAPI** + Uvicorn (async Python)
- **httpx** — async HTTP for Binance + CoinGecko
- **NumPy / Pandas** — indicator math
- **Redis** — OHLCV caching (60s TTL)
- **PyTorch + ONNX** — Phase 2 model inference

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — no keys needed for public Binance endpoints

# Run development server
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/chart/{ticker}?interval=1d` | OHLCV + indicators |
| GET | `/api/predict/{ticker}` | DNN price prediction |
| GET | `/api/search?q=BTC` | Ticker autocomplete |
| GET | `/api/health` | Health check |

## Data Sources

| Source | Usage | Cost |
|--------|-------|------|
| Binance REST | Primary OHLCV (all intervals) | Free |
| Binance WebSocket | Real-time 1m/5m (Phase 3) | Free |
| CoinGecko | Fallback + market cap | Free tier |

## Folder Structure

```
app/
  main.py           # FastAPI app, CORS, lifespan
  config.py         # Settings (pydantic-settings)
  models.py         # Pydantic response models
  routers/
    chart.py        # GET /api/chart/{ticker}
    predict.py      # GET /api/predict/{ticker}
    search.py       # GET /api/search
  services/
    binance.py      # Binance API + indicator math
```

## Phase Roadmap

| Phase | Status | Notes |
|-------|--------|-------|
| 1 — Data Pipeline | ✅ In progress | Binance OHLCV + indicators |
| 2 — ML Training | 🔜 Weeks 4-7 | LSTM + Transformer training |
| 3 — Inference | 🔜 Weeks 8-10 | ONNX inference in /predict |
| 4 — Scale | 🔜 Weeks 11-12 | Redis cache, Lambda deploy |
