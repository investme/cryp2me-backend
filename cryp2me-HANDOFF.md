# cryp2me.ai — Project Handoff & Runbook

**AI crypto + tokenized gold prediction platform under C50 Labs.**
Last updated: May 28, 2026

---

## 1. What This Project Is

An AI-powered web platform that predicts price direction for ~31 crypto assets plus tokenized gold (PAXG). It combines two engines:

- **AI models** — LSTM + Transformer + MetaLearner ensemble, trained per-timeframe. Directional accuracy ~52–56% (honest market-realistic numbers).
- **Statistical signals** — linear regression channels, mean reversion z-scores, volatility regimes, support/resistance, momentum, and a confluence score that fires high-conviction signals only when AI + stats agree.

The product sells access to the dashboard: a free tier (BTC/ETH/SOL, 1H/1D timeframes) and a Pro tier ($39.99/mo or $335.99/yr) unlocking all assets, all 6 timeframes, gold, confluence signals, and drawing tools.

---

## 2. Architecture (Everything on Render, One Origin)

```
cryp2me.ai (target domain, not yet connected)
└── Render web service  →  cryp2me-backend-1.onrender.com
      /            landing page      (static/index.html)
      /app         trading dashboard (static/app/index.html)
      /legal       legal & contact   (static/legal/index.html)
      /api/*       FastAPI endpoints
      /lightweight-charts.js  charting library used by dashboard
```

Everything is served from one origin on purpose, so the browser `localStorage` Pro flag is shared between the landing page and the dashboard.

**Stack:** FastAPI (Python 3.12) backend, ONNX Runtime for inference, plain HTML/CSS/JS frontend (no framework), live market data pulled from Binance public API (no data stored server-side).

---

## 3. The Two Machines

| Machine | Purpose | Path |
|---------|---------|------|
| **GPU station** (RTX 3070) | Train models | `~/Desktop/cryp2me-Claude/cryp2me-phase5-final/phase5-final` |
| **Backend repo** (either PC) | Run/deploy backend | `~/Desktop/cryp2me-Claude/cryp2me-backend-phase5/cryp2me-backend` |

Training happens on the GPU station. The backend is a separate folder that gets pushed to GitHub → auto-deploys to Render.

---

## 4. Key Repos & Services

- **GitHub backend repo:** `github.com/investme/cryp2me-backend` (branch `main`)
- **Render service:** `cryp2me-backend-1.onrender.com` (region Frankfurt, free tier, persistent disk for models)
- **GitHub Release** holding the ONNX models: tag `models-v1`, asset `onnx-models.tar.gz` (~75MB)
- **Domains owned (Namecheap):** cryp2me.ai, cryp2me.io, cryp2me.org

---

## 5. How To Run It Locally

```bash
cd <backend folder>
source venv/Scripts/activate          # Windows Git Bash
uvicorn app.main:app --reload --port 8000
```

Then open:
- http://localhost:8000/         (landing)
- http://localhost:8000/app      (dashboard)
- http://localhost:8000/legal    (legal)
- http://localhost:8000/api/health   (should return models_loaded: true)

Models must exist at `models/onnx/{1m,5m,15m,1h,2h,1d}/` (each folder has lstm.onnx, transformer.onnx, meta_learner.onnx).

---

## 6. How To Deploy a Change

```bash
cd <backend folder>
git add -A
git commit -m "describe change"
git push origin main
```

Render auto-deploys on every push to `main`. Build takes ~3 minutes. Watch the Logs tab; wait for "Your service is live".

**Note:** The ONNX models are NOT in git (they're in `.gitignore`). They live on Render's persistent disk, uploaded once via the GitHub Release. They survive redeploys. Only re-upload if you retrain (see section 8).

---

## 7. How The Code Is Organized (backend)

```
app/
  main.py              app setup, mounts static/ at /, loads models on startup
  state.py             singleton inference engine
  services/
    inference.py       MTF ONNX engine + feature builder (22 features!)
    stat_signals.py    quant signals (channels, mean rev, volatility, confluence)
    backtester.py      walk-forward backtest
    binance.py         live candle fetching
  routers/
    predict.py         /api/predict/{ticker}?interval=  (main endpoint)
    backtest.py        /api/backtest/{ticker}
    chart.py, search.py
static/
  index.html           landing page
  app/index.html       dashboard
  legal/index.html     legal page
  lightweight-charts.js
models/onnx/{tf}/       ONNX models (on Render disk, not in git)
```

**Critical:** feature count is **22**, not 35. `inference.py` `N_FEATURES = 22` must match the trained models exactly, or the AI silently falls back to stub mode.

---

## 8. How To Retrain / Add Assets

On the GPU station:
```bash
cd <phase5-final folder>
source venv/Scripts/activate
bash run_mtf.sh        # option 2 = single timeframe (safer with power cuts)
```

To add assets: edit `TICKERS_FULL` in `configs/timeframes.py` and `ASSET_NAMES` in the dashboard. ONNX size stays ~75MB regardless of asset count (models are shared across all assets; backend pulls each asset live from Binance).

After retraining, copy new ONNX to backend `models/onnx/{tf}/`, re-tar, upload as a new GitHub Release, and download into Render Shell:
```bash
cd /opt/render/project/src/models
curl -L -o onnx-models.tar.gz <release-url>
tar -xzf onnx-models.tar.gz && rm onnx-models.tar.gz
```
Then restart the Render service.

---

## 9. Access Control (Pro Gating)

- Free vs Pro is enforced client-side via `localStorage` keys `cryp2me_pro` and `cryp2me_pro_until`.
- Free tier: BTC/ETH/SOL only, 1H/1D timeframes only.
- Promo codes (in the landing page `PROMO_CODES` object) grant N days free, skipping payment.
- **Known limitation:** client-side gating can be bypassed by a savvy user via dev tools. Before scaling paid users, build real server-side auth + subscription validation (the backend should refuse to return Pro data without a valid token).

---

## 10. Outstanding TODO (to fully launch)

| # | Task | Notes |
|---|------|-------|
| 1 | Connect cryp2me.ai → Render | Add custom domain in Render, set Namecheap DNS (CNAME/A records) |
| 2 | Set up support@cryp2me.ai | Namecheap email forwarding → personal inbox |
| 3 | Create Arbitrum wallet | Replace `WALLET_ADDRESS` placeholder in landing page for crypto payments |
| 4 | Set up Stripe account | Replace `STRIPE_KEY` placeholder; wire real Stripe checkout (currently simulated) |
| 5 | Wire real crypto payment verification | Currently simulated with a timeout |
| 6 | Build server-side auth | Replace localStorage gating with real subscriptions before scaling |
| 7 | (Optional) Retrain 1D model on 5yr data | Marginal gain; captures more market cycles |

---

## 11. Honest Notes / Philosophy

- The platform is **honest by design** — it shows real ~52–56% AI accuracy and uses statistical confluence rather than fake "70% accurate" claims. Keep it that way; it's the differentiator and the integrity of the product.
- Predictions are **probabilistic, not guarantees**. The legal page and disclaimers reflect this.
- When AI + mean reversion + momentum + trend all agree, confluence fires STRONG — that alignment is the actual sellable signal.

---

*Built by Hussein Matar / C50 Labs. This runbook is a starting point — keep it updated as the project evolves.*
