from fastapi import APIRouter, HTTPException, Query
from app.models import PredictionResponse, PredictionHorizon
from app.services.binance import fetch_candles
from app.services.inference import build_features_from_candles
from app.state import engine
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/{ticker}", response_model=PredictionResponse)
async def get_prediction(ticker: str, interval: str = Query("1h")):
    ticker = ticker.upper().strip()
    try:
        candles = await fetch_candles(ticker, "1h", limit=200)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Fetch error for {ticker}: {e}")
        raise HTTPException(500, "Failed to fetch candle data")

    candle_dicts = [
        {"time": c.time, "open": c.open, "high": c.high,
         "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles
    ]

    features = build_features_from_candles(candle_dicts)
    if features is None:
        raise HTTPException(422, f"Insufficient data for {ticker}")

    result  = engine.predict(features)
    price   = candles[-1].close

    def est_price(conf, hours):
        d = 1 if conf > 0.5 else -1
        m = abs(conf - 0.5) * 2 * 0.02 * (hours / 24)
        return price * (1 + d * m)

    horizons = [
        PredictionHorizon(
            label="T+24h",
            price=est_price(result["conf_24h"], 24),
            change_percent=(est_price(result["conf_24h"], 24) - price) / price * 100,
            confidence=result["conf_24h"] * 100,
        ),
        PredictionHorizon(
            label="T+48h",
            price=est_price(result["conf_48h"], 48),
            change_percent=(est_price(result["conf_48h"], 48) - price) / price * 100,
            confidence=result["conf_48h"] * 100,
        ),
        PredictionHorizon(
            label="T+72h",
            price=est_price(result["conf_72h"], 72),
            change_percent=(est_price(result["conf_72h"], 72) - price) / price * 100,
            confidence=result["conf_72h"] * 100,
        ),
    ]

    return PredictionResponse(
        ticker=ticker,
        signal=result["signal"],
        confidence=result["confidence"] * 100,
        horizons=horizons,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_version="v5.0-onnx" if result["loaded"] else "v0.0-stub",
    )
