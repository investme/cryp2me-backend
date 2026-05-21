"""
app/routers/predict.py — cryp2me.ai Phase 5 MTF + Statistical Signals
Combines AI predictions with proven quant techniques.
"""

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from app.models import PredictionResponse, PredictionHorizon
from app.services.binance import fetch_candles
from app.services.inference import build_features_from_candles, HORIZON_LABELS
from app.services.stat_signals import full_statistical_analysis
from app.state import engine
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])

BINANCE_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "2h": "2h", "1d": "1d",
}
CANDLE_LIMIT = {
    "1m": 300, "5m": 300, "15m": 300,
    "1h": 300, "2h": 300, "1d": 300,
}


@router.get("/{ticker}")
async def get_prediction(ticker: str, interval: str = Query("1h")):
    ticker   = ticker.upper().strip()
    interval = interval.lower().strip()
    if interval not in BINANCE_INTERVAL_MAP:
        interval = "1h"

    binance_interval = BINANCE_INTERVAL_MAP[interval]
    limit = CANDLE_LIMIT.get(interval, 300)

    try:
        candles = await fetch_candles(ticker, binance_interval, limit=limit)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Fetch error {ticker}: {e}")
        raise HTTPException(500, "Failed to fetch candle data")

    candle_dicts = [
        {"time": c.time, "open": c.open, "high": c.high,
         "low": c.low, "close": c.close, "volume": c.volume}
        for c in candles
    ]

    # AI Prediction (ONNX model)
    features = build_features_from_candles(candle_dicts, interval)
    ai_result = engine.predict(features, interval) if features is not None else {
        "signal": "NEUTRAL", "confidence": 0.0,
        "conf_24h": 0.5, "conf_48h": 0.5, "conf_72h": 0.5,
        "loaded": False, "used_model": interval
    }

    # Statistical Analysis (least squares + quant)
    prices  = np.array([c.close  for c in candles], dtype=float)
    volumes = np.array([c.volume for c in candles], dtype=float)
    stats   = full_statistical_analysis(prices, volumes,
                                        ai_signal=ai_result["signal"],
                                        ai_conf=ai_result["confidence"])

    price = candles[-1].close
    labels = HORIZON_LABELS.get(interval, HORIZON_LABELS["1h"])
    used_model = ai_result.get("used_model", interval)

    # Build price horizons using statistical channel + AI
    channel  = stats["channel"]
    mean_rev = stats["mean_reversion"]

    TF_MULTIPLIERS = {
        "1m":  [0.001, 0.002, 0.003],
        "5m":  [0.002, 0.005, 0.010],
        "15m": [0.005, 0.010, 0.015],
        "1h":  [0.010, 0.018, 0.025],
        "2h":  [0.018, 0.030, 0.045],
        "1d":  [0.025, 0.050, 0.080],
    }
    mults = TF_MULTIPLIERS.get(interval, TF_MULTIPLIERS["1h"])
    confs = [ai_result["conf_24h"], ai_result["conf_48h"], ai_result["conf_72h"]]

    horizons = []
    for i in range(3):
        conf = confs[i]
        # Combine AI confidence with channel slope direction
        d = 1 if conf > 0.5 else -1
        # If channel has strong trend, blend with AI
        if channel.get("valid") and abs(channel.get("slope_pct", 0)) > 0.05:
            slope_dir = 1 if channel["slope_pct"] > 0 else -1
            d = slope_dir if abs(channel["slope_pct"]) > 0.1 else d

        m = abs(conf - 0.5) * 2 * mults[i]
        # Add trend component
        if channel.get("valid"):
            m += abs(channel.get("slope_pct", 0)) / 100 * (i + 1) * 0.3

        new_price = price * (1 + d * m)
        horizons.append({
            "label":          labels[i],
            "price":          float(new_price),
            "change_percent": float((new_price - price) / price * 100),
            "confidence":     float(conf * 100),
        })

    # Use confluence signal instead of raw AI signal
    confluence = stats["confluence"]
    final_signal = confluence["signal"]
    final_conf   = abs(confluence["confluence"]) * 100

    model_version = f"v5.1-{used_model}+stats" if ai_result["loaded"] else "v5.1-stats-only"

    return {
        "ticker":        ticker,
        "signal":        final_signal,
        "confidence":    final_conf,
        "horizons":      horizons,
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,

        # NEW — Statistical signals breakdown
        "ai_signal":     ai_result["signal"],
        "ai_confidence": ai_result["confidence"] * 100,

        "stats": {
            "channel": {
                "trend":      channel.get("trend"),
                "slope_pct":  channel.get("slope_pct"),
                "r_squared":  channel.get("r_squared"),
                "trend_line": channel.get("trend_line"),
                "upper_2sig": channel.get("upper_2sig"),
                "lower_2sig": channel.get("lower_2sig"),
                "z_score":    channel.get("z_score"),
                "position":   channel.get("position"),
            },
            "mean_reversion": {
                "signal":     mean_rev.get("signal"),
                "z_score":    mean_rev.get("z_score"),
                "conviction": mean_rev.get("conviction"),
            },
            "volatility": stats["volatility"],
            "support_resistance": stats["support_resistance"],
            "momentum": stats["momentum"],
            "confluence": {
                "signal":     confluence["signal"],
                "score":      confluence["confluence"],
                "agreement":  confluence["agreement"],
            },
        },
    }
