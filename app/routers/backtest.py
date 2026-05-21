"""
app/routers/backtest.py — cryp2me.ai Backtest endpoint
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query
from app.services.binance import fetch_candles
from app.services.inference import build_features_from_candles
from app.services.stat_signals import full_statistical_analysis
from app.services.backtester import run_backtest
from app.state import engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.get("/{ticker}")
async def backtest_ticker(
    ticker:      str,
    interval:    str   = Query("1h"),
    horizon_idx: int   = Query(0, ge=0, le=2),
    candles:     int   = Query(1000, ge=300, le=1000),
    min_conf:    float = Query(0.55, ge=0.0, le=1.0),
):
    """
    Run backtest on historical data.

    Args:
        ticker: BTC, ETH, SOL, etc.
        interval: 1m, 5m, 15m, 1h, 2h, 1d
        horizon_idx: 0=short, 1=mid, 2=long horizon
        candles: Number of historical candles to backtest (max 1000)
        min_conf: Minimum confluence score to take trade
    """
    ticker   = ticker.upper().strip()
    interval = interval.lower().strip()

    if interval not in ["1m","5m","15m","1h","2h","1d"]:
        raise HTTPException(400, "Invalid interval")

    try:
        # Fetch candles
        binance_candles = await fetch_candles(ticker, interval, limit=candles)
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch candles: {e}")

    candle_dicts = [
        {"time":c.time, "open":c.open, "high":c.high,
         "low":c.low, "close":c.close, "volume":c.volume}
        for c in binance_candles
    ]

    logger.info(f"Starting backtest: {ticker} {interval} h={horizon_idx} candles={len(candle_dicts)}")

    # Run in thread to not block event loop (backtest is CPU heavy)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: run_backtest(
            candles      = candle_dicts,
            interval     = interval,
            engine       = engine,
            feature_fn   = build_features_from_candles,
            stat_fn      = full_statistical_analysis,
            horizon_idx  = horizon_idx,
            min_confidence = min_conf,
            step         = 4,
        )
    )

    result["ticker"] = ticker
    return result
