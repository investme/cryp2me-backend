from fastapi import APIRouter, HTTPException, Query
from app.models import ChartResponse
from app.services import fetch_candles, fetch_ticker_info, compute_indicators
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chart", tags=["chart"])

VALID_INTERVALS = {"1d", "1h", "5m", "1m"}


@router.get("/{ticker}", response_model=ChartResponse)
async def get_chart(
    ticker: str,
    interval: str = Query("1d", description="Candle interval: 1d, 1h, 5m, 1m"),
):
    ticker = ticker.upper().strip()

    if interval not in VALID_INTERVALS:
        raise HTTPException(400, f"Invalid interval '{interval}'. Choose from: {VALID_INTERVALS}")

    try:
        candles    = await fetch_candles(ticker, interval)
        indicators = compute_indicators(candles)
        info       = await fetch_ticker_info(ticker, candles)

        return ChartResponse(
            ticker=ticker,
            interval=interval,
            candles=candles,
            indicators=indicators,
            ticker_info=info,
        )

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.exception(f"Chart error for {ticker}/{interval}: {e}")
        raise HTTPException(500, "Failed to fetch chart data")
