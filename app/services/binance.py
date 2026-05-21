"""
Binance Data Service
────────────────────
Fetches OHLCV from Binance public REST API.
Falls back to CoinGecko for tokens not on Binance.
Computes all indicators server-side.
"""

import httpx
import numpy as np
from typing import List, Optional
from datetime import datetime, timezone

from app.models import Candle, IndicatorPoint, MACDPoint, Indicators, TickerInfo

# ── Constants ─────────────────────────────────────────────────────────────────

BINANCE_BASE = "https://api.binance.com/api/v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

INTERVAL_MAP = {
    "1d": "1d",
    "1h": "1h",
    "5m": "5m",
    "1m": "1m",
}

INTERVAL_LIMITS = {
    "1d": 365,
    "1h": 720,
    "5m": 1440,
    "1m": 1440,
}

KNOWN_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "BNB": "BNB", "SOL": "Solana",
    "XRP": "XRP", "ADA": "Cardano", "DOGE": "Dogecoin", "AVAX": "Avalanche",
    "LINK": "Chainlink", "DOT": "Polkadot", "MATIC": "Polygon", "UNI": "Uniswap",
    "ATOM": "Cosmos", "LTC": "Litecoin", "SHIB": "Shiba Inu", "TRX": "Tron",
    "DAI": "Dai", "USDC": "USD Coin", "NEAR": "NEAR Protocol", "OP": "Optimism",
    "ARB": "Arbitrum", "PEPE": "Pepe", "FIL": "Filecoin", "INJ": "Injective",
    "SUI": "Sui", "APT": "Aptos", "IMX": "ImmutableX",
    "PAXG": "Pax Gold",
    "XAUT": "Tether Gold",
}

# ── Fetch Candles ─────────────────────────────────────────────────────────────

async def fetch_candles(ticker: str, interval: str, limit: int = None) -> List[Candle]:
    """Fetch OHLCV from Binance. Tries USDT pair, then BUSD, then BTC pair."""
    symbol = f"{ticker.upper()}USDT"
    limit  = limit or INTERVAL_LIMITS.get(interval, 365)
    binance_interval = INTERVAL_MAP.get(interval, "1d")

    async with httpx.AsyncClient(timeout=15.0) as client:
        for quote in ["USDT", "BUSD", "BTC"]:
            sym = f"{ticker.upper()}{quote}"
            try:
                resp = await client.get(
                    f"{BINANCE_BASE}/klines",
                    params={"symbol": sym, "interval": binance_interval, "limit": limit}
                )
                if resp.status_code == 200:
                    raw = resp.json()
                    return [
                        Candle(
                            time=int(row[0]) // 1000,
                            open=float(row[1]),
                            high=float(row[2]),
                            low=float(row[3]),
                            close=float(row[4]),
                            volume=float(row[5]),
                        )
                        for row in raw
                    ]
            except Exception:
                continue

    raise ValueError(f"Could not find ticker '{ticker}' on Binance")


async def fetch_ticker_info(ticker: str, candles: List[Candle]) -> TickerInfo:
    """Build ticker info from candle data + Binance 24h stats."""
    last  = candles[-1]
    prev  = candles[-2] if len(candles) > 1 else candles[-1]

    change_dollar  = last.close - prev.close
    change_percent = (change_dollar / prev.close * 100) if prev.close else 0.0

    # Try to get market cap from Binance 24h endpoint
    market_cap = 0.0
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{BINANCE_BASE}/ticker/24hr",
                params={"symbol": f"{ticker.upper()}USDT"}
            )
            if resp.status_code == 200:
                data = resp.json()
                market_cap = float(data.get("quoteVolume", 0))  # approx
    except Exception:
        pass

    return TickerInfo(
        ticker=ticker.upper(),
        name=KNOWN_NAMES.get(ticker.upper(), ticker.upper()),
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume,
        market_cap=market_cap,
        change_percent=change_percent,
        change_dollar=change_dollar,
    )


# ── Indicator Calculations ────────────────────────────────────────────────────

def _ema(prices: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    out = np.empty(len(prices))
    out[:] = np.nan
    if len(prices) < period:
        return out
    seed = np.mean(prices[:period])
    out[period - 1] = seed
    for i in range(period, len(prices)):
        out[i] = prices[i] * k + out[i - 1] * (1 - k)
    return out


def calc_ema(candles: List[Candle], period: int) -> List[IndicatorPoint]:
    prices = np.array([c.close for c in candles])
    times  = [c.time for c in candles]
    ema    = _ema(prices, period)
    return [
        IndicatorPoint(time=times[i], value=float(ema[i]))
        for i in range(len(candles))
        if not np.isnan(ema[i])
    ]


def calc_rsi(candles: List[Candle], period: int = 14) -> List[IndicatorPoint]:
    prices = np.array([c.close for c in candles])
    times  = [c.time for c in candles]
    result = []

    if len(prices) < period + 1:
        return result

    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    def rsi_val(ag, al):
        return 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    result.append(IndicatorPoint(time=times[period], value=rsi_val(avg_gain, avg_loss)))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        result.append(IndicatorPoint(time=times[i + 1], value=rsi_val(avg_gain, avg_loss)))

    return result


def calc_macd(
    candles: List[Candle],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> List[MACDPoint]:
    prices = np.array([c.close for c in candles])
    times  = [c.time for c in candles]

    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)
    macd_line = ema_fast - ema_slow

    # signal = EMA(9) of MACD
    valid_start = slow - 1
    macd_valid  = macd_line[valid_start:]
    times_valid = times[valid_start:]

    k = 2.0 / (signal_period + 1)
    signal_arr = np.empty(len(macd_valid))
    signal_arr[:] = np.nan

    if len(macd_valid) < signal_period:
        return []

    signal_arr[signal_period - 1] = np.mean(macd_valid[:signal_period])
    for i in range(signal_period, len(macd_valid)):
        signal_arr[i] = macd_valid[i] * k + signal_arr[i - 1] * (1 - k)

    result = []
    for i in range(signal_period - 1, len(macd_valid)):
        if np.isnan(signal_arr[i]):
            continue
        result.append(MACDPoint(
            time=times_valid[i],
            macd=float(macd_valid[i]),
            signal=float(signal_arr[i]),
            histogram=float(macd_valid[i] - signal_arr[i]),
        ))
    return result


def compute_indicators(candles: List[Candle]) -> Indicators:
    return Indicators(
        ema10=calc_ema(candles, 10),
        ema20=calc_ema(candles, 20),
        ema34=calc_ema(candles, 34),
        rsi=calc_rsi(candles, 14),
        macd=calc_macd(candles, 12, 26, 9),
    )
