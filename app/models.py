from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Candle(BaseModel):
    time:   int
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


class IndicatorPoint(BaseModel):
    time:  int
    value: float


class MACDPoint(BaseModel):
    time:      int
    macd:      float
    signal:    float
    histogram: float


class Indicators(BaseModel):
    ema10: List[IndicatorPoint]
    ema20: List[IndicatorPoint]
    ema34: List[IndicatorPoint]
    rsi:   List[IndicatorPoint]
    macd:  List[MACDPoint]


class TickerInfo(BaseModel):
    ticker:         str
    name:           str
    open:           float
    high:           float
    low:            float
    close:          float
    volume:         float
    marketCap:      float = Field(alias="market_cap", default=0.0)
    changePercent:  float = Field(alias="change_percent", default=0.0)
    changeDollar:   float = Field(alias="change_dollar", default=0.0)

    model_config = {"populate_by_name": True}


class ChartResponse(BaseModel):
    ticker:     str
    interval:   str
    candles:    List[Candle]
    indicators: Indicators
    tickerInfo: TickerInfo = Field(alias="ticker_info")

    model_config = {"populate_by_name": True}


Signal = Literal["BULLISH", "BEARISH", "NEUTRAL", "LOW_CONFIDENCE"]


class PredictionHorizon(BaseModel):
    label:         str
    price:         float
    changePercent: float = Field(alias="change_percent", default=0.0)
    confidence:    float

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    ticker:       str
    signal:       Signal
    confidence:   float
    horizons:     List[PredictionHorizon]
    generatedAt:  str = Field(alias="generated_at")
    modelVersion: str = Field(alias="model_version")

    model_config = {"populate_by_name": True}


class SearchResult(BaseModel):
    ticker: str
    name:   str
    icon:   Optional[str] = None
