from fastapi import APIRouter, Query
from app.models import SearchResult
from app.services.binance import KNOWN_NAMES
from typing import List

router = APIRouter(prefix="/search", tags=["search"])

# Top 100 tokens - will be enriched from Binance exchange info at startup
TOP_TICKERS = [
    "PAXG", "XAUT",  # Tokenized Gold
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "UNI", "ATOM", "LTC", "SHIB", "TRX", "DAI", "NEAR", "OP", "ARB",
    "PEPE", "FIL", "INJ", "SUI", "APT", "IMX", "AAVE", "MKR", "SNX", "CRV",
    "ALGO", "ICP", "HBAR", "XLM", "VET", "SAND", "MANA", "AXS", "ENJ", "CHZ",
    "EGLD", "THETA", "EOS", "XTZ", "FTM", "CAKE", "ONE", "ZEC", "DASH", "COMP",
    "YFI", "SUSHI", "BAT", "ZIL", "ICX", "KAVA", "CELO", "SKL", "STORJ", "BNT",
    "GRT", "LRC", "1INCH", "REN", "OCEAN", "NMR", "BAND", "KNC", "CTSI", "RSR",
]


@router.get("", response_model=List[SearchResult])
async def search_tickers(q: str = Query(..., min_length=1)):
    q = q.upper().strip()
    results = [
        SearchResult(ticker=t, name=KNOWN_NAMES.get(t, t))
        for t in TOP_TICKERS
        if t.startswith(q) or q in KNOWN_NAMES.get(t, "").upper()
    ]
    return results[:10]
