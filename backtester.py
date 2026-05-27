"""
app/services/backtester.py — cryp2me.ai Backtester
Runs predictions over historical data and measures profitability.

The honest test — does our system actually make money?
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def run_backtest(
    candles:       list,        # List of candle dicts with open/high/low/close/volume/time
    interval:      str,
    engine,                     # MTF inference engine
    feature_fn,                 # build_features_from_candles
    stat_fn,                    # full_statistical_analysis
    horizon_idx:   int = 0,     # Which horizon to test (0=short, 1=mid, 2=long)
    min_confidence:float = 0.55,# Only trade when confluence >= this
    step:          int = 4,     # Skip every N candles to speed up
) -> Dict:
    """
    Replay history. At each point, generate prediction, check if it was correct
    Y candles later (Y = horizon length).

    Returns full performance stats.
    """
    # Horizon lengths per interval (candles ahead)
    horizons = {
        "1m":  [5, 15, 30],
        "5m":  [3, 12, 48],
        "15m": [4, 16, 32],
        "1h":  [24, 48, 72],
        "2h":  [24, 48, 72],
        "1d":  [3,  7, 14],
    }
    hz_list = horizons.get(interval, [24, 48, 72])
    hz_candles = hz_list[horizon_idx]

    # Minimum lookback needed for inference + statistics
    min_lookback = 220

    if len(candles) < min_lookback + hz_candles + 50:
        return {"error": f"Insufficient candles ({len(candles)}). Need at least {min_lookback + hz_candles + 50}"}

    signals       = []  # All signals generated
    trades        = []  # Trades that passed confidence threshold
    wins          = 0
    losses        = 0
    total_pnl_pct = 0.0
    max_drawdown  = 0.0
    peak_equity   = 1.0
    equity        = 1.0
    equity_curve  = []

    # Track confluence-only and AI-only separately
    confluence_correct = 0
    confluence_total   = 0
    ai_correct = 0
    ai_total   = 0

    # Walk forward through history
    test_range = range(min_lookback, len(candles) - hz_candles, step)
    total_steps = len(list(test_range))
    logger.info(f"Backtesting {interval} h{horizon_idx} — {total_steps} steps over {len(candles)} candles")

    for i in test_range:
        # Use only past data — no lookahead
        history = candles[:i + 1]
        current_price = history[-1]["close"]

        # Future price at horizon
        future_idx = i + hz_candles
        if future_idx >= len(candles):
            break
        future_price = candles[future_idx]["close"]

        actual_return_pct = (future_price - current_price) / current_price * 100
        actual_up = future_price > current_price

        # Generate prediction using actual inference engine
        try:
            features = feature_fn(history, interval)
            if features is None:
                continue
            ai_result = engine.predict(features, interval)

            prices = np.array([c["close"] for c in history], dtype=float)
            volumes = np.array([c["volume"] for c in history], dtype=float)
            stats = stat_fn(prices, volumes,
                          ai_signal=ai_result["signal"],
                          ai_conf=ai_result["confidence"])
        except Exception as e:
            continue

        confluence = stats["confluence"]
        sig = confluence["signal"]
        conf_score = abs(confluence["confluence"])

        # Track AI-only performance
        ai_sig = ai_result["signal"]
        if ai_sig in ("BULLISH", "BEARISH"):
            ai_total += 1
            ai_predicted_up = (ai_sig == "BULLISH")
            if ai_predicted_up == actual_up:
                ai_correct += 1

        # Track confluence performance
        if sig in ("BULLISH", "BEARISH", "STRONG_BULLISH", "STRONG_BEARISH"):
            confluence_total += 1
            predicted_up = ("BULLISH" in sig)
            if predicted_up == actual_up:
                confluence_correct += 1

        signals.append({
            "i":            i,
            "signal":       sig,
            "conf_score":   float(conf_score),
            "ai_signal":    ai_sig,
            "current":      float(current_price),
            "future":       float(future_price),
            "actual_pct":   float(actual_return_pct),
        })

        # Only "trade" if confluence passes threshold
        if conf_score < min_confidence:
            continue
        if sig not in ("BULLISH", "BEARISH", "STRONG_BULLISH", "STRONG_BEARISH"):
            continue

        predicted_up = ("BULLISH" in sig)

        # Simulate trade — bet on the prediction
        # Position sizing: 2% of equity per trade
        position = 0.02
        trade_return = actual_return_pct / 100 * position
        if not predicted_up:
            trade_return = -trade_return  # Short side

        equity *= (1 + trade_return)
        equity_curve.append(equity)

        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if (predicted_up and actual_up) or (not predicted_up and not actual_up):
            wins += 1
            total_pnl_pct += abs(actual_return_pct)
        else:
            losses += 1
            total_pnl_pct -= abs(actual_return_pct)

        trades.append({
            "i":         i,
            "signal":    sig,
            "predicted": "UP" if predicted_up else "DOWN",
            "actual":    "UP" if actual_up else "DOWN",
            "pct":       float(actual_return_pct),
            "win":       (predicted_up == actual_up),
            "equity":    float(equity),
        })

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_win = np.mean([t["pct"] for t in trades if t["win"]]) if any(t["win"] for t in trades) else 0
    avg_loss = np.mean([abs(t["pct"]) for t in trades if not t["win"]]) if any(not t["win"] for t in trades) else 0
    profit_factor = (avg_win * wins) / (avg_loss * losses) if (avg_loss > 0 and losses > 0) else float('inf') if wins > 0 else 0

    confluence_acc = confluence_correct / confluence_total if confluence_total > 0 else 0
    ai_acc = ai_correct / ai_total if ai_total > 0 else 0

    # Sharpe-like ratio (simplified)
    if len(trades) > 1:
        returns = [t["pct"] / 100 for t in trades]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
    else:
        sharpe = 0

    # Verdict
    verdict = "❌ NOT PROFITABLE"
    if win_rate >= 0.52 and profit_factor >= 1.1:
        verdict = "✅ PROFITABLE"
    elif win_rate >= 0.50 and profit_factor >= 1.0:
        verdict = "⚠️ MARGINAL"

    return {
        "interval":    interval,
        "horizon_idx": horizon_idx,
        "horizon_candles": hz_candles,
        "total_signals":   len(signals),
        "total_trades":    total_trades,
        "wins":            wins,
        "losses":          losses,
        "win_rate":        float(win_rate),
        "ai_only_acc":     float(ai_acc),
        "ai_total":        ai_total,
        "confluence_acc":  float(confluence_acc),
        "confluence_total":confluence_total,
        "avg_win":         float(avg_win),
        "avg_loss":        float(avg_loss),
        "profit_factor":   float(profit_factor) if profit_factor != float('inf') else 999.0,
        "total_return_pct":float((equity - 1) * 100),
        "max_drawdown_pct":float(max_drawdown * 100),
        "sharpe":          float(sharpe),
        "verdict":         verdict,
        "equity_curve":    [float(e) for e in equity_curve[-100:]],  # Last 100 for display
        "sample_trades":   trades[-20:],  # Last 20 for review
    }
