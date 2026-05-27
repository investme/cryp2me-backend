"""
app/services/stat_signals.py — cryp2me.ai Statistical Signals
Least squares regression channels, mean reversion z-scores, volatility forecasting.
These complement the AI predictions with proven quant techniques.
"""

import numpy as np
from typing import Dict, List, Optional


def linear_regression_channel(prices: np.ndarray, lookback: int = 100) -> Dict:
    """
    Fit a least squares regression line through recent prices.
    Returns trend slope, channel boundaries (±2σ), and current position.

    Used by traders to identify trend direction and overbought/oversold zones.
    """
    if len(prices) < lookback:
        lookback = len(prices)
    if lookback < 20:
        return {"valid": False}

    y = prices[-lookback:]
    x = np.arange(lookback, dtype=float)

    # Least squares fit: y = slope * x + intercept
    n = lookback
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # Regression line
    fitted = slope * x + intercept

    # Residuals (price - trend)
    residuals = y - fitted
    std_resid = np.std(residuals)

    # Channels at ±1σ, ±2σ
    upper_1 = fitted + std_resid
    lower_1 = fitted - std_resid
    upper_2 = fitted + 2 * std_resid
    lower_2 = fitted - 2 * std_resid

    current_price = y[-1]
    current_trend = fitted[-1]
    z_score = (current_price - current_trend) / (std_resid + 1e-9)

    # Trend strength: R² (how well the line fits)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y_mean) ** 2) + 1e-9
    r_squared = 1.0 - (ss_res / ss_tot)

    # Annualized slope as % change per period
    slope_pct = (slope / (current_trend + 1e-9)) * 100

    # Classify trend
    if abs(z_score) > 2:
        position = "EXTREME"
    elif abs(z_score) > 1:
        position = "STRETCHED"
    else:
        position = "NORMAL"

    if slope_pct > 0.05 and r_squared > 0.3:
        trend = "UPTREND"
    elif slope_pct < -0.05 and r_squared > 0.3:
        trend = "DOWNTREND"
    else:
        trend = "RANGE"

    return {
        "valid":       True,
        "lookback":    int(lookback),
        "slope_pct":   float(slope_pct),
        "r_squared":   float(r_squared),
        "trend":       trend,
        "z_score":     float(z_score),
        "position":    position,
        "current":     float(current_price),
        "trend_line":  float(current_trend),
        "upper_1sig":  float(upper_1[-1]),
        "lower_1sig":  float(lower_1[-1]),
        "upper_2sig":  float(upper_2[-1]),
        "lower_2sig":  float(lower_2[-1]),
        "channel_line_array": fitted.tolist(),
        "upper_2_array":      upper_2.tolist(),
        "lower_2_array":      lower_2.tolist(),
    }


def mean_reversion_score(prices: np.ndarray, lookback: int = 50) -> Dict:
    """
    Calculate mean reversion score using rolling z-score from regression line.

    Score interpretation:
    -2.5+  : Extreme oversold - HIGH probability bounce up
    -2.0+  : Strong oversold - moderate bounce probability
    -1.0+  : Mildly oversold
     0.0   : At trend
    +1.0+  : Mildly overbought
    +2.0+  : Strong overbought - bounce down likely
    +2.5+  : Extreme overbought - HIGH probability reversal
    """
    channel = linear_regression_channel(prices, lookback)
    if not channel["valid"]:
        return {"valid": False}

    z = channel["z_score"]

    # Signal strength
    if z >= 2.5:
        signal = "STRONG_SELL"
        conviction = 0.85
    elif z >= 2.0:
        signal = "SELL"
        conviction = 0.70
    elif z >= 1.0:
        signal = "WEAK_SELL"
        conviction = 0.55
    elif z <= -2.5:
        signal = "STRONG_BUY"
        conviction = 0.85
    elif z <= -2.0:
        signal = "BUY"
        conviction = 0.70
    elif z <= -1.0:
        signal = "WEAK_BUY"
        conviction = 0.55
    else:
        signal = "NEUTRAL"
        conviction = 0.30

    return {
        "valid":      True,
        "z_score":    z,
        "signal":     signal,
        "conviction": conviction,
        "trend":      channel["trend"],
        "r_squared":  channel["r_squared"],
    }


def volatility_forecast(prices: np.ndarray, lookback: int = 30) -> Dict:
    """
    Forecast next-period volatility using EWMA (similar to RiskMetrics).
    Much more predictable than direction.

    Returns:
    - current_vol: annualized volatility %
    - regime: LOW/NORMAL/HIGH/EXTREME
    - forecast_change: expected vol direction
    """
    if len(prices) < lookback + 1:
        return {"valid": False}

    returns = np.diff(np.log(prices + 1e-9))
    recent_returns = returns[-lookback:]

    # Exponentially weighted moving variance (lambda=0.94 per RiskMetrics)
    lam = 0.94
    weights = np.array([lam ** i for i in range(lookback)][::-1])
    weights /= weights.sum()
    ewma_var = np.sum(weights * (recent_returns ** 2))
    ewma_vol = np.sqrt(ewma_var)

    # Annualized (assume hourly = 24*365 periods)
    annual_vol = ewma_vol * np.sqrt(24 * 365) * 100

    # Compare current to historical average
    hist_vol = np.std(returns) * np.sqrt(24 * 365) * 100

    vol_ratio = annual_vol / (hist_vol + 1e-9)

    if vol_ratio > 1.8:
        regime = "EXTREME"
    elif vol_ratio > 1.3:
        regime = "HIGH"
    elif vol_ratio > 0.7:
        regime = "NORMAL"
    else:
        regime = "LOW"

    # Forecast: volatility is mean-reverting
    if vol_ratio > 1.5:
        forecast = "DECREASING"
    elif vol_ratio < 0.7:
        forecast = "INCREASING"
    else:
        forecast = "STABLE"

    return {
        "valid":       True,
        "current_vol": float(annual_vol),
        "hist_vol":    float(hist_vol),
        "ratio":       float(vol_ratio),
        "regime":      regime,
        "forecast":    forecast,
    }


def support_resistance(prices: np.ndarray, lookback: int = 200) -> Dict:
    """
    Identify statistically significant support/resistance levels
    using pivot points and frequency analysis.
    """
    if len(prices) < lookback:
        lookback = len(prices)
    if lookback < 30:
        return {"valid": False}

    p = prices[-lookback:]

    # Find local maxima (resistance) and minima (support)
    window = 5
    highs, lows = [], []
    for i in range(window, len(p) - window):
        if p[i] == max(p[i - window:i + window + 1]):
            highs.append(p[i])
        if p[i] == min(p[i - window:i + window + 1]):
            lows.append(p[i])

    current = p[-1]

    # Find nearest above (resistance) and below (support)
    resistance = None
    support = None
    for h in sorted(highs):
        if h > current * 1.001:
            resistance = float(h)
            break
    for l in sorted(lows, reverse=True):
        if l < current * 0.999:
            support = float(l)
            break

    if resistance is None:
        resistance = float(max(p) * 1.05) if len(p) else current * 1.05
    if support is None:
        support = float(min(p) * 0.95) if len(p) else current * 0.95

    upside = (resistance - current) / current * 100
    downside = (current - support) / current * 100
    rr = abs(upside) / (abs(downside) + 1e-9)

    return {
        "valid":      True,
        "current":    float(current),
        "support":    support,
        "resistance": resistance,
        "upside_pct": float(upside),
        "downside_pct": float(downside),
        "risk_reward": float(rr),
    }


def momentum_score(prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
    """
    Multi-period momentum score combining short, medium, long term.
    Volume-weighted if volumes provided.
    """
    if len(prices) < 100:
        return {"valid": False}

    current = prices[-1]

    # Returns over different windows
    ret_short  = (current - prices[-10])  / prices[-10]  * 100   # 10 periods
    ret_medium = (current - prices[-30])  / prices[-30]  * 100   # 30 periods
    ret_long   = (current - prices[-100]) / prices[-100] * 100   # 100 periods

    # Acceleration: is momentum increasing?
    accel = ret_short - (ret_medium / 3)

    # Composite score weighted: 50% short, 30% medium, 20% long
    composite = 0.5 * ret_short + 0.3 * ret_medium + 0.2 * ret_long

    if composite > 5 and accel > 0:
        signal = "STRONG_UP"
        conviction = 0.75
    elif composite > 2:
        signal = "UP"
        conviction = 0.60
    elif composite < -5 and accel < 0:
        signal = "STRONG_DOWN"
        conviction = 0.75
    elif composite < -2:
        signal = "DOWN"
        conviction = 0.60
    else:
        signal = "FLAT"
        conviction = 0.30

    return {
        "valid":      True,
        "ret_short":  float(ret_short),
        "ret_medium": float(ret_medium),
        "ret_long":   float(ret_long),
        "composite":  float(composite),
        "accel":      float(accel),
        "signal":     signal,
        "conviction": conviction,
    }


def confluence_score(ai_signal: str, ai_conf: float,
                     mean_rev: Dict, momentum: Dict,
                     channel: Dict) -> Dict:
    """
    Combine AI prediction with statistical signals.
    When multiple methods agree → HIGH conviction signal.

    This is the MAGIC sauce — confluence is what real traders use.
    """
    votes_up = 0
    votes_down = 0
    total_weight = 0

    # AI vote (weighted by confidence)
    if ai_signal == "BULLISH" and ai_conf >= 0.52:
        votes_up += ai_conf
        total_weight += 1
    elif ai_signal == "BEARISH" and ai_conf >= 0.52:
        votes_down += ai_conf
        total_weight += 1

    # Mean reversion vote
    if mean_rev.get("valid"):
        sig = mean_rev["signal"]
        conv = mean_rev["conviction"]
        if "BUY" in sig:
            votes_up += conv
            total_weight += 1
        elif "SELL" in sig:
            votes_down += conv
            total_weight += 1

    # Momentum vote
    if momentum.get("valid"):
        sig = momentum["signal"]
        conv = momentum["conviction"]
        if "UP" in sig:
            votes_up += conv
            total_weight += 1
        elif "DOWN" in sig:
            votes_down += conv
            total_weight += 1

    # Trend channel vote
    if channel.get("valid"):
        trend = channel["trend"]
        if trend == "UPTREND":
            votes_up += 0.5
            total_weight += 1
        elif trend == "DOWNTREND":
            votes_down += 0.5
            total_weight += 1

    if total_weight == 0:
        return {"signal": "NEUTRAL", "confluence": 0.0, "agreement": 0}

    net = (votes_up - votes_down) / total_weight
    agreement = max(votes_up, votes_down) / (votes_up + votes_down + 1e-9)

    if net > 0.4:
        signal = "STRONG_BULLISH"
    elif net > 0.15:
        signal = "BULLISH"
    elif net < -0.4:
        signal = "STRONG_BEARISH"
    elif net < -0.15:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"

    return {
        "signal":     signal,
        "confluence": float(net),
        "agreement":  float(agreement),
        "votes_up":   float(votes_up),
        "votes_down": float(votes_down),
    }


def full_statistical_analysis(prices: np.ndarray,
                              volumes: Optional[np.ndarray] = None,
                              ai_signal: str = "NEUTRAL",
                              ai_conf:   float = 0.0) -> Dict:
    """
    Complete statistical analysis combining all methods.
    This is what gets returned to the user.
    """
    channel  = linear_regression_channel(prices, lookback=100)
    mean_rev = mean_reversion_score(prices, lookback=50)
    vol      = volatility_forecast(prices, lookback=30)
    sr       = support_resistance(prices, lookback=200)
    mom      = momentum_score(prices, volumes)
    conf     = confluence_score(ai_signal, ai_conf, mean_rev, mom, channel)

    return {
        "channel":      channel,
        "mean_reversion": mean_rev,
        "volatility":   vol,
        "support_resistance": sr,
        "momentum":     mom,
        "confluence":   conf,
    }
