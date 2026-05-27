"""
app/services/inference.py — cryp2me.ai Phase 5 MTF
Loads one ONNX model set per timeframe.
Falls back to 1h model if specific timeframe not trained yet.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SEQ_LEN_MAP = {
    "1m": 120, "5m": 144, "15m": 168,
    "1h": 168, "2h": 168, "1d": 90,
}

N_FEATURES = 22

HORIZON_LABELS = {
    "1m":  ["T+5m",  "T+15m", "T+30m"],
    "5m":  ["T+15m", "T+1h",  "T+4h"],
    "15m": ["T+1h",  "T+4h",  "T+8h"],
    "1h":  ["T+24h", "T+48h", "T+72h"],
    "2h":  ["T+48h", "T+96h", "T+6d"],
    "1d":  ["T+3d",  "T+7d",  "T+14d"],
}

CONF_THRESHOLD = 0.65


class ONNXModelSet:
    """One set of LSTM + Transformer + Meta for a single timeframe."""

    def __init__(self, model_dir: Path, interval: str):
        self.model_dir = model_dir
        self.interval  = interval
        self.seq_len   = SEQ_LEN_MAP.get(interval, 168)
        self.lstm      = None
        self.tf        = None
        self.meta      = None
        self.loaded    = False

    def load(self):
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2

            lstm_path = self.model_dir / "lstm.onnx"
            tf_path   = self.model_dir / "transformer.onnx"
            meta_path = self.model_dir / "meta_learner.onnx"

            if not lstm_path.exists() or not tf_path.exists():
                return False

            self.lstm = ort.InferenceSession(str(lstm_path), opts)
            self.tf   = ort.InferenceSession(str(tf_path),   opts)
            if meta_path.exists():
                self.meta = ort.InferenceSession(str(meta_path), opts)

            self.loaded = True
            logger.info(f"  ✓ {self.interval.upper()} models loaded")
            return True
        except Exception as e:
            logger.error(f"  ✗ {self.interval} load failed: {e}")
            return False

    def predict(self, features: np.ndarray) -> Dict:
        if not self.loaded:
            return self._stub()
        try:
            x = features[-self.seq_len:].astype(np.float32)
            x = np.expand_dims(x, axis=0)

            lstm_out = self.lstm.run(None, {self.lstm.get_inputs()[0].name: x})
            tf_out   = self.tf.run(None,   {self.tf.get_inputs()[0].name:   x})
            lstm_cls = lstm_out[1]
            tf_cls   = tf_out[1]

            if self.meta:
                meta_in  = np.hstack([lstm_cls, tf_cls]).astype(np.float32)
                meta_out = self.meta.run(None, {self.meta.get_inputs()[0].name: meta_in})
                probs    = meta_out[0][0]
            else:
                probs = (lstm_cls[0] * 0.45 + tf_cls[0] * 0.55)

            avg = float(np.mean(probs))
            if avg >= CONF_THRESHOLD:      signal = "BULLISH"
            elif avg <= 1-CONF_THRESHOLD:  signal = "BEARISH"
            elif avg >= 0.55:              signal = "BULLISH"
            elif avg <= 0.45:              signal = "BEARISH"
            else:                          signal = "NEUTRAL"

            return {
                "signal":     signal,
                "confidence": avg,
                "conf_24h":   float(probs[0]),
                "conf_48h":   float(probs[1]),
                "conf_72h":   float(probs[2]),
                "loaded":     True,
                "interval":   self.interval,
            }
        except Exception as e:
            logger.error(f"Inference error ({self.interval}): {e}")
            return self._stub()

    def _stub(self):
        return {"signal":"LOW_CONFIDENCE","confidence":0.0,
                "conf_24h":0.0,"conf_48h":0.0,"conf_72h":0.0,
                "loaded":False,"interval":self.interval}


class MTFInferenceEngine:
    """
    Multi-timeframe inference engine.
    Loads all available timeframe models.
    Falls back to 1h if specific TF not available.
    """

    def __init__(self, models_root: Path):
        self.models_root = models_root
        self.models: Dict[str, ONNXModelSet] = {}
        self.loaded_intervals = []

    def load(self):
        logger.info("Loading MTF ONNX models...")
        intervals = ["1m","5m","15m","1h","2h","1d"]

        for interval in intervals:
            model_dir = self.models_root / interval
            if not model_dir.exists():
                # Try flat structure (old single model)
                if interval == "1h":
                    model_dir = self.models_root
                else:
                    continue

            ms = ONNXModelSet(model_dir, interval)
            if ms.load():
                self.models[interval] = ms
                self.loaded_intervals.append(interval)

        if self.loaded_intervals:
            logger.info(f"✓ Loaded timeframes: {self.loaded_intervals}")
        else:
            logger.warning("⚠ No ONNX models found — stub mode")

    @property
    def loaded(self):
        return len(self.loaded_intervals) > 0

    def get_model(self, interval: str) -> Optional[ONNXModelSet]:
        """Get model for interval, fall back to 1h if not available."""
        if interval in self.models:
            return self.models[interval]
        if "1h" in self.models:
            logger.info(f"  Fallback: {interval} → 1h model")
            return self.models["1h"]
        # Use any available model
        if self.models:
            return list(self.models.values())[0]
        return None

    def predict(self, features: np.ndarray, interval: str = "1h") -> Dict:
        model = self.get_model(interval)
        if model is None:
            return {"signal":"LOW_CONFIDENCE","confidence":0.0,
                    "conf_24h":0.0,"conf_48h":0.0,"conf_72h":0.0,
                    "loaded":False,"interval":interval}
        result = model.predict(features)
        result["interval"] = interval
        result["used_model"] = model.interval
        return result


# ── Feature Engineering ────────────────────────────────────────────────────────

def _ema(prices, period):
    k = 2.0/(period+1); out = np.full(len(prices), np.nan)
    if len(prices) < period: return out
    out[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        out[i] = prices[i]*k + out[i-1]*(1-k)
    return out

def _rsi(prices, period=14):
    out = np.full(len(prices), 50.0); d = np.diff(prices)
    g = np.where(d>0,d,0.0); l = np.where(d<0,-d,0.0)
    if len(d)<period: return out
    ag,al = np.mean(g[:period]),np.mean(l[:period])
    for i in range(period,len(d)):
        ag=(ag*(period-1)+g[i])/period; al=(al*(period-1)+l[i])/period
        out[i+1]=100.0-100.0/(1.0+(ag/al if al>0 else 100.0))
    return out

def build_features_from_candles(candles, interval="1h"):
    seq_len = SEQ_LEN_MAP.get(interval, 168)
    if len(candles) < seq_len + 10:
        return None
    o = np.array([c["open"]   for c in candles], dtype=np.float64)
    h = np.array([c["high"]   for c in candles], dtype=np.float64)
    l = np.array([c["low"]    for c in candles], dtype=np.float64)
    c = np.array([c["close"]  for c in candles], dtype=np.float64)
    v = np.array([c["volume"] for c in candles], dtype=np.float64)
    n = len(c)
    f = np.zeros((n, N_FEATURES), dtype=np.float32)
    f[1:,0]=np.diff(o)/(o[:-1]+1e-8); f[1:,1]=np.diff(h)/(h[:-1]+1e-8)
    f[1:,2]=np.diff(l)/(l[:-1]+1e-8); f[1:,3]=np.diff(c)/(c[:-1]+1e-8)
    vm=np.mean(v)+1e-8; v20=np.convolve(v,np.ones(20)/20,mode='same')
    f[:,4]=v/vm; f[:,5]=v/(v20+1e-8)
    f[1:,6]=np.diff(v)/(v[:-1]+1e-8)
    f[:,7]=np.where(f[:,3]>0,f[:,5],-f[:,5])
    f[1:,8]=np.diff(v20)/(v20[:-1]+1e-8)
    obv=np.zeros(n); obv[1:]=np.cumsum(np.where(f[1:,3]>0,v[1:],-v[1:]))
    f[:,9]=obv/(np.max(np.abs(obv))+1e-8)
    e10=_ema(c,10); e20=_ema(c,20); e34=_ema(c,34)
    f[:,10]=np.nan_to_num((c-e10)/(c+1e-8))
    f[:,11]=np.nan_to_num((c-e20)/(c+1e-8))
    f[:,12]=np.nan_to_num((c-e34)/(c+1e-8))
    f[:,13]=(_rsi(c,14)-50.0)/50.0
    ml=_ema(c,12)-_ema(c,26); sl=_ema(np.nan_to_num(ml),9)
    f[:,14]=np.nan_to_num(ml/(c+1e-8))
    f[:,15]=np.nan_to_num(sl/(c+1e-8))
    f[:,16]=f[:,14]-f[:,15]
    tr=np.maximum(h[1:]-l[1:],np.maximum(np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])))
    atr=np.full(n,np.nan)
    if len(tr)>=14:
        atr[14]=np.mean(tr[:14])
        for i in range(15,n): atr[i]=(atr[i-1]*13+tr[i-1])/14
    f[:,17]=np.nan_to_num(atr/(c+1e-8))
    ma20=np.convolve(c,np.ones(20)/20,mode='same')
    std20=np.array([np.std(c[max(0,i-20):i]) for i in range(n)])
    f[:,18]=np.nan_to_num(2*std20/(ma20+1e-8))
    ret=np.nan_to_num(f[:,3])
    vol=np.array([np.std(ret[max(0,i-24):i+1]) for i in range(n)])
    reg=np.zeros(n,dtype=int)
    if np.any(vol>0):
        p33,p66=np.percentile(vol[vol>0],[33,66])
        reg[vol>p66]=2; reg[(vol>p33)&(vol<=p66)]=1
    for i in range(n): f[i,19+reg[i]]=1.0
    return np.clip(np.nan_to_num(f,nan=0.,posinf=0.,neginf=0.),-10.,10.).astype(np.float32)
