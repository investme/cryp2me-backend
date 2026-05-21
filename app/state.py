"""
app/state.py — MTF inference engine singleton
"""
from pathlib import Path
from app.services.inference import MTFInferenceEngine

MODEL_DIR = Path(__file__).parent.parent / "models" / "onnx"
engine = MTFInferenceEngine(MODEL_DIR)
