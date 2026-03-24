from pathlib import Path
from app.services.inference import ONNXInferenceEngine

MODEL_DIR = Path(__file__).parent.parent / "models" / "onnx"
engine = ONNXInferenceEngine(MODEL_DIR)
