from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.routers import chart_router, search_router, predict_router
from app.routers.backtest import router as backtest_router
from app.state import engine

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 cryp2me.ai backend starting — Phase 5")
    engine.load()
    if engine.loaded:
        logger.info("✓ ONNX models loaded")
    else:
        logger.warning("⚠ No ONNX models found — running in stub mode")
    yield
    logger.info("👋 cryp2me.ai backend shutting down")


app = FastAPI(
    title="cryp2me.ai API",
    description="AI-powered crypto prediction — Phase 5",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chart_router,   prefix="/api")
app.include_router(search_router,  prefix="/api")
app.include_router(predict_router,  prefix="/api")
app.include_router(backtest_router, prefix="/api")


@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "version":       "5.0.0",
        "models_loaded": engine.loaded,
    }





app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
