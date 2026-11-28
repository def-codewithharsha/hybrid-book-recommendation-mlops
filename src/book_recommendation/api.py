"""FastAPI serving layer for the hybrid recommender."""
import re
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from book_recommendation.config import get_settings
from book_recommendation.logger import configure_logging, get_logger
from book_recommendation.model_artifact import load_model
from book_recommendation.model_trainer import HybridRecommender

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.logging.level, settings.logging.format)
    try:
        app.state.model = load_model(settings.model_path)
        logger.info("Model loaded from %s", settings.model_path)
    except FileNotFoundError:
        app.state.model = None
        logger.warning("No model artifact found at %s", settings.model_path)
    yield


app = FastAPI(title="Hybrid Book Recommendation API", version="0.1.0", lifespan=lifespan)


class RecommendationRequest(BaseModel):
    user_id: str | None = None
    book_id: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    supplied_request_id = request.headers.get("x-request-id", "")
    request_id = (
        supplied_request_id
        if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", supplied_request_id)
        else str(uuid4())
    )
    start = perf_counter()
    context = {
        "request_id": request_id,
        "http_method": request.method,
        "http_path": request.url.path,
    }
    try:
        response = await call_next(request)
    except Exception:
        context["duration_ms"] = round((perf_counter() - start) * 1000, 2)
        logger.exception("request failed", extra=context)
        raise
    context["status_code"] = response.status_code
    context["duration_ms"] = round((perf_counter() - start) * 1000, 2)
    response.headers["x-request-id"] = request_id
    logger.info("request complete", extra=context)
    response.headers["server-timing"] = f"app;dur={context['duration_ms']:.2f}"
    return response


@app.get("/health/live")
def live():
    return {"status": "live"}


@app.get("/health/ready")
def ready(request: Request):
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready", "books": len(request.app.state.model.item_ids)}


@app.get("/health")
def health(request: Request):
    return {"status": "ready" if request.app.state.model is not None else "not_ready"}


@app.get("/metrics")
def metrics(request: Request):
    model = request.app.state.model
    return {"model_loaded": model is not None, "catalog_books": len(model.item_ids) if model else 0}


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest, request: Request):
    if payload.user_id is None and payload.book_id is None:
        raise HTTPException(status_code=422, detail="Provide user_id or book_id")
    model: HybridRecommender | None = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        items = model.recommend(payload.user_id, payload.book_id, payload.limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"recommendations": items, "count": len(items)}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled request error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
