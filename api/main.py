from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from api.schemas import BatchRequest, PredictRequest
from database.service import analytics_summary, init_database, save_prediction
from src.config import load_config
from src.inference import load_bundle, predict, predict_batch


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    yield


app = FastAPI(title="Customer Voice Intelligence API", version="1.0.0", description="Sentiment prediction with transparent topic and complaint-risk rules.", lifespan=lifespan)


@app.get("/health")
def health(): return {"status": "ok"}


@app.get("/model-info")
def model_info():
    bundle = load_bundle(); return {key: bundle[key] for key in ("model_name", "version", "labels")}


@app.post("/predict")
def predict_one(request: PredictRequest):
    try:
        result = predict(request.text); review_id = save_prediction(result, request.source)
        return {"review_id": review_id, **result}
    except (ValueError, FileNotFoundError) as error: raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/predict/batch")
def predict_many(request: BatchRequest):
    try:
        results = predict_batch(request.texts)
        for result in results: save_prediction(result, request.source)
        return {"count": len(results), "predictions": results}
    except (ValueError, FileNotFoundError) as error: raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/analytics/summary")
def summary(): return analytics_summary()

