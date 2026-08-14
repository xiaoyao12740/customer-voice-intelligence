from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    source: str = Field(default="api", max_length=50)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=5000)
    source: str = Field(default="batch", max_length=50)


class PredictionResponse(BaseModel):
    sentiment: str; confidence: float; topic: str; risk: str; risk_score: int; model_name: str; model_version: str; disclaimer: str

