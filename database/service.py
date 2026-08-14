from __future__ import annotations

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from database.models import Base, Prediction, Review
from src.config import database_url


def get_engine(url: str | None = None):
    value = url or database_url(); kwargs = {"connect_args": {"check_same_thread": False}} if value.startswith("sqlite") else {}
    return create_engine(value, pool_pre_ping=True, **kwargs)


def init_database(engine=None):
    engine = engine or get_engine(); Base.metadata.create_all(engine); return engine


def save_prediction(result: dict, source="api", engine=None) -> int:
    engine = engine or init_database()
    with Session(engine) as session:
        review = Review(text=result["text"], source=source); session.add(review); session.flush()
        prediction = Prediction(review_id=review.id, sentiment=result["sentiment"], confidence=result["confidence"], topic=result["topic"], risk_level=result["risk"], risk_score=result["risk_score"], model_version=result["model_version"])
        session.add(prediction); session.commit(); return review.id


def analytics_summary(engine=None) -> dict:
    engine = engine or init_database()
    with Session(engine) as session:
        total = session.scalar(select(func.count()).select_from(Prediction)) or 0
        sentiments = dict(session.execute(select(Prediction.sentiment, func.count()).group_by(Prediction.sentiment)).all())
        topics = dict(session.execute(select(Prediction.topic, func.count()).group_by(Prediction.topic)).all())
        high_risk = session.scalar(select(func.count()).select_from(Prediction).where(Prediction.risk_level == "high")) or 0
    return {"total_predictions": total, "sentiments": sentiments, "topics": topics, "high_risk": high_risk}

