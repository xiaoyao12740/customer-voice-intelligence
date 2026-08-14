from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np

from src.config import load_config
from src.preprocessing import validate_text
from src.rules import assess_risk, classify_topic


@lru_cache(maxsize=2)
def load_bundle(path: str | None = None):
    model_path = Path(path or load_config()["outputs"]["model_path"])
    if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}. Run python -m src.benchmark first.")
    return joblib.load(model_path)


def sentiment_probability(model, text: str, prediction: str) -> float:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_); probs = model.predict_proba([text])[0]
        return float(probs[classes.index(prediction)])
    score = float(model.decision_function([text])[0]); positive = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
    return positive if prediction == "positive" else 1 - positive


def _sentiment_confidences(model, texts, predictions) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba(texts)
        return np.asarray([probabilities[index, classes.index(label)] for index, label in enumerate(predictions)])
    scores = np.asarray(model.decision_function(texts), dtype=float)
    positive = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
    return np.where(np.asarray(predictions) == "positive", positive, 1 - positive)


def predict(text: str, model_path: str | None = None) -> dict:
    clean = validate_text(text); bundle = load_bundle(model_path); model = bundle["model"]
    sentiment = str(model.predict([clean])[0]); confidence = sentiment_probability(model, clean, sentiment)
    topic, topic_keywords = classify_topic(clean); risk = assess_risk(clean, sentiment, confidence, topic)
    return {"text": text, "clean_text": clean, "sentiment": sentiment, "confidence": confidence, "topic": topic, "topic_keywords": topic_keywords, "risk": risk["level"], "risk_score": risk["score"], "risk_keywords": risk["matched_keywords"], "model_name": bundle["model_name"], "model_version": bundle["version"], "disclaimer": risk["disclaimer"]}


def predict_batch(texts: list[str], model_path: str | None = None) -> list[dict]:
    if not texts or len(texts) > 5000: raise ValueError("Batch must contain between 1 and 5000 texts.")
    clean_texts = [validate_text(text) for text in texts]
    bundle = load_bundle(model_path); model = bundle["model"]
    sentiments = model.predict(clean_texts)
    confidences = _sentiment_confidences(model, clean_texts, sentiments)
    results = []
    for text, clean, sentiment, confidence in zip(texts, clean_texts, sentiments, confidences):
        sentiment = str(sentiment); confidence = float(confidence)
        topic, topic_keywords = classify_topic(clean)
        risk = assess_risk(clean, sentiment, confidence, topic)
        results.append({"text": text, "clean_text": clean, "sentiment": sentiment, "confidence": confidence,
            "topic": topic, "topic_keywords": topic_keywords, "risk": risk["level"], "risk_score": risk["score"],
            "risk_keywords": risk["matched_keywords"], "model_name": bundle["model_name"],
            "model_version": bundle["version"], "disclaimer": risk["disclaimer"]})
    return results
