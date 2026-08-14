from __future__ import annotations

import re

TOPIC_KEYWORDS = {
    "delivery": {"delivery", "shipping", "arrived", "late", "courier"},
    "quality": {"quality", "broken", "defect", "durable", "material"},
    "service": {"service", "support", "staff", "waiter", "customer"},
    "price": {"price", "cost", "expensive", "cheap", "value"},
    "refund": {"refund", "return", "money", "charge", "cancel"},
    "usability": {"easy", "difficult", "interface", "use", "setup"},
    "product": {"product", "phone", "battery", "screen", "device"},
    "experience": {"movie", "film", "restaurant", "food", "place"},
}
HIGH_RISK = {"legal", "lawyer", "unsafe", "dangerous", "fraud", "scam", "injury", "lawsuit"}
MEDIUM_RISK = {"refund", "broken", "complaint", "angry", "cancel", "terrible", "worst"}


def classify_topic(text: str) -> tuple[str, list[str]]:
    words = set(re.findall(r"[a-z']+", text.lower()))
    scores = {topic: sorted(words & keywords) for topic, keywords in TOPIC_KEYWORDS.items()}
    topic = max(scores, key=lambda item: len(scores[item]))
    return (topic, scores[topic]) if scores[topic] else ("other", [])


def assess_risk(text: str, sentiment: str, confidence: float, topic: str) -> dict:
    words = set(re.findall(r"[a-z']+", text.lower()))
    high_hits, medium_hits = sorted(words & HIGH_RISK), sorted(words & MEDIUM_RISK)
    score = 5
    if sentiment == "negative": score += 35
    if confidence >= 0.8 and sentiment == "negative": score += 10
    if topic in {"refund", "service", "quality"}: score += 10
    score += min(40, len(high_hits) * 30 + len(medium_hits) * 12)
    score = min(100, score)
    level = "high" if score >= 70 else "medium" if score >= 40 else "low"
    return {"level": level, "score": score, "matched_keywords": high_hits + medium_hits, "disclaimer": "Demonstration decision-support rule; not an automated business decision."}
