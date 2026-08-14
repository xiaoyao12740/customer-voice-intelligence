from src.rules import assess_risk, classify_topic


def test_topic_and_risk_are_explainable():
    topic,keywords=classify_topic("Delivery was late and I need a refund")
    assert topic in {"delivery","refund"} and keywords
    risk=assess_risk("This unsafe product is a scam", "negative", .95, "quality")
    assert risk["level"]=="high" and risk["matched_keywords"]


def test_rule_tokenization_handles_punctuation():
    topic, keywords = classify_topic("I need a refund.")
    risk = assess_risk("This is terrible!", "negative", .9, topic)
    assert topic == "refund" and keywords == ["refund"]
    assert "terrible" in risk["matched_keywords"]
