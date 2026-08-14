import numpy as np

import src.inference as inference


class CountingModel:
    classes_ = np.array(["negative", "positive"])

    def __init__(self):
        self.predict_calls = 0
        self.probability_calls = 0

    def predict(self, texts):
        self.predict_calls += 1
        return np.array(["positive"] * len(texts))

    def predict_proba(self, texts):
        self.probability_calls += 1
        return np.tile([0.1, 0.9], (len(texts), 1))


def test_maximum_batch_uses_one_vectorized_model_call(monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(inference, "load_bundle", lambda path=None: {
        "model": model, "model_name": "counting", "version": "test"})
    results = inference.predict_batch(["great service"] * 5000)
    assert len(results) == 5000
    assert model.predict_calls == 1
    assert model.probability_calls == 1
