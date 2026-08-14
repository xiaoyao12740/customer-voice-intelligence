from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def features(max_features: int = 30000):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98, sublinear_tf=True, max_features=max_features)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_features)),
    ])


def build_pipeline(name: str, max_features: int = 30000) -> Pipeline:
    estimators = {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "multinomial_nb": MultinomialNB(alpha=.5),
        "logistic_regression": LogisticRegression(C=3.0, max_iter=2000, random_state=42),
        "linear_svc": LinearSVC(C=.8, random_state=42),
        "sgd_logistic": SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=2000, random_state=42),
    }
    if name not in estimators: raise ValueError(f"Unknown model: {name}")
    return Pipeline([("features", features(max_features)), ("classifier", estimators[name])])


MODEL_NAMES = ["dummy", "multinomial_nb", "logistic_regression", "linear_svc", "sgd_logistic"]

