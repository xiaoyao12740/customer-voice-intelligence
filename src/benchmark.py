from __future__ import annotations

import json
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import load_config
from src.eda import run_eda
from src.models import MODEL_NAMES, build_pipeline


def scores_from_model(model, texts):
    if hasattr(model, "predict_proba"): return model.predict_proba(texts)[:, 1]
    raw = model.decision_function(texts)
    return 1 / (1 + np.exp(-np.clip(raw, -30, 30)))


def bootstrap_ci(y_true, y_pred, seed=42, rounds=1000):
    rng = np.random.default_rng(seed); values = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for _ in range(rounds):
        idx = rng.integers(0, len(y_true), len(y_true)); values.append(f1_score(y_true[idx], y_pred[idx], pos_label="positive"))
    return [float(np.quantile(values, .025)), float(np.quantile(values, .975))]


def benchmark() -> dict:
    config = load_config(); processed = Path(config["data"]["processed_dir"]); output = Path(config["outputs"]["root"]); output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(processed / "train.csv"); validation = pd.read_csv(processed / "validation.csv"); test = pd.read_csv(processed / "test.csv")
    all_data = pd.concat([train, validation, test], ignore_index=True); eda = run_eda(all_data, output / "eda")
    x_train, y_train = train["clean_text"], train["sentiment"]; x_test, y_test = test["clean_text"], test["sentiment"]
    results, fitted = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(config["project"]["seed"]))
    for name in MODEL_NAMES:
        model = build_pipeline(name, int(config["model"]["max_features"])); started = time.perf_counter(); model.fit(x_train, y_train); train_seconds = time.perf_counter() - started
        started = time.perf_counter(); prediction = model.predict(x_test); inference_ms = (time.perf_counter() - started) * 1000 / len(test)
        probabilities = scores_from_model(model, x_test); cv_scores = cross_val_score(build_pipeline(name, int(config["model"]["max_features"])), x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        row = {"model": name, "accuracy": accuracy_score(y_test, prediction), "precision": precision_score(y_test, prediction, pos_label="positive", zero_division=0), "recall": recall_score(y_test, prediction, pos_label="positive", zero_division=0), "f1": f1_score(y_test, prediction, pos_label="positive", zero_division=0), "macro_f1": f1_score(y_test, prediction, average="macro"), "weighted_f1": f1_score(y_test, prediction, average="weighted"), "roc_auc": roc_auc_score((y_test == "positive").astype(int), probabilities), "pr_auc": average_precision_score((y_test == "positive").astype(int), probabilities), "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std(), "train_seconds": train_seconds, "inference_ms_per_row": inference_ms}
        results.append(row); fitted[name] = (model, prediction, probabilities)
    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False); table.to_csv(output / "benchmark.csv", index=False)
    production_name = config["model"]["production"]; production, prediction, probabilities = fitted[production_name]
    model_path = Path(config["outputs"]["model_path"]); model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": production, "model_name": production_name, "version": config["project"]["model_version"], "labels": ["negative", "positive"]}; joblib.dump(bundle, model_path, compress=3)
    errors = test.assign(prediction=prediction, positive_probability=probabilities); errors[errors.sentiment != errors.prediction].to_csv(output / "error_analysis.csv", index=False)
    report = {"production_model": production_name, "test_rows": len(test), "metrics": next(row for row in results if row["model"] == production_name), "f1_95_ci": bootstrap_ci(y_test, prediction), "classification_report": classification_report(y_test, prediction, output_dict=True), "eda": eda}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(table, y_test, prediction, production, output)
    return report


def plot_results(table, y_test, prediction, model, output):
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.barh(table.model, table.macro_f1, color="#3478bf"); ax.set_xlim(.4, 1); ax.set_xlabel("Macro F1"); ax.set_title("Model benchmark"); ax.grid(axis="x", alpha=.2); fig.tight_layout(); fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)
    matrix = confusion_matrix(y_test, prediction, labels=["negative", "positive"]); fig, ax = plt.subplots(figsize=(5, 4)); image=ax.imshow(matrix,cmap="Blues"); fig.colorbar(image,ax=ax); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["negative","positive"],yticklabels=["negative","positive"],xlabel="Predicted",ylabel="Actual",title="Logistic Regression Confusion Matrix")
    for i in range(2):
        for j in range(2): ax.text(j,i,matrix[i,j],ha="center",va="center")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)
    features = model.named_steps["features"].get_feature_names_out(); coef=model.named_steps["classifier"].coef_[0]; idx=np.r_[np.argsort(coef)[:12],np.argsort(coef)[-12:]]; colors=["#d9534f" if coef[i]<0 else "#2ca58d" for i in idx]
    fig,ax=plt.subplots(figsize=(9,7)); ax.barh([features[i] for i in idx],coef[idx],color=colors); ax.set_title("Most influential TF-IDF features"); fig.tight_layout(); fig.savefig(output / "feature_importance.png",dpi=180); plt.close(fig)


if __name__ == "__main__": print(json.dumps(benchmark(), indent=2))

