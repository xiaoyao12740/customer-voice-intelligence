from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
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


def classification_metrics(y_true, prediction, probabilities) -> dict:
    binary = (np.asarray(y_true) == "positive").astype(int)
    return {"accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, pos_label="positive", zero_division=0),
        "recall": recall_score(y_true, prediction, pos_label="positive", zero_division=0),
        "f1": f1_score(y_true, prediction, pos_label="positive", zero_division=0),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "weighted_f1": f1_score(y_true, prediction, average="weighted"),
        "roc_auc": roc_auc_score(binary, probabilities),
        "pr_auc": average_precision_score(binary, probabilities),
        "brier_score": brier_score_loss(binary, probabilities)}


def benchmark() -> dict:
    config = load_config(); processed = Path(config["data"]["processed_dir"]); output = Path(config["outputs"]["root"]); output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(processed / "train.csv"); validation = pd.read_csv(processed / "validation.csv"); test = pd.read_csv(processed / "test.csv")
    all_data = pd.concat([train, validation, test], ignore_index=True); eda = run_eda(all_data, output / "eda")
    x_train, y_train = train["clean_text"], train["sentiment"]
    x_validation, y_validation = validation["clean_text"], validation["sentiment"]
    x_test, y_test = test["clean_text"], test["sentiment"]
    results, fitted = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(config["project"]["seed"]))
    for name in MODEL_NAMES:
        model = build_pipeline(name, int(config["model"]["max_features"])); started = time.perf_counter(); model.fit(x_train, y_train); train_seconds = time.perf_counter() - started
        started = time.perf_counter(); prediction = model.predict(x_validation); inference_ms = (time.perf_counter() - started) * 1000 / len(validation)
        probabilities = scores_from_model(model, x_validation)
        cv_scores = cross_val_score(build_pipeline(name, int(config["model"]["max_features"])), x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        row = {"model": name, **classification_metrics(y_validation, prediction, probabilities),
            "native_probability": hasattr(model, "predict_proba"), "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(), "train_seconds": train_seconds, "inference_ms_per_row": inference_ms}
        results.append(row); fitted[name] = model
    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    table.to_csv(output / "benchmark.csv", index=False)

    # The product contract requires native probabilities. Select only among
    # eligible candidates on validation; the test split remains untouched.
    require_probability = bool(config["model"].get("require_native_probability", True))
    eligible = table[table.model != "dummy"]
    if require_probability: eligible = eligible[eligible.native_probability]
    if eligible.empty: raise ValueError("No probability-capable production candidate was evaluated")
    production_name = str(eligible.iloc[0].model)
    development = pd.concat([train, validation], ignore_index=True)
    production = build_pipeline(production_name, int(config["model"]["max_features"]))
    production.fit(development["clean_text"], development["sentiment"])
    started = time.perf_counter(); prediction = production.predict(x_test)
    test_inference_ms = (time.perf_counter() - started) * 1000 / len(test)
    probabilities = scores_from_model(production, x_test)
    test_metrics = classification_metrics(y_test, prediction, probabilities)
    test_metrics["inference_ms_per_row"] = test_inference_ms
    model_path = Path(config["outputs"]["model_path"]); model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": production, "model_name": production_name, "version": config["project"]["model_version"], "labels": ["negative", "positive"]}; joblib.dump(bundle, model_path, compress=3)
    errors = test.assign(prediction=prediction, positive_probability=probabilities); errors[errors.sentiment != errors.prediction].to_csv(output / "error_analysis.csv", index=False)
    report = {"selection_protocol": "candidates compared on validation; selected model refit on train+validation; test evaluated once",
        "selection_metric": config["model"].get("selection_metric", "validation_macro_f1"),
        "require_native_probability": require_probability, "production_model": production_name,
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "validation_results": results, "test_metrics": test_metrics, "f1_95_ci": bootstrap_ci(y_test, prediction),
        "classification_report": classification_report(y_test, prediction, output_dict=True), "eda": eda}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_results(table, y_test, prediction, probabilities, production, output)
    evidence_dir = Path("docs/screenshots"); evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_files = {"model_comparison.png": "model-comparison.png",
        "confusion_matrix.png": "confusion-matrix.png", "feature_importance.png": "feature-importance.png",
        "calibration_curve.png": "calibration-curve.png"}
    for source, destination in evidence_files.items(): shutil.copyfile(output / source, evidence_dir / destination)
    return report


def plot_results(table, y_test, prediction, probabilities, model, output):
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.barh(table.model, table.macro_f1, color="#3478bf"); ax.set_xlim(.4, 1); ax.set_xlabel("Validation Macro F1"); ax.set_title("Validation model selection"); ax.grid(axis="x", alpha=.2); fig.tight_layout(); fig.savefig(output / "model_comparison.png", dpi=180); plt.close(fig)
    matrix = confusion_matrix(y_test, prediction, labels=["negative", "positive"]); fig, ax = plt.subplots(figsize=(5, 4)); image=ax.imshow(matrix,cmap="Blues"); fig.colorbar(image,ax=ax); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["negative","positive"],yticklabels=["negative","positive"],xlabel="Predicted",ylabel="Actual",title="Final Test Confusion Matrix")
    for i in range(2):
        for j in range(2): ax.text(j,i,matrix[i,j],ha="center",va="center")
    fig.tight_layout(); fig.savefig(output / "confusion_matrix.png", dpi=180); plt.close(fig)
    observed, predicted = calibration_curve((np.asarray(y_test) == "positive").astype(int), probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4)); ax.plot(predicted, observed, marker="o", label="model"); ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal"); ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title="Test calibration curve"); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output / "calibration_curve.png", dpi=180); plt.close(fig)
    features = model.named_steps["features"].get_feature_names_out(); coef=model.named_steps["classifier"].coef_[0]; idx=np.r_[np.argsort(coef)[:12],np.argsort(coef)[-12:]]; colors=["#d9534f" if coef[i]<0 else "#2ca58d" for i in idx]
    fig,ax=plt.subplots(figsize=(9,7)); ax.barh([features[i] for i in idx],coef[idx],color=colors); ax.set_title("Most influential TF-IDF features"); fig.tight_layout(); fig.savefig(output / "feature_importance.png",dpi=180); plt.close(fig)


if __name__ == "__main__": print(json.dumps(benchmark(), indent=2))
