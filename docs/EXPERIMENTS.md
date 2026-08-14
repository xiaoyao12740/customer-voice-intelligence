# Experiments / 实验记录

## Setup / 实验设置

- Dataset: UCI Sentiment Labelled Sentences, DOI `10.24432/C57604`
- After deduplication: 2,982 rows; 18 exact duplicate records removed
- Split: train 2,087, validation 447, test 448
- Stratification: sentiment and source; random seed 42
- Features: word and character TF-IDF, maximum 30,000 features
- Validation: 5-fold stratified cross-validation on the training split
- Test policy: the held-out test split is used for the reported final comparison

数据处理、划分和模型参数均由代码固定，可从原始下载重新生成，不依赖手工修改的 CSV。

## Results / 结果

| Model | Accuracy | Precision (+) | Recall (+) | F1 (+) | Macro F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LinearSVC | 0.8348 | — | — | — | 0.8348 | — | — |
| Logistic Regression | 0.8281 | 0.8210 | 0.8393 | 0.8300 | 0.8281 | 0.9148 | 0.9287 |
| MultinomialNB | 0.8237 | — | — | — | 0.8236 | — | — |
| SGD Logistic | 0.8214 | — | — | — | 0.8213 | — | — |
| Dummy | 0.5000 | — | — | — | 0.3333 | — | — |

完整的逐模型数值由 `outputs/benchmark.csv` 生成。该目录被忽略，以确保结果必须通过训练代码复现。

## Selection / 模型选择

LinearSVC achieved the best test macro F1. Logistic Regression was selected for the serving example because it provides calibrated-like native probabilities without an extra calibration stage and loses only 0.0067 macro F1 on this split. Confidence values should still not be interpreted as guaranteed correctness.

LinearSVC 的 Macro F1 最高；服务层选择 Logistic Regression，是为了直接输出概率并保持实现简洁。该概率用于产品演示，不代表严格校准后的真实正确率。

## Reproduction / 复现

```bash
python scripts/prepare_data.py
python -m src.benchmark
```

Generated artifacts include metrics JSON, benchmark CSV, error cases, confusion matrix, feature coefficients and dataset-distribution charts.
