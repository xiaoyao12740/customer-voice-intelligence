# Experiments / 实验记录

## Setup / 实验设置

- Dataset: UCI Sentiment Labelled Sentences, DOI `10.24432/C57604`
- After deduplication: 2,982 rows; 18 exact duplicate records removed
- Split: train 2,087, validation 447, test 448
- Stratification: sentiment and source; random seed 42
- Features: word and character TF-IDF, maximum 30,000 features
- Development: 5-fold stratified cross-validation on train; candidate comparison on validation
- Selection rule: highest validation macro F1 among non-dummy models with native `predict_proba`
- Test policy: after selection, refit on train+validation and evaluate the selected model once on test

数据处理、划分和模型参数均由代码固定，可从原始下载重新生成，不依赖手工修改的 CSV。

## Results / 结果

| Model | Validation Accuracy | Validation Macro F1 | 5-fold CV Macro F1 | Native probability |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8501 | 0.8501 | 0.8188 ± 0.0212 | yes |
| MultinomialNB | 0.8456 | 0.8456 | 0.8206 ± 0.0182 | yes |
| LinearSVC | 0.8277 | 0.8277 | 0.8236 ± 0.0213 | no |
| SGD Logistic | 0.8210 | 0.8209 | 0.8164 ± 0.0222 | yes |
| Dummy | 0.4989 | 0.3328 | 0.3337 ± 0.0003 | yes, excluded baseline |

完整的逐模型数值由 `outputs/benchmark.csv` 生成。该目录被忽略，以确保结果必须通过训练代码复现。

## Selection / 模型选择

Logistic Regression was selected before test evaluation because it had the highest validation macro F1 among candidates satisfying the native-probability product requirement. It was refit on train+validation, then evaluated once on test: macro F1 0.8482, ROC-AUC 0.9225, PR-AUC 0.9340, and Brier score 0.1164.

Logistic Regression 在满足原生概率要求的候选模型中取得最高验证集 Macro F1，因此在接触测试集结果之前即被选定。随后使用 train+validation 重训，并只在 test 上执行一次最终评估。原生概率不等同于已经校准，需结合 calibration curve 与 Brier score 解读。

## Reproduction / 复现

```bash
python scripts/prepare_data.py
python -m src.benchmark
```

Generated artifacts include metrics JSON, validation benchmark CSV, test error cases, confusion matrix, calibration curve, feature coefficients and dataset-distribution charts.
