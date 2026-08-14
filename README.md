# Customer Voice Intelligence / 客户之声智能分析平台

[![CI](https://github.com/xiaoyao12740/customer-voice-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaoyao12740/customer-voice-intelligence/actions/workflows/ci.yml)

[中文](#中文说明) · [English](#english)

An end-to-end, evidence-driven NLP portfolio project for sentiment analysis, topic routing, complaint-risk triage, batch analytics and API serving.

一个端到端、以真实实验为依据的 NLP 项目，覆盖情感分析、主题路由、投诉风险分级、批量分析与 API 服务。

![Dashboard overview](docs/screenshots/dashboard-overview.png)

## 中文说明

### 项目亮点

- 使用 UCI Sentiment Labelled Sentences 的 2,982 条去重英文评论，而非自造演示数据。
- 对比 Dummy、MultinomialNB、SGD、Logistic Regression 与 LinearSVC 五种方案，并保留交叉验证结果。
- 提供 Accuracy、Precision、Recall、F1、Macro F1、ROC-AUC、PR-AUC、95% Bootstrap 区间、混淆矩阵和误差样本。
- 提供单条/批量预测、FastAPI、Streamlit Dashboard、SQLite 持久化，以及可选 MySQL。
- Docker Compose 一次启动 API、Dashboard 和 MySQL；GitHub Actions 自动重跑数据处理、训练和测试。
- 主题与风险分级是可解释的关键词规则，不伪装成训练模型；风险结果仅用于演示辅助判断。

### 真实实验结果

固定随机种子 42；按 `sentiment + source` 分层切分为 2,087/447/448 条训练、验证和测试数据。候选模型只在验证集比较；产品要求模型提供原生概率，因此在符合条件的模型中按验证集 Macro F1 选择 Logistic Regression。随后用训练集和验证集重训，最终测试集只评估一次。

| 模型 | 验证集 Accuracy | 验证集 Macro F1 | 5-fold CV Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | **85.01%** | **85.01%** | 81.88% ± 2.12% |
| MultinomialNB | 84.56% | 84.56% | 82.06% ± 1.82% |
| LinearSVC | 82.77% | 82.77% | 82.36% ± 2.13% |
| SGD Logistic | 82.10% | 82.09% | 81.64% ± 2.22% |
| Dummy | 49.89% | 33.28% | 33.37% ± 0.03% |

锁定 Logistic Regression 后的最终测试结果为：Macro F1 84.82%、ROC-AUC 92.25%、PR-AUC 93.40%、Brier score 0.1164；正类 F1 的 95% Bootstrap 区间为 80.95%–88.25%。原生 `predict_proba` 不是“已经校准”的同义词，因此仓库同时输出 calibration curve 和 Brier score，置信度仍只用于辅助判断。

![Model comparison](docs/screenshots/model-comparison.png)

![Calibration curve](docs/screenshots/calibration-curve.png)

### 界面与模型证据

| 单条预测 | FastAPI 文档 |
|---|---|
| ![Prediction](docs/screenshots/dashboard-prediction.png) | ![API docs](docs/screenshots/api-docs.png) |

| 混淆矩阵 | 特征解释 |
|---|---|
| ![Confusion matrix](docs/screenshots/confusion-matrix.png) | ![Feature importance](docs/screenshots/feature-importance.png) |

### 架构

```mermaid
flowchart LR
  D["UCI 数据集"] --> P["清洗、去重、分层切分"]
  P --> T["TF-IDF + 模型基准"]
  T --> M["Logistic Regression 模型包"]
  M --> S["共享推理服务"]
  S --> A["FastAPI :8005"]
  S --> U["Streamlit :8505"]
  S --> R["透明主题/风险规则"]
  A --> DB["SQLite / MySQL"]
  U --> DB
```

### 快速开始

需要 Python 3.11+。本机开发可复用项目群根目录的 `.venv`，仓库本身不提交环境目录。

```powershell
pip install -r requirements.txt
python scripts/prepare_data.py
python -m src.benchmark
uvicorn api.main:app --host 0.0.0.0 --port 8005
```

另开终端启动界面：

```powershell
streamlit run dashboard/app.py --server.port 8505
```

- Dashboard: `http://localhost:8505`
- API 文档: `http://localhost:8005/docs`
- 健康检查: `http://localhost:8005/health`

Docker 运行：

```powershell
docker compose up --build
```

Published image / 已发布镜像：

```powershell
docker pull ghcr.io/xiaoyao12740/customer-voice-intelligence:v1.0.0
docker run --rm -p 8005:8005 ghcr.io/xiaoyao12740/customer-voice-intelligence:v1.0.0
```

测试与复现：

```powershell
pytest -q
python -m src.benchmark
```

### 数据来源与限制

数据来自 [UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences)，DOI `10.24432/C57604`，许可为 CC BY 4.0。原始 3,000 条 Amazon、IMDb、Yelp 英文句子经确定性流程去除 18 条重复记录；原始压缩包与处理结果由脚本下载/生成，不提交 Git。

当前版本只做英文正/负二分类，没有中性标签；数据规模较小且句子较短。主题路由与风险分级是规则基线，不是监督学习结果，也不应用于自动拒绝、处罚或客服决策。Transformer、MLflow 和云部署留作有真实实验依据后的迭代方向。

## English

### What this project demonstrates

- A reproducible pipeline built on 2,982 deduplicated, real review sentences from UCI.
- Five-model benchmarking with stratified cross-validation and a held-out test set.
- Complete evaluation: accuracy, precision, recall, F1, macro F1, ROC-AUC, PR-AUC, bootstrap confidence interval, confusion matrix and error export.
- Single and batch inference through Streamlit and FastAPI, with SQLite by default and optional MySQL persistence.
- Docker Compose packaging and GitHub Actions reproduction.
- Explicit separation between the trained sentiment model and transparent topic/risk rules.

### Reproduce it

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python -m src.benchmark
pytest -q
uvicorn api.main:app --host 0.0.0.0 --port 8005
streamlit run dashboard/app.py --server.port 8505
```

Candidate models are compared on validation only. Logistic Regression wins validation macro F1 among models that satisfy the native-probability product requirement; it is then refit on train+validation and evaluated once on the untouched test split. See [experiment details](docs/EXPERIMENTS.md), [model card](docs/MODEL_CARD.md), and [API examples](docs/API.md).

### Repository layout

```text
api/                 FastAPI routes and schemas
dashboard/           Streamlit application
database/            SQLAlchemy storage and MySQL schema
data/                Dataset provenance; generated data is ignored
docs/                 Evidence, screenshots and technical notes
scripts/              Deterministic data preparation
src/                  Preprocessing, training, evaluation and inference
tests/                Unit and API tests
.github/workflows/    Reproducible CI pipeline
```

### License

Project code is released under the MIT License. The UCI dataset retains its own CC BY 4.0 license and attribution requirements.
