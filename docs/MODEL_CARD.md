# Model Card / 模型卡

## Intended use / 预期用途

The model classifies short English customer-review sentences as positive or negative. It is a portfolio and decision-support demonstration, suitable for prototyping routing and analytics—not an autonomous customer-service decision maker.

模型用于对简短英文客户评论进行正负面分类，适合原型验证、路由与统计展示，不应替代人工客服决策。

## Model / 模型

- TF-IDF feature union
- Logistic Regression classifier
- Version: 1.0.0
- Held-out macro F1: 0.8281
- ROC-AUC: 0.9148
- Positive-class F1 95% bootstrap CI: [0.7884, 0.8632]

## Limitations / 局限

- English only; binary labels only; no neutral or mixed-sentiment class.
- Small, short-sentence dataset collected from Amazon, IMDb and Yelp.
- Domain shift, sarcasm, negation and evolving language may reduce performance.
- Probability output has not undergone a dedicated calibration study.
- Topic and risk outputs are deterministic keyword rules and must not be interpreted as learned predictions.

## Responsible use / 负责任使用

Do not use the output to deny service, punish employees, prioritize safety complaints without review, or make legal/financial decisions. Monitor drift, audit errors by source and language, protect submitted customer text, and keep a human review path for consequential use.
