# Business Case / 业务场景

## Problem / 问题

Customer-feedback queues are difficult to summarize manually. This project demonstrates a first-pass system that groups feedback, surfaces negative sentiment and makes routing evidence visible.

人工阅读大量反馈成本高。本项目演示如何先进行情感分类、主题归类和透明风险提示，再把结果交给人员复核。

## Example workflow / 示例流程

1. Receive one review or a CSV batch.
2. Predict positive/negative sentiment and probability.
3. Apply auditable keyword rules for topic and complaint-risk hints.
4. Persist the result for aggregate Dashboard analysis.
5. Route important cases to a human reviewer.

## Success criteria / 成功标准

- Reproducible model evidence instead of a screenshot-only demo.
- Fast local inference and a documented HTTP interface.
- Explainable limitations and a human review path.
- Clear upgrade path: neutral/multilingual labels, calibrated thresholds, supervised topic classification, drift monitoring and authenticated deployment.
