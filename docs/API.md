# API / 接口说明

Start / 启动：

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8005
```

Interactive OpenAPI documentation is available at `http://localhost:8005/docs`.

## Endpoints

- `GET /health` — service and model readiness
- `GET /model-info` — model metadata
- `POST /predict` — one feedback item
- `POST /predict/batch` — multiple feedback items
- `GET /analytics/summary` — persisted prediction summary

Example / 示例：

```bash
curl -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The delivery was late and the product arrived broken."}'
```

The response contains sentiment, confidence, topic, transparent risk score, matched keywords and model version. Input validation and exact schemas are published in OpenAPI.
