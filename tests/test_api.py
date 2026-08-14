from pathlib import Path
import os
import pytest
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
os.environ["DATABASE_URL"] = f"sqlite:///{ROOT / 'test_voc.sqlite3'}"
from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as value: yield value


def test_health(client): assert client.get("/health").json()=={"status":"ok"}


def test_predict_and_validation(client):
    response=client.post("/predict",json={"text":"Excellent product and fast delivery"}); assert response.status_code==200; assert response.json()["sentiment"] in {"positive","negative"}
    assert client.post("/predict",json={"text":""}).status_code==422


def test_batch(client):
    response=client.post("/predict/batch",json={"texts":["Great service","Terrible broken product"]}); assert response.status_code==200 and response.json()["count"]==2

