from pathlib import Path
import os
import pytest
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
os.environ["DATABASE_URL"] = f"sqlite:///{ROOT / 'test_voc.sqlite3'}"
from api.main import app
from database.models import Prediction, Review
from database.service import get_engine
from sqlalchemy import func, select
from sqlalchemy.orm import Session


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as value: yield value


def test_health(client): assert client.get("/health").json()=={"status":"ok"}


def test_predict_and_validation(client):
    response=client.post("/predict",json={"text":"Excellent product and fast delivery"}); assert response.status_code==200; assert response.json()["sentiment"] in {"positive","negative"}
    assert client.post("/predict",json={"text":""}).status_code==422


def test_batch(client):
    engine=get_engine(os.environ["DATABASE_URL"])
    with Session(engine) as session:
        reviews_before=session.scalar(select(func.count()).select_from(Review)) or 0
        predictions_before=session.scalar(select(func.count()).select_from(Prediction)) or 0
    response=client.post("/predict/batch",json={"texts":["Great service","Terrible broken product"]})
    assert response.status_code==200 and response.json()["count"]==2
    assert all("review_id" in item for item in response.json()["predictions"])
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Review))==reviews_before+2
        assert session.scalar(select(func.count()).select_from(Prediction))==predictions_before+2
