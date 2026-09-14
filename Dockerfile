FROM docker.m.daocloud.io/library/python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple/ -r requirements.txt
COPY . .
RUN python scripts/prepare_data.py && python -m src.benchmark
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8005"]
