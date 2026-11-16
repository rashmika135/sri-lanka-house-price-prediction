FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-docker.txt .

RUN pip install --no-cache-dir --no-compile -r requirements-docker.txt

COPY app/ ./app/
COPY src/features.py src/__init__.py ./src/
COPY frontend/ ./frontend/
COPY models/house_price_model.pkl ./models/

USER 10001

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]