FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      numpy==1.26.4 \
      pandas==2.2.2 \
      polars==1.35.1 \
      uvicorn==0.37.0 \
      pyyaml \
      pydantic \
      pyarrow

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
