# FeatureForge product image: REST API + dashboard (http://localhost:8000/ui).
#   docker build -f docker/featureforge.Dockerfile -t featureforge .
#   docker run -p 8000:8000 -v ff_data:/data featureforge
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    FF_WORK_DIR=/data \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY featureforge ./featureforge
COPY data_cleaning_env ./data_cleaning_env
COPY agents ./agents
RUN pip install --no-deps . \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT','8000'))" || exit 1

CMD ["sh", "-c", "uvicorn featureforge.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${FF_WORKERS:-1}"]
