FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces expects the app on port 7860. Override with -e PORT=...
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.getenv('PORT','7860'))" || exit 1

CMD ["sh", "-c", "uvicorn data_cleaning_env.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
