FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables
ENV OPENENV_PORT=7860
ENV HOST=0.0.0.0

# Expose port (HuggingFace standard)
EXPOSE 7860

# Run OpenEnv server
CMD ["uvicorn", "data_cleaning_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]