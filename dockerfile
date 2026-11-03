# Dockerfile.update
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY watttime_client.py watttime_pipeline.py daily_update.py ./

# Non-root user
RUN useradd -m -s /bin/bash updater
USER updater

ENTRYPOINT ["python", "daily_update.py"]