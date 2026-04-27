FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY app.py .
COPY run_pipeline.py .
COPY weekly_retrain.py .
COPY validate_weekly.py .
COPY schedule_config.py .

RUN mkdir -p logs

EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000/ || exit 1

CMD ["python", "app.py"]
