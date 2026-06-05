FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY mock_app/requirements.txt mock_app/requirements.txt
RUN pip install --no-cache-dir -r mock_app/requirements.txt

# Shared ML pipeline (needed by influx_fetcher and influx_writer)
# Note: ml_pipeline is also used by fall_dashboard — keep it at the repo root
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/

COPY mock_app/ mock_app/

# Patient confirmation popup browser UI
EXPOSE 8005

CMD ["python", "-m", "mock_app.main"]
