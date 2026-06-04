FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY local_dev/mock_app/requirements.txt local_dev/mock_app/requirements.txt
RUN pip install --no-cache-dir -r local_dev/mock_app/requirements.txt

# Shared ML pipeline (needed by influx_fetcher and influx_writer)
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/

# mock_app source (includes __init__.py so python -m local_dev.mock_app.main works)
COPY local_dev/ local_dev/

# Patient confirmation popup browser UI
EXPOSE 8005

CMD ["python", "-m", "local_dev.mock_app.main"]
