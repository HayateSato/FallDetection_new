FROM python:3.11-slim

WORKDIR /app

# Shared dependencies first — better layer caching
COPY shared_db/ shared_db/
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/
COPY model/ model/
# alembic.ini is needed when this image runs as db-migrate (alembic upgrade head)
COPY alembic.ini .

COPY inference_server/requirements.txt inference_server/requirements.txt
RUN pip install --no-cache-dir -r inference_server/requirements.txt

COPY inference_server/ inference_server/

EXPOSE 8001

# workers=1: Prometheus counters and asyncio timers are per-process
CMD ["uvicorn", "inference_server.server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
