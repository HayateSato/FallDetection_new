FROM python:3.11-slim

WORKDIR /app

COPY ml_dashboard/requirements.txt ml_dashboard/requirements.txt
RUN pip install --no-cache-dir -r ml_dashboard/requirements.txt

COPY ml_dashboard/ ml_dashboard/

# retrain.py and shared_db are needed because ml_dashboard spawns retrain as subprocess
# and reads MLflow registry via the shared client.
COPY retrain/ retrain/
COPY shared_db/ shared_db/
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/
COPY model/ model/

EXPOSE 8004
CMD ["python", "-m", "ml_dashboard.main"]
