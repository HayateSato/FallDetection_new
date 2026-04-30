FROM python:3.11-slim

# Pin mlflow to match retrain/requirements.txt and inference_server/requirements.txt.
# Mismatched versions cause API drift (e.g. /api/2.0/mlflow/logged-models is 3.x only).
# Python 3.11 chosen because 3.12-slim strips setuptools, which mlflow still imports.
RUN pip install --no-cache-dir \
    mlflow==3.11.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

EXPOSE 5000
ENTRYPOINT ["mlflow"]
