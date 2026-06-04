FROM python:3.11-slim

WORKDIR /app

COPY shared_db/__init__.py shared_db/__init__.py
COPY shared_db/db/__init__.py shared_db/db/__init__.py
COPY shared_db/db/models.py shared_db/db/models.py
COPY shared_db/db/session.py shared_db/db/session.py
COPY ml_pipeline/ ml_pipeline/
COPY config/ config/

COPY fall_dashboard/requirements.txt fall_dashboard/requirements.txt
RUN pip install --no-cache-dir -r fall_dashboard/requirements.txt

COPY fall_dashboard/ fall_dashboard/

EXPOSE 8002

CMD ["python", "-m", "fall_dashboard.main"]
