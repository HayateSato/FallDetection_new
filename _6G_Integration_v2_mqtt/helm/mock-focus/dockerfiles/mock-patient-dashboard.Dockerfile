FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx

COPY helm/mock-focus/dockerfiles/mock_patient_dashboard.py ./server.py
COPY helm/mock-focus/dockerfiles/dashboard.html ./dashboard.html

EXPOSE 8090

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]
