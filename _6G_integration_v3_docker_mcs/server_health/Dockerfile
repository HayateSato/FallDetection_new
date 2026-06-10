FROM python:3.11-slim

WORKDIR /app

COPY server_health/requirements.txt server_health/requirements.txt
RUN pip install --no-cache-dir -r server_health/requirements.txt

COPY server_health/ server_health/

EXPOSE 8006
CMD ["python", "-m", "server_health.main"]
