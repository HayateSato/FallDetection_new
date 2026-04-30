FROM python:3.11-slim

WORKDIR /app

COPY local_dev/mock_focus/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY local_dev/mock_focus/fhir_server.py ./mock_focus/fhir_server.py

EXPOSE 8003

CMD ["uvicorn", "mock_focus.fhir_server:app", "--host", "0.0.0.0", "--port", "8003"]
