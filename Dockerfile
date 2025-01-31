FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7681", "--workers", "4", "--log-level", "warning"]
