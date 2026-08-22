FROM python:3.8-slim

WORKDIR /app

RUN pip install --no-cache-dir uvicorn==0.33.0 fastapi==0.115.8 civitdl==2.1.1 httpx==0.28.1 pytest==8.3.4 PyYAML==6.0.2

COPY . .
ENV PYTHONPATH=/app

# One worker only: the async download tasks live in a process-local dict, so
# extra workers each get their own copy and GET /status/{task_id} 404s
# whenever it lands on a worker that did not create the task.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7681", "--workers", "1", "--log-level", "warning"]
