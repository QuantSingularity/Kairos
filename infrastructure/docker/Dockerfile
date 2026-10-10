# Build from the project root:
#   docker build -f infrastructure/docker/Dockerfile -t kairos .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
#       -v "$PWD/reports:/app/reports" kairos all
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KAIROS_ROOT=/app

COPY infrastructure/requirements.txt infrastructure/requirements.txt
RUN pip install --no-cache-dir -r infrastructure/requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY code code
COPY scripts scripts

ENTRYPOINT ["python", "scripts/run.py"]
CMD ["all"]
