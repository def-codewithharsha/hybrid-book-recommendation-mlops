FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 BOOK_REC_MODEL_PATH=/app/models/book_recommender.joblib
WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/models && chown -R appuser:appuser /app
USER 10001:10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8080/health/live', timeout=3)" || exit 1
CMD ["uvicorn", "book_recommendation.api:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "src"]
