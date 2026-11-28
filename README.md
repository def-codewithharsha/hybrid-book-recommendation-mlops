# Hybrid book recommendation

A Python 3.11 service scaffold for a hybrid book recommender. It combines user-item collaborative signals, TF-IDF book metadata similarity, and smoothed popularity ranking. The first version uses a compact in-memory sparse artifact suited to a single API replica; a catalog-scale deployment should move neighbor retrieval and model storage to a dedicated online store.

## Data contract

The Book-Crossing loader reads `Users.csv`, `Books.csv`, and `Ratings.csv` from `data/raw/`. It produces canonical `users`, `books`, and `ratings` dataframes. Rating `0` is retained as an implicit event and kept separate from explicit ratings `1`–`10`. The source contains no event timestamp. User location and age and the book image URLs are intentionally omitted from the loaded feature tables.

## Split and offline evaluation

`book_recommendation.data_split.split_userwise` deterministically assigns up to one held-out relevant rating to validation and one to test for each eligible user. By default relevance means an explicit rating of at least 7. Users retain at least one explicit training rating; users without a valid holdout remain in training. Zero ratings are returned separately as `implicit_train` and are not treated as explicit preference scores.

`book_recommendation.model_evaluation.evaluate_ranking` reports macro Precision@K, Recall@K, NDCG@K, MAP@K, recommendation catalog coverage, and held-out item warm-start coverage. Because there are no timestamps, this is a user-wise holdout evaluation, not a future-time simulation. The current recommender training path uses the explicit `train` partition; the implicit training signal still needs model integration.

## Local run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m book_recommendation.train
uvicorn book_recommendation.api:app --reload --app-dir src
```

Use `python -m book_recommendation.train --dry-run` to load and validate the configured CSVs and report data/split statistics without fitting a model. Select a different YAML file with `--config config/your-env.yaml`; `--no-mlflow` skips optional tracking.

## Exploratory analysis

The reproducible Book-Crossing EDA is in `notebooks/01_book_crossing_eda.ipynb`. Run it from the repository root after installing the project dependencies. It profiles data quality, implicit versus explicit ratings, user/book activity, catalog completeness, and interaction sparsity. The older `dump/expriments.ipynb` is an exploratory scratch notebook for Project Gutenberg metadata, not the EDA for the selected Book-Crossing dataset.

Settings load from `config/dev.yaml`; process environment variables override YAML values. Set `BOOK_REC_CONFIG` to select another YAML file, or use nested variables such as `BOOK_REC_PATHS__MODEL` and `BOOK_REC_LOGGING__LEVEL`. Training logs a run and artifact to MLflow when the tracking extra is installed; configure `MLFLOW_TRACKING_URI` to use a remote tracking server. Install tracking support with `pip install -e ".[tracking]"`.

## API

- `GET /health/live`: process liveness.
- `GET /health/ready`: readiness (503 until a model artifact loads).
- `GET /metrics`: basic model/catalog gauges; production monitoring should export request/error/latency counters to Prometheus or an OTEL collector.
- `POST /recommendations`: JSON `{"user_id":"u-1","limit":10}` or `{"book_id":"b-1","limit":10}`. A known user's previously rated books and the seed book are excluded.
- `GET /docs`: interactive OpenAPI documentation.

## Container

Train the artifact first, then:

```powershell
docker build -f Dockerfile -t hybrid-book-recommendation .
docker run --rm -p 8080:8080 -v "${PWD}/models:/app/models:ro" hybrid-book-recommendation
```

## Roadmap

1. Compare popularity, content, and collaborative baselines using the new user-wise ranking split; add quality gates after reviewing the first results.
2. Add representative tests and data validation, then CI lint/build/security checks and a reproducible container build.
3. Add a deployment target, model registry promotion/rollback, and request-level recommendation feedback instrumentation.
4. Introduce DVC when datasets need reproducible remote versioning. Add Airflow only when ingestion/training orchestration needs scheduled retries and backfills; the current batch CLI is intentionally standalone.
