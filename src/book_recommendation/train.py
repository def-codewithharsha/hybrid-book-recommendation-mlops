"""CLI training entry point: python -m book_recommendation.train"""
import pandas as pd

from book_recommendation.config import get_settings
from book_recommendation.logger import configure_logging, get_logger
from book_recommendation.model_artifact import save_model
from book_recommendation.model_trainer import train_recommender


def main() -> None:
    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    logger = get_logger(__name__)
    books = pd.read_csv(settings.books_path)
    interactions = pd.read_csv(settings.interactions_path)
    model = train_recommender(
        books,
        interactions,
        similarity_neighbors=settings.training.similarity_neighbors,
        content_weight=settings.training.content_weight,
        collaborative_weight=settings.training.collaborative_weight,
        popularity_weight=settings.training.popularity_weight,
    )
    save_model(model, settings.model_path)
    logger.info("Saved model with %d books and %d users to %s",
                len(model.item_ids), len(model.user_ids), settings.model_path)
    try:
        import mlflow

        mlflow.set_experiment("hybrid-book-recommendation")
        with mlflow.start_run():
            mlflow.log_params({"books": len(model.item_ids), "users": len(model.user_ids),
                               "similarity_neighbors": settings.similarity_neighbors})
            mlflow.log_artifact(str(settings.model_path), artifact_path="model")
    except Exception as exc:
        logger.warning("MLflow tracking unavailable; local artifact was saved: %s", exc)


if __name__ == "__main__":
    main()
