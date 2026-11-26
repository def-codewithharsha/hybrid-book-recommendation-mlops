"""CLI training entry point: python -m book_recommendation.train"""
import logging

import pandas as pd

from book_recommendation.config import get_settings
from book_recommendation.model_artifact import save_model
from book_recommendation.model_trainer import train_recommender


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    books = pd.read_csv(settings.books_path)
    interactions = pd.read_csv(settings.interactions_path)
    model = train_recommender(books, interactions, settings.similarity_neighbors)
    save_model(model, settings.model_path)
    logging.info("Saved model with %d books and %d users to %s",
                 len(model.item_ids), len(model.user_ids), settings.model_path)
    try:
        import mlflow

        mlflow.set_experiment("hybrid-book-recommendation")
        with mlflow.start_run():
            mlflow.log_params({"books": len(model.item_ids), "users": len(model.user_ids),
                               "similarity_neighbors": settings.similarity_neighbors})
            mlflow.log_artifact(str(settings.model_path), artifact_path="model")
    except Exception as exc:
        logging.warning("MLflow tracking unavailable; local artifact was saved: %s", exc)


if __name__ == "__main__":
    main()
