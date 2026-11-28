"""Train and evaluate the configured recommender.

Run ``book-rec-train`` (or ``python -m book_recommendation.train``). Use
``--dry-run`` to validate inputs and report split sizes without fitting a model.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
from typing import Sequence

from book_recommendation.config import get_settings
from book_recommendation.data_loader import load_book_crossing_data, report_as_dict
from book_recommendation.data_split import split_report_as_dict, split_userwise
from book_recommendation.logger import configure_logging, get_logger
from book_recommendation.model_artifact import save_model
from book_recommendation.model_evaluation import evaluate_ranking
from book_recommendation.model_trainer import train_recommender


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hybrid book recommender")
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML settings file (relative paths are resolved from the project root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load and validate input data and create split statistics without fitting",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="skip optional MLflow run and artifact logging",
    )
    return parser.parse_args(argv)


def _log_mlflow(
    *,
    model_path: Path,
    params: dict[str, str | int | float],
    metrics: dict[str, float],
    experiment_name: str,
    logger: logging.Logger,
) -> None:
    if importlib.util.find_spec("mlflow") is None:
        logger.info("MLflow is not installed; skipping experiment tracking")
        return
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path), artifact_path="model")
    except Exception:
        logger.exception("MLflow logging failed; the local model artifact remains available")


def run_training(*, dry_run: bool = False, track_mlflow: bool = True) -> None:
    """Load, split, fit, evaluate, and atomically save one model candidate."""
    settings = get_settings()
    configure_logging(settings.logging.level, settings.logging.format)
    logger = get_logger(__name__)

    dataset = load_book_crossing_data(
        users_path=settings.users_path,
        books_path=settings.books_path,
        ratings_path=settings.ratings_path,
    )
    split = split_userwise(
        dataset.ratings,
        relevance_threshold=settings.evaluation.relevance_threshold,
        seed=settings.random_seed,
    )
    logger.info("Loaded Book-Crossing dataset: %s", report_as_dict(dataset.report))
    logger.info("Created user-wise data split: %s", split_report_as_dict(split.report))
    if len(split.implicit_train):
        logger.warning(
            "%d implicit zero-rating events are retained but not used by the current explicit-rating model",
            len(split.implicit_train),
        )

    if dry_run:
        logger.info("Dry run complete; no model was fitted or written")
        return
    if split.train.empty:
        raise RuntimeError("No explicit ratings remain in the training partition")
    if split.validation.empty:
        raise RuntimeError("No validation ratings are available; refusing to save an unevaluated model")

    model = train_recommender(
        dataset.books,
        split.train,
        similarity_neighbors=settings.training.similarity_neighbors,
        content_weight=settings.training.content_weight,
        collaborative_weight=settings.training.collaborative_weight,
        popularity_weight=settings.training.popularity_weight,
    )

    validation_metrics: dict[str, float] = {}
    minimum_recall = settings.quality_gates.min_recall_at_10
    validation = evaluate_ranking(model, split, partition="validation")
    validation_metrics = validation.metrics
    logger.info(
        "Validation metrics for %d users: %s",
        validation.evaluated_users,
        validation_metrics,
    )
    observed_recall = validation_metrics.get("recall@10", 0.0)
    if minimum_recall is not None and observed_recall < minimum_recall:
        raise RuntimeError(
            "Model failed the configured Recall@10 gate: "
            f"{observed_recall:.4f} < {minimum_recall:.4f}"
        )

    test_metrics: dict[str, float] = {}
    if not split.test.empty:
        test = evaluate_ranking(model, split, partition="test")
        test_metrics = test.metrics
        logger.info(
            "Final held-out test metrics for %d users: %s",
            test.evaluated_users,
            test_metrics,
        )
    else:
        logger.warning("Test split is empty; no final ranking metrics are available")

    save_model(model, settings.model_path)
    logger.info(
        "Saved model candidate with %d books and %d explicit-training users to %s",
        len(model.item_ids),
        len(model.user_ids),
        settings.model_path,
    )
    if track_mlflow:
        mlflow_metrics = {
            **{
                f"validation_{name.replace('@', '_at_')}": value
                for name, value in validation_metrics.items()
            },
            **{
                f"test_{name.replace('@', '_at_')}": value
                for name, value in test_metrics.items()
            },
        }
        _log_mlflow(
            model_path=settings.model_path,
            params={
                "random_seed": settings.random_seed,
                "similarity_neighbors": settings.training.similarity_neighbors,
                "content_weight": settings.training.content_weight,
                "collaborative_weight": settings.training.collaborative_weight,
                "popularity_weight": settings.training.popularity_weight,
                "training_books": len(model.item_ids),
                "training_users": len(model.user_ids),
                "training_ratings": len(split.train),
            },
            metrics=mlflow_metrics,
            experiment_name=settings.project,
            logger=logger,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.config is not None:
        os.environ["BOOK_REC_CONFIG"] = str(args.config)
        get_settings.cache_clear()
    run_training(dry_run=args.dry_run, track_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
