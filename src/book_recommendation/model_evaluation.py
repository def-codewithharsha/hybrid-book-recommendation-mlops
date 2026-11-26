"""Offline top-K ranking metrics for recommenders implementing ``recommend``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from book_recommendation.data_split import RecommendationSplit


@dataclass
class RankingEvaluation:
    """Aggregate ranking metrics and per-user metric rows."""

    metrics: dict[str, float]
    per_user: pd.DataFrame
    evaluated_users: int


def _user_ranking_metrics(recommended: list[str], relevant: set[str], k: int) -> dict[str, float]:
    ranked = recommended[:k]
    hit_flags = np.asarray([item in relevant for item in ranked], dtype=np.float64)
    hits = float(hit_flags.sum())
    discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
    dcg = float(np.sum(hit_flags * discounts))
    ideal_length = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal_length + 2))))
    precision_at_hits = np.cumsum(hit_flags) / np.arange(1, len(hit_flags) + 1)
    average_precision = float(np.sum(precision_at_hits * hit_flags) / ideal_length) if ideal_length else 0.0
    return {
        f"precision@{k}": hits / k,
        f"recall@{k}": hits / len(relevant) if relevant else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        f"map@{k}": average_precision,
    }


def evaluate_ranking(
    recommender: Any,
    split: RecommendationSplit,
    *,
    partition: str = "test",
    k_values: tuple[int, ...] = (5, 10),
    relevance_threshold: int | None = None,
) -> RankingEvaluation:
    """Evaluate held-out explicit preferences using macro top-K ranking metrics.

    The recommendation object's ``recommend(user_id=..., limit=...)`` method
    must return records containing ``book_id``. Metrics are macro-averaged over
    users with at least one relevant held-out book. Catalog coverage and held-out
    item warm-start coverage are included in ``metrics``.
    """
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be 'validation' or 'test'")
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    target = getattr(split, partition)
    if target.empty:
        raise ValueError(f"the {partition} partition is empty")

    threshold = relevance_threshold or split.report.relevance_threshold
    relevant_rows = target[target["rating"] >= threshold]
    if relevant_rows.empty:
        raise ValueError(f"the {partition} partition has no ratings >= {threshold}")

    max_k = max(k_values)
    train_items = set(split.train["book_id"].astype(str)) | set(split.implicit_train["book_id"].astype(str))
    catalog_items = set(getattr(recommender, "item_ids", train_items))
    user_rows: list[dict[str, float | str | int]] = []
    all_recommended: set[str] = set()
    target_items: set[str] = set()
    warm_target_items: set[str] = set()

    for user_id, group in relevant_rows.groupby("user_id", sort=False):
        relevant = set(group["book_id"].astype(str))
        target_items.update(relevant)
        warm_target_items.update(relevant & train_items)
        results = recommender.recommend(user_id=str(user_id), limit=max_k)
        recommended = [str(row["book_id"]) for row in results if "book_id" in row]
        all_recommended.update(recommended)
        row: dict[str, float | str | int] = {
            "user_id": str(user_id),
            "relevant_count": len(relevant),
        }
        for k in k_values:
            row.update(_user_ranking_metrics(recommended, relevant, k))
        user_rows.append(row)

    per_user = pd.DataFrame(user_rows)
    metrics = {
        name: float(per_user[name].mean())
        for name in per_user.columns
        if name.startswith(("precision@", "recall@", "ndcg@", "map@"))
    }
    metrics["catalog_coverage@max_k"] = (
        len(all_recommended & catalog_items) / len(catalog_items) if catalog_items else 0.0
    )
    metrics["heldout_item_warm_coverage"] = (
        len(warm_target_items) / len(target_items) if target_items else 0.0
    )
    return RankingEvaluation(metrics=metrics, per_user=per_user, evaluated_users=len(per_user))
