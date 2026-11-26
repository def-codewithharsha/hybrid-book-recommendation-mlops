"""Deterministic user-wise holdout splits for Book-Crossing ranking evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitReport:
    relevance_threshold: int
    seed: int
    explicit_train_rows: int
    implicit_train_rows: int
    validation_rows: int
    test_rows: int
    validation_users: int
    test_users: int
    validation_cold_item_rows: int
    test_cold_item_rows: int


@dataclass
class RecommendationSplit:
    """Explicit train/validation/test rows and a separate implicit training set."""

    train: pd.DataFrame
    implicit_train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    report: SplitReport


def split_userwise(
    ratings: pd.DataFrame,
    *,
    relevance_threshold: int = 7,
    seed: int = 42,
) -> RecommendationSplit:
    """Hold out up to one relevant book for validation and one for testing per user.

    Relevant means an explicit rating greater than or equal to
    ``relevance_threshold``. Every held-out user retains at least one explicit
    training rating. Users with fewer than two explicit ratings, or no relevant
    ratings, remain in training only. Zero ratings are preserved separately in
    ``implicit_train`` and never enter explicit-rating ranking targets.

    The source has no timestamps, so this split is deterministic but does not
    simulate a future-time evaluation.
    """
    required = {"user_id", "book_id", "rating"}
    if missing := required - set(ratings.columns):
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")
    if not 1 <= relevance_threshold <= 10:
        raise ValueError("relevance_threshold must be between 1 and 10")

    frame = ratings.copy().reset_index(drop=True)
    frame["user_id"] = frame["user_id"].astype("string").str.strip()
    frame["book_id"] = frame["book_id"].astype("string").str.strip()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["user_id", "book_id", "rating"])
    frame = frame[frame["rating"].between(0, 10, inclusive="both")].reset_index(drop=True)
    if frame.duplicated(["user_id", "book_id"]).any():
        raise ValueError("ratings must contain at most one row per user_id and book_id")

    if "is_explicit" in frame:
        explicit_mask = frame["is_explicit"].fillna(False).astype(bool)
    else:
        explicit_mask = frame["rating"].gt(0)
    explicit = frame[explicit_mask].copy().reset_index(drop=True)
    implicit = frame[~explicit_mask].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    validation_positions: list[int] = []
    test_positions: list[int] = []
    for positions in explicit.groupby("user_id", sort=False).indices.values():
        user_positions = np.asarray(positions, dtype=np.int64)
        if len(user_positions) < 2:
            continue
        relevant_positions = user_positions[
            explicit.iloc[user_positions]["rating"].to_numpy() >= relevance_threshold
        ]
        if len(relevant_positions) == 0:
            continue
        shuffled = rng.permutation(relevant_positions)
        holdout_count = min(2, len(shuffled), len(user_positions) - 1)
        test_positions.append(int(shuffled[0]))
        if holdout_count == 2:
            validation_positions.append(int(shuffled[1]))

    held_out = set(validation_positions) | set(test_positions)
    train_positions = [i for i in range(len(explicit)) if i not in held_out]
    train = explicit.iloc[train_positions].reset_index(drop=True)
    validation = explicit.iloc[validation_positions].reset_index(drop=True)
    test = explicit.iloc[test_positions].reset_index(drop=True)

    train_items = set(train["book_id"].astype(str)) | set(implicit["book_id"].astype(str))
    validation_cold = int((~validation["book_id"].astype(str).isin(train_items)).sum())
    test_cold = int((~test["book_id"].astype(str).isin(train_items)).sum())
    report = SplitReport(
        relevance_threshold=relevance_threshold,
        seed=seed,
        explicit_train_rows=len(train),
        implicit_train_rows=len(implicit),
        validation_rows=len(validation),
        test_rows=len(test),
        validation_users=validation["user_id"].nunique(),
        test_users=test["user_id"].nunique(),
        validation_cold_item_rows=validation_cold,
        test_cold_item_rows=test_cold,
    )
    return RecommendationSplit(train, implicit, validation, test, report)


def split_report_as_dict(report: SplitReport) -> dict[str, int]:
    """Return split statistics in a JSON-serializable form."""
    return asdict(report)
