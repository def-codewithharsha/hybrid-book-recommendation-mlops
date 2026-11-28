"""Train and serve sparse-content, latent-collaborative, popularity hybrid ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_ARTIFACT_VERSION = 2


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).ravel()
    low, high = float(values.min(initial=0)), float(values.max(initial=0))
    if high - low <= np.finfo(np.float32).eps:
        return np.zeros_like(values)
    return (values - low) / (high - low)


@dataclass
class HybridRecommender:
    """Immutable-at-serving-time model state for user and seed-book ranking."""

    books: pd.DataFrame
    content_matrix: csr_matrix
    user_item: csr_matrix
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    popularity: np.ndarray
    content_weight: float
    collaborative_weight: float
    popularity_weight: float
    content_profile_min_rating: int
    artifact_version: int = MODEL_ARTIFACT_VERSION

    def recommend(
        self,
        user_id: str | None = None,
        book_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scores = self.popularity_weight * self.popularity.copy()
        content_scores = np.zeros(len(self.item_ids), dtype=np.float32)
        user_key = str(user_id) if user_id is not None else None
        book_key = str(book_id) if book_id is not None else None
        user_position = self.user_index.get(user_key) if user_key is not None else None

        if user_position is not None:
            latent_scores = self.user_factors[user_position] @ self.item_factors.T
            scores += self.collaborative_weight * _minmax_scale(latent_scores)

            profile = self.user_item.getrow(user_position).copy()
            profile.data = np.maximum(profile.data - self.content_profile_min_rating + 1, 0)
            profile.eliminate_zeros()
            if profile.nnz:
                weighted_content = profile @ self.content_matrix
                content_scores = (
                    weighted_content @ self.content_matrix.T
                ).toarray().ravel()
                content_scores /= float(profile.data.sum())

        if book_key is not None and book_key in self.item_index:
            seed_position = self.item_index[book_key]
            seed_scores = (
                self.content_matrix.getrow(seed_position) @ self.content_matrix.T
            ).toarray().ravel()
            content_scores = np.maximum(content_scores, seed_scores)
        scores += self.content_weight * content_scores

        eligible = np.ones(len(self.item_ids), dtype=bool)
        if user_position is not None:
            eligible[self.user_item.getrow(user_position).indices] = False
        if book_key is not None and book_key in self.item_index:
            eligible[self.item_index[book_key]] = False
        candidate_positions = np.flatnonzero(eligible)
        order = candidate_positions[np.argsort(scores[candidate_positions])[::-1][:limit]]
        result = self.books.loc[[self.item_ids[position] for position in order]].copy()
        result["score"] = scores[order].astype(float)
        return result.reset_index(drop=True).to_dict(orient="records")


def train_recommender(
    books: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    latent_factors: int,
    max_content_features: int,
    popularity_prior_strength: float,
    content_profile_min_rating: int,
    content_weight: float,
    collaborative_weight: float,
    popularity_weight: float,
    random_seed: int,
) -> HybridRecommender:
    """Fit sparse TF-IDF content features, explicit-rating SVD, and smoothed popularity."""
    weights = (content_weight, collaborative_weight, popularity_weight)
    if any(weight < 0 or weight > 1 for weight in weights) or not np.isclose(sum(weights), 1.0):
        raise ValueError("content, collaborative, and popularity weights must be in [0, 1] and sum to 1")
    if latent_factors < 1 or max_content_features < 1 or popularity_prior_strength <= 0:
        raise ValueError("latent_factors, max_content_features, and prior strength must be positive")
    if not 1 <= content_profile_min_rating <= 10:
        raise ValueError("content_profile_min_rating must be between 1 and 10")
    if missing := {"book_id", "title"} - set(books.columns):
        raise ValueError(f"books is missing required columns: {sorted(missing)}")
    if missing := {"user_id", "book_id", "rating"} - set(interactions.columns):
        raise ValueError(f"interactions is missing required columns: {sorted(missing)}")

    books = books.copy()
    interactions = interactions.copy()
    books["book_id"] = books["book_id"].astype("string").str.strip()
    interactions["book_id"] = interactions["book_id"].astype("string").str.strip()
    interactions["user_id"] = interactions["user_id"].astype("string").str.strip()
    books["title"] = books["title"].astype("string").str.strip()
    books = books.dropna(subset=["book_id", "title"])
    books = books[books["title"].ne("")].drop_duplicates("book_id").reset_index(drop=True)
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions = interactions.dropna(subset=["user_id", "book_id", "rating"])
    interactions = interactions[
        interactions["rating"].between(1, 10, inclusive="both")
        & interactions["rating"].mod(1).eq(0)
    ]
    interactions = interactions[interactions["book_id"].isin(books["book_id"])].reset_index(drop=True)
    if books.empty:
        raise ValueError("at least one valid book is required")
    if interactions.empty:
        raise ValueError("at least one explicit user-book rating is required")
    if interactions.duplicated(["user_id", "book_id"]).any():
        raise ValueError("interactions must contain at most one explicit rating per user and book")

    for column in ("authors", "publisher", "description", "categories", "publication_year"):
        if column not in books:
            books[column] = ""
    content_columns = ["title", "authors", "publisher", "description", "categories", "publication_year"]
    content_text = books[content_columns].fillna("").astype("string").agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_content_features,
        dtype=np.float32,
    )
    try:
        content_matrix = vectorizer.fit_transform(content_text).tocsr()
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        vectorizer = TfidfVectorizer(max_features=max_content_features, dtype=np.float32)
        content_matrix = vectorizer.fit_transform(content_text).tocsr()

    item_ids = books["book_id"].astype(str).tolist()
    item_index = {book_id: position for position, book_id in enumerate(item_ids)}
    user_ids = sorted(interactions["user_id"].astype(str).unique().tolist())
    user_index = {user_id: position for position, user_id in enumerate(user_ids)}
    row_indices = interactions["user_id"].astype(str).map(user_index).to_numpy()
    col_indices = interactions["book_id"].astype(str).map(item_index).to_numpy()
    ratings = interactions["rating"].to_numpy(dtype=np.float32)
    shape = (len(user_ids), len(item_ids))
    user_item = csr_matrix((ratings, (row_indices, col_indices)), shape=shape, dtype=np.float32)

    user_means = interactions.groupby("user_id", sort=False)["rating"].mean()
    centered_values = ratings - interactions["user_id"].map(user_means).to_numpy(dtype=np.float32)
    centered = csr_matrix((centered_values, (row_indices, col_indices)), shape=shape, dtype=np.float32)
    component_count = min(latent_factors, min(shape) - 1)
    if component_count >= 1:
        factorizer = TruncatedSVD(n_components=component_count, random_state=random_seed)
        user_factors = factorizer.fit_transform(centered).astype(np.float32, copy=False)
        item_factors = factorizer.components_.T.astype(np.float32, copy=False)
    else:
        user_factors = np.zeros((len(user_ids), 1), dtype=np.float32)
        item_factors = np.zeros((len(item_ids), 1), dtype=np.float32)

    counts = np.bincount(col_indices, minlength=len(item_ids)).astype(np.float32)
    sums = np.bincount(col_indices, weights=ratings, minlength=len(item_ids)).astype(np.float32)
    prior = float(ratings.mean())
    popularity = (sums + popularity_prior_strength * prior) / (counts + popularity_prior_strength)
    popularity = _minmax_scale(popularity)

    books = books.set_index("book_id", drop=False)
    return HybridRecommender(
        books=books,
        content_matrix=content_matrix,
        user_item=user_item,
        user_factors=user_factors,
        item_factors=item_factors,
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        popularity=popularity,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
        popularity_weight=popularity_weight,
        content_profile_min_rating=content_profile_min_rating,
    )
