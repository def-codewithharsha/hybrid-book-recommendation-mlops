"""Train a compact hybrid recommender from books and user-book interactions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class HybridRecommender:
    books: pd.DataFrame
    item_similarity: Any
    user_item: Any
    user_ids: list[str]
    item_ids: list[str]
    popularity: np.ndarray
    content_weight: float = 0.35
    collaborative_weight: float = 0.45
    popularity_weight: float = 0.20

    def recommend(self, user_id: str | None = None, book_id: str | None = None,
                  limit: int = 10) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        scores = self.popularity_weight * self.popularity.copy()
        known_user = user_id is not None and str(user_id) in self.user_ids
        if known_user:
            row = self.user_item[self.user_ids.index(str(user_id))]
            scores += self.collaborative_weight * (row @ self.item_similarity).toarray().ravel()
        if book_id is not None and str(book_id) in self.item_ids:
            idx = self.item_ids.index(str(book_id))
            scores += self.content_weight * self.item_similarity[idx].toarray().ravel()
        eligible = np.ones(len(self.item_ids), dtype=bool)
        if known_user:
            eligible &= self.user_item[self.user_ids.index(str(user_id))].toarray().ravel() == 0
        if book_id is not None and str(book_id) in self.item_ids:
            eligible[self.item_ids.index(str(book_id))] = False
        candidates = np.flatnonzero(eligible)
        order = candidates[np.argsort(scores[candidates])[::-1][:limit]]
        result = self.books.set_index("book_id").loc[[self.item_ids[i] for i in order]].reset_index()
        result["score"] = scores[order].astype(float)
        return result.to_dict(orient="records")


def train_recommender(books: pd.DataFrame, interactions: pd.DataFrame,
                      similarity_neighbors: int = 100) -> HybridRecommender:
    if missing := {"book_id", "title"} - set(books.columns):
        raise ValueError(f"books is missing required columns: {sorted(missing)}")
    if missing := {"user_id", "book_id", "rating"} - set(interactions.columns):
        raise ValueError(f"interactions is missing required columns: {sorted(missing)}")
    books, interactions = books.copy(), interactions.copy()
    books["book_id"] = books.book_id.astype(str)
    interactions["book_id"] = interactions.book_id.astype(str)
    interactions["user_id"] = interactions.user_id.astype(str)
    books = books.drop_duplicates("book_id").reset_index(drop=True)
    interactions["rating"] = pd.to_numeric(interactions.rating, errors="coerce")
    interactions = interactions.dropna(subset=["rating"])
    interactions = interactions[interactions.book_id.isin(books.book_id)]
    if books.empty:
        raise ValueError("at least one book is required")
    for col in ("authors", "description", "categories"):
        if col not in books:
            books[col] = ""
    text = books[["title", "authors", "description", "categories"]].fillna("").agg(" ".join, axis=1)
    text = text.where(text.str.strip().ne(""), books.book_id)
    content = cosine_similarity(
        TfidfVectorizer(stop_words="english", max_features=100_000).fit_transform(text),
        dense_output=False,
    ).tocsr()
    item_ids = books.book_id.tolist()
    item_index = {item: i for i, item in enumerate(item_ids)}
    user_ids = sorted(interactions.user_id.unique().tolist())
    user_index = {user: i for i, user in enumerate(user_ids)}
    if interactions.empty:
        user_item = csr_matrix((0, len(item_ids)), dtype=np.float32)
        popularity = np.zeros(len(item_ids), dtype=np.float32)
    else:
        rows = interactions.user_id.map(user_index).to_numpy()
        cols = interactions.book_id.map(item_index).to_numpy()
        values = interactions.rating.to_numpy(dtype=np.float32)
        user_item = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))
        counts = np.bincount(cols, minlength=len(item_ids)).astype(np.float32)
        sums = np.bincount(cols, weights=np.clip(values, 0, None), minlength=len(item_ids)).astype(np.float32)
        prior = float(np.mean(np.clip(values, 0, None)))
        popularity = (sums + 5 * prior) / (counts + 5)
        lo, hi = popularity.min(), popularity.max()
        if hi > lo:
            popularity = (popularity - lo) / (hi - lo)
    raw = (user_item.T @ user_item).tocsr()
    item_cf = raw.copy()
    item_cf.setdiag(0)
    norms = np.sqrt(np.asarray(raw.diagonal()).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    item_cf = item_cf.multiply(inv[:, None]).multiply(inv[None, :]).tocsr()
    if similarity_neighbors > 0:
        for row in range(item_cf.shape[0]):
            start, end = item_cf.indptr[row], item_cf.indptr[row + 1]
            if end - start > similarity_neighbors:
                keep = np.argpartition(item_cf.data[start:end], -similarity_neighbors)[-similarity_neighbors:]
                mask = np.zeros(end - start, dtype=bool)
                mask[keep] = True
                item_cf.data[start:end][~mask] = 0
        item_cf.eliminate_zeros()
    return HybridRecommender(books, (0.5 * content + 0.5 * item_cf).tocsr(),
                             user_item, user_ids, item_ids, popularity)
