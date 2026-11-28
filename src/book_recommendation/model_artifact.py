"""Atomic, checksummed persistence for trusted recommender model artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from book_recommendation.model_trainer import MODEL_ARTIFACT_VERSION, HybridRecommender

MANIFEST_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 1024 * 1024


def _manifest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model(model: HybridRecommender) -> None:
    if model.__dict__.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError("Refusing to save a model with an unsupported artifact version")
    item_count = len(model.item_ids)
    user_count = len(model.user_ids)
    if len(model.item_index) != item_count or len(model.user_index) != user_count:
        raise ValueError("Model ID lists and lookup indexes have inconsistent sizes")
    if any(model.item_index.get(book_id) != index for index, book_id in enumerate(model.item_ids)):
        raise ValueError("Book ID lookup index is inconsistent with the catalog")
    if any(model.user_index.get(user_id) != index for index, user_id in enumerate(model.user_ids)):
        raise ValueError("User ID lookup index is inconsistent with the user list")
    if len(model.books) != item_count or "book_id" not in model.books:
        raise ValueError("Book metadata does not match the model catalog")
    if model.content_matrix.shape[0] != item_count:
        raise ValueError("Content matrix does not match the model catalog")
    if model.user_item.shape != (user_count, item_count):
        raise ValueError("User-item matrix does not match model users and catalog")
    if model.user_factors.ndim != 2 or model.user_factors.shape[0] != user_count:
        raise ValueError("User factor matrix has an invalid shape")
    if model.item_factors.ndim != 2 or model.item_factors.shape[0] != item_count:
        raise ValueError("Item factor matrix has an invalid shape")
    if model.user_factors.shape[1] != model.item_factors.shape[1]:
        raise ValueError("User and item latent factor dimensions do not match")
    if model.popularity.shape != (item_count,):
        raise ValueError("Popularity scores do not match the model catalog")
    numeric_arrays = (model.user_factors, model.item_factors, model.popularity)
    if any(not np.isfinite(array).all() for array in numeric_arrays):
        raise ValueError("Model contains non-finite values")
    if not np.isfinite(model.content_matrix.data).all() or not np.isfinite(model.user_item.data).all():
        raise ValueError("Sparse model matrices contain non-finite values")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_model(model: HybridRecommender, path: Path) -> Path:
    """Write a model atomically and return its JSON integrity manifest path."""
    _validate_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            joblib.dump(model, stream, compress=3)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "model_type": "HybridRecommender",
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifact_file": path.name,
        "artifact_size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "catalog_books": len(model.item_ids),
        "training_users": len(model.user_ids),
        "latent_factors": int(model.item_factors.shape[1]),
        "content_features": int(model.content_matrix.shape[1]),
    }
    manifest_path = _manifest_path(path)
    _atomic_json(manifest_path, manifest)
    return manifest_path


def load_model(path: Path) -> HybridRecommender:
    """Verify a trusted local artifact and its manifest before deserializing it.

    Joblib uses pickle under the hood. Only load artifacts from a trusted source;
    the checksum detects accidental corruption but is not a signature.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}. Train a model before serving.")
    manifest_path = _manifest_path(path)
    if not manifest_path.is_file():
        raise ValueError(f"Model manifest not found: {manifest_path}; retrain the model")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Model manifest must contain a JSON object: {manifest_path}")

    if manifest.get("model_type") != "HybridRecommender":
        raise ValueError(f"Unsupported model type in manifest: {manifest_path}")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported model manifest version at {manifest_path}")
    if manifest.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Incompatible model artifact version in {manifest_path}; retrain the model")
    if manifest.get("artifact_file") != path.name:
        raise ValueError(f"Model manifest points to a different artifact: {manifest_path}")
    if manifest.get("artifact_size_bytes") != path.stat().st_size:
        raise ValueError(f"Model artifact size does not match its manifest: {path}")
    expected_hash = manifest.get("sha256")
    actual_hash = _sha256(path)
    if not isinstance(expected_hash, str) or not hmac.compare_digest(expected_hash, actual_hash):
        raise ValueError(f"Model artifact checksum does not match its manifest: {path}")

    # joblib/pickle deserialization can execute code. The artifact must come from
    # trusted, access-controlled storage even when its checksum is valid.
    model = joblib.load(path)
    if not isinstance(model, HybridRecommender):
        raise ValueError(f"Unsupported model artifact at {path}")
    _validate_model(model)
    expected_metadata = {
        "catalog_books": len(model.item_ids),
        "training_users": len(model.user_ids),
        "latent_factors": int(model.item_factors.shape[1]),
        "content_features": int(model.content_matrix.shape[1]),
    }
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"Model dimensions do not match manifest metadata: {manifest_path}")
    return model
