"""Versioned, atomic local model artifact persistence."""
import os
import tempfile
from pathlib import Path

import joblib

from book_recommendation.model_trainer import MODEL_ARTIFACT_VERSION, HybridRecommender


def save_model(model: HybridRecommender, path: Path) -> None:
    if model.__dict__.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Refusing to save unsupported model artifact version {model.artifact_version}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        joblib.dump(model, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_model(path: Path) -> HybridRecommender:
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}. Run `python -m book_recommendation.train` first.")
    model = joblib.load(path)
    if not isinstance(model, HybridRecommender):
        raise ValueError(f"Unsupported model artifact at {path}")
    if model.__dict__.get("artifact_version") != MODEL_ARTIFACT_VERSION:
        raise ValueError(f"Incompatible model artifact version at {path}; retrain with the current code")
    required_state = ("content_matrix", "user_factors", "item_factors", "user_index", "item_index")
    if any(not hasattr(model, field) for field in required_state):
        raise ValueError(f"Incomplete model artifact at {path}; retrain with the current code")
    return model
