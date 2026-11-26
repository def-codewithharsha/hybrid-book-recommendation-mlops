"""Versioned, atomic local model artifact persistence."""
import os
import tempfile
from pathlib import Path

import joblib

from book_recommendation.model_trainer import HybridRecommender


def save_model(model: HybridRecommender, path: Path) -> None:
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
    return model
