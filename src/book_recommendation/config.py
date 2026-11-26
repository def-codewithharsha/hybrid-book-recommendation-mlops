"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOK_REC_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/book_recommender.joblib")
    books_path: Path = Path("data/raw/books.csv")
    interactions_path: Path = Path("data/raw/interactions.csv")
    log_level: str = "INFO"
    recommendation_limit: int = 50
    similarity_neighbors: int = 100


@lru_cache
def get_settings() -> Settings:
    return Settings()
