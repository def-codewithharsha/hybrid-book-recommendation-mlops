"""Typed application configuration from YAML with environment overrides."""

from __future__ import annotations

import os
from functools import lru_cache
from math import isclose
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


def find_project_root() -> Path:
    """Find the source checkout/container root independently of launch directory."""
    module_path = Path(__file__).resolve()
    candidates = [*module_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "config" / "dev.yaml").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate project root containing config/dev.yaml and pyproject.toml")


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else find_project_root() / path


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_ConfigModel):
    model: Path
    users: Path
    books: Path
    ratings: Path


class TrainingConfig(_ConfigModel):
    similarity_neighbors: int = Field(ge=1)
    content_weight: float = Field(ge=0, le=1)
    collaborative_weight: float = Field(ge=0, le=1)
    popularity_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> TrainingConfig:
        total = self.content_weight + self.collaborative_weight + self.popularity_weight
        if not isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError("content, collaborative, and popularity weights must sum to 1")
        return self


class EvaluationConfig(_ConfigModel):
    relevance_threshold: int = Field(ge=1, le=10)


class APIConfig(_ConfigModel):
    host: str
    port: int = Field(ge=1, le=65535)
    recommendation_limit: int = Field(ge=1, le=50)


class LoggingConfig(_ConfigModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: Literal["json", "text"]


class QualityGatesConfig(_ConfigModel):
    min_recall_at_10: float | None = Field(ge=0, le=1)


class Settings(BaseSettings):
    """Validated settings; environment variables override the selected YAML file.

    Nested overrides use the ``BOOK_REC_`` prefix and ``__`` delimiter, for
    example ``BOOK_REC_PATHS__MODEL`` or ``BOOK_REC_TRAINING__SIMILARITY_NEIGHBORS``.
    """

    model_config = SettingsConfigDict(
        env_prefix="BOOK_REC_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    project: str
    environment: Literal["development", "staging", "production"]
    random_seed: int = Field(ge=0)
    paths: PathsConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    api: APIConfig
    logging: LoggingConfig
    quality_gates: QualityGatesConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_value = os.getenv("BOOK_REC_CONFIG")
        config_path = Path(config_value) if config_value else Path("config/dev.yaml")
        if not config_path.is_absolute():
            config_path = find_project_root() / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=config_path)
        # Earlier sources have higher priority: explicit values, process env,
        # .env, YAML configuration, then mounted secret files.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings,
        )

    @property
    def model_path(self) -> Path:
        return _resolve_path(self.paths.model)

    @property
    def users_path(self) -> Path:
        return _resolve_path(self.paths.users)

    @property
    def books_path(self) -> Path:
        return _resolve_path(self.paths.books)

    @property
    def ratings_path(self) -> Path:
        return _resolve_path(self.paths.ratings)

    @property
    def interactions_path(self) -> Path:
        """Compatibility alias while the training entry point is being migrated."""
        return self.ratings_path

    @property
    def similarity_neighbors(self) -> int:
        return self.training.similarity_neighbors

    @property
    def recommendation_limit(self) -> int:
        return self.api.recommendation_limit

    @property
    def log_level(self) -> str:
        return self.logging.level

    @property
    def log_format(self) -> str:
        return self.logging.format


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache validated settings for the current process."""
    return Settings()
