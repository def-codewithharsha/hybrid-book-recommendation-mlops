"""Load and validate the Book-Crossing users, books, and ratings CSV files.

The source files are read as Latin-1 and identifiers are kept as strings so ISBNs
retain leading zeroes. User attributes are intentionally not loaded: user ID is
all the recommender currently needs, and location/age are not required features.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DataQualityReport:
    users_source_rows: int
    books_source_rows: int
    ratings_source_rows: int
    users_missing_id_removed: int
    books_missing_id_or_title_removed: int
    ratings_missing_key_removed: int
    duplicate_users_removed: int
    duplicate_books_removed: int
    duplicate_user_book_ratings_removed: int
    invalid_ratings_removed: int
    ratings_with_unknown_users_removed: int
    ratings_with_unknown_books_removed: int
    implicit_zero_ratings: int
    explicit_ratings: int
    users_loaded: int
    books_loaded: int
    ratings_loaded: int


@dataclass
class BookCrossingData:
    """Canonical dataframes and a summary of rows removed during loading."""

    users: pd.DataFrame
    books: pd.DataFrame
    ratings: pd.DataFrame
    report: DataQualityReport


def _read_csv(
    path: Path,
    required: set[str],
    *,
    usecols: list[str] | None = None,
    encoding: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    try:
        columns = pd.read_csv(path, encoding=encoding, nrows=0).columns
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame = pd.read_csv(
            path,
            encoding=encoding,
            dtype="string",
            usecols=usecols,
        )
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise ValueError(f"Could not parse dataset file {path}: {exc}") from exc
    return frame


def _normalize_identifier(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def load_book_crossing_data(
    data_dir: str | Path | None = None,
    *,
    users_path: str | Path | None = None,
    books_path: str | Path | None = None,
    ratings_path: str | Path | None = None,
    users_file: str = "Users.csv",
    books_file: str = "Books.csv",
    ratings_file: str = "Ratings.csv",
    encoding: str = "latin-1",
) -> BookCrossingData:
    """Load the Book-Crossing files into canonical user, book, and rating tables.

    Returned schemas:

    - users: ``user_id``
    - books: ``book_id``, ``title``, ``authors``, ``publisher``,
      ``publication_year``
    - ratings: ``user_id``, ``book_id``, ``rating``, ``is_explicit``

    Book-Crossing rating 0 is preserved as an implicit interaction and marked
    ``is_explicit=False``. Ratings 1 through 10 are explicit ratings. No event
    timestamp exists in the source data.
    """
    explicit_paths = (users_path, books_path, ratings_path)
    if any(path is not None for path in explicit_paths):
        if not all(path is not None for path in explicit_paths):
            raise ValueError("users_path, books_path, and ratings_path must be provided together")
        if data_dir is not None:
            raise ValueError("provide either data_dir or all three explicit paths, not both")
        input_users, input_books, input_ratings = map(Path, explicit_paths)
    elif data_dir is None:
        from book_recommendation.config import get_settings

        settings = get_settings()
        input_users, input_books, input_ratings = (
            settings.users_path,
            settings.books_path,
            settings.ratings_path,
        )
    else:
        root = Path(data_dir)
        input_users, input_books, input_ratings = (
            root / users_file,
            root / books_file,
            root / ratings_file,
        )

    user_source = _read_csv(
        input_users, {"User-ID"}, usecols=["User-ID"], encoding=encoding
    )
    book_source = _read_csv(
        input_books,
        {"ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"},
        usecols=["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"],
        encoding=encoding,
    )
    rating_source = _read_csv(
        input_ratings,
        {"User-ID", "ISBN", "Book-Rating"},
        usecols=["User-ID", "ISBN", "Book-Rating"],
        encoding=encoding,
    )

    user_source_rows, book_source_rows, rating_source_rows = (
        len(user_source), len(book_source), len(rating_source)
    )

    users = pd.DataFrame({"user_id": _normalize_identifier(user_source["User-ID"])})
    users = users.dropna(subset=["user_id"])
    users_missing_id_removed = user_source_rows - len(users)
    users_before_deduplication = len(users)
    users = users.drop_duplicates("user_id").reset_index(drop=True)
    duplicate_users_removed = users_before_deduplication - len(users)

    books = pd.DataFrame(
        {
            "book_id": _normalize_identifier(book_source["ISBN"]),
            "title": book_source["Book-Title"].astype("string").str.strip(),
            "authors": book_source["Book-Author"].astype("string").str.strip(),
            "publisher": book_source["Publisher"].astype("string").str.strip(),
            "publication_year": pd.to_numeric(
                book_source["Year-Of-Publication"], errors="coerce"
            ).astype("Int64"),
        }
    )
    for column in ("title", "authors", "publisher"):
        books[column] = books[column].str.strip().replace("", pd.NA)
    plausible_year = books["publication_year"].between(1450, 2100, inclusive="both")
    books.loc[~plausible_year.fillna(False), "publication_year"] = pd.NA
    books = books.dropna(subset=["book_id", "title"])
    books_missing_id_or_title_removed = book_source_rows - len(books)
    books_before_deduplication = len(books)
    books = books.drop_duplicates("book_id")
    duplicate_books_removed = books_before_deduplication - len(books)
    books = books.reset_index(drop=True)

    ratings = pd.DataFrame(
        {
            "user_id": _normalize_identifier(rating_source["User-ID"]),
            "book_id": _normalize_identifier(rating_source["ISBN"]),
            "rating": pd.to_numeric(rating_source["Book-Rating"], errors="coerce"),
        }
    )
    valid_keys = ratings["user_id"].notna() & ratings["book_id"].notna()
    ratings_missing_key_removed = int((~valid_keys).sum())
    ratings = ratings[valid_keys]
    valid_rating = (
        ratings["rating"].between(0, 10, inclusive="both")
        & ratings["rating"].mod(1).eq(0)
    )
    invalid_ratings_removed = int((~valid_rating.fillna(False)).sum())
    ratings = ratings[valid_rating.fillna(False)].copy()
    before_deduplication = len(ratings)
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last")
    duplicate_user_book_ratings_removed = before_deduplication - len(ratings)

    known_users = set(users["user_id"])
    known_books = set(books["book_id"])
    unknown_user_mask = ~ratings["user_id"].isin(known_users)
    ratings_with_unknown_users_removed = int(unknown_user_mask.sum())
    ratings = ratings[~unknown_user_mask]
    unknown_book_mask = ~ratings["book_id"].isin(known_books)
    ratings_with_unknown_books_removed = int(unknown_book_mask.sum())
    ratings = ratings[~unknown_book_mask].copy()
    ratings["rating"] = ratings["rating"].astype("int8")
    ratings["is_explicit"] = ratings["rating"].gt(0)
    ratings = ratings.reset_index(drop=True)

    if users.empty:
        raise ValueError("No valid users remain after loading and validation")
    if books.empty:
        raise ValueError("No valid books remain after loading and validation")
    if ratings.empty:
        raise ValueError("No valid ratings remain after loading and referential checks")

    report = DataQualityReport(
        users_source_rows=user_source_rows,
        books_source_rows=book_source_rows,
        ratings_source_rows=rating_source_rows,
        users_missing_id_removed=users_missing_id_removed,
        books_missing_id_or_title_removed=books_missing_id_or_title_removed,
        ratings_missing_key_removed=ratings_missing_key_removed,
        duplicate_users_removed=duplicate_users_removed,
        duplicate_books_removed=duplicate_books_removed,
        duplicate_user_book_ratings_removed=duplicate_user_book_ratings_removed,
        invalid_ratings_removed=invalid_ratings_removed,
        ratings_with_unknown_users_removed=ratings_with_unknown_users_removed,
        ratings_with_unknown_books_removed=ratings_with_unknown_books_removed,
        implicit_zero_ratings=int(ratings["rating"].eq(0).sum()),
        explicit_ratings=int(ratings["is_explicit"].sum()),
        users_loaded=len(users),
        books_loaded=len(books),
        ratings_loaded=len(ratings),
    )
    removed_rows = {
        name: count
        for name, count in asdict(report).items()
        if name.endswith("_removed") and count
    }
    if removed_rows:
        logger.warning("Invalid or unmatched source rows were removed: %s", removed_rows)
    else:
        logger.info("All source rows passed identifier, rating, and referential checks")
    return BookCrossingData(users=users, books=books, ratings=ratings, report=report)


def report_as_dict(report: DataQualityReport) -> dict[str, int]:
    """Return the data-quality summary in a JSON-serializable form."""
    return asdict(report)
