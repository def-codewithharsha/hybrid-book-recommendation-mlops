"""Load and validate the Book-Crossing users, books, and ratings CSV files.

The source files are read as Latin-1 and identifiers are kept as strings so ISBNs
retain leading zeroes. User attributes are intentionally not loaded: user ID is
all the recommender currently needs, and location/age are not required features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


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


def _read_csv(path: Path, required: set[str], *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    try:
        columns = pd.read_csv(path, encoding="latin-1", nrows=0).columns
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        frame = pd.read_csv(
            path,
            encoding="latin-1",
            dtype="string",
            usecols=usecols,
            low_memory=False,
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not parse dataset file {path}: {exc}") from exc
    return frame


def _normalize_identifier(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def load_book_crossing_data(
    data_dir: str | Path = "data/raw",
    *,
    users_file: str = "Users.csv",
    books_file: str = "Books.csv",
    ratings_file: str = "Ratings.csv",
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
    root = Path(data_dir)
    user_source = _read_csv(root / users_file, {"User-ID"}, usecols=["User-ID"])
    book_source = _read_csv(
        root / books_file,
        {"ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"},
        usecols=["ISBN", "Book-Title", "Book-Author", "Publisher", "Year-Of-Publication"],
    )
    rating_source = _read_csv(
        root / ratings_file,
        {"User-ID", "ISBN", "Book-Rating"},
        usecols=["User-ID", "ISBN", "Book-Rating"],
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
    books["title"] = books["title"].replace("", pd.NA)
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
    valid_rating = ratings["rating"].between(0, 10, inclusive="both")
    invalid_ratings_removed = int((~valid_rating.fillna(False)).sum())
    ratings = ratings[valid_rating.fillna(False)].dropna(subset=["user_id", "book_id"])
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
    return BookCrossingData(users=users, books=books, ratings=ratings, report=report)


def report_as_dict(report: DataQualityReport) -> dict[str, int]:
    """Return the data-quality summary in a JSON-serializable form."""
    return asdict(report)
