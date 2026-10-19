"""Dataset loading, validation, preprocessing, and reproducible data splits."""

from pathlib import Path
import csv
import re

from sklearn.model_selection import train_test_split

from .taxonomy import CATEGORIES

REQUIRED_COLUMNS = ("subject", "description", "category")
TEXT_MAX_CHARS = 20_000


def validate_ticket_text(subject: str, description: str) -> tuple[str, str]:
    """Validate user-supplied text and return trimmed values."""
    if not isinstance(subject, str) or not isinstance(description, str):
        raise ValueError("subject and description must be strings")
    subject, description = subject.strip(), description.strip()
    if not subject:
        raise ValueError("subject must not be empty")
    if not description:
        raise ValueError("description must not be empty")
    if len(subject) > TEXT_MAX_CHARS or len(description) > TEXT_MAX_CHARS:
        raise ValueError(f"subject and description must be at most {TEXT_MAX_CHARS} characters each")
    return subject, description


def prepare_text(subject: str, description: str) -> str:
    """Combine fields and normalize whitespace while preserving useful punctuation."""
    subject, description = validate_ticket_text(subject, description)
    return re.sub(r"\s+", " ", f"{subject} {description}").strip()


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load a UTF-8 CSV with subject, description, and category columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. See DATASET.md for preparation steps.")
    texts, labels = [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(REQUIRED_COLUMNS)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                text = prepare_text(row["subject"], row["description"])
            except ValueError as exc:
                raise ValueError(f"Invalid ticket at CSV row {row_number}: {exc}") from exc
            label = (row["category"] or "").strip()
            if label not in CATEGORIES:
                raise ValueError(f"Unknown category {label!r} at CSV row {row_number}; allowed: {', '.join(CATEGORIES)}")
            texts.append(text)
            labels.append(label)
    if not texts:
        raise ValueError("Dataset has no ticket rows")
    return texts, labels


def split_dataset(texts: list[str], labels: list[str], seed: int = 42):
    """Return stratified train/validation/test splits (70/15/15)."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have equal length")
    from collections import Counter

    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 3:
        raise ValueError("Each of at least two categories needs 3 or more examples for stratified splits")
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
