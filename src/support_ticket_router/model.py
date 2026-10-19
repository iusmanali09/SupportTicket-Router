"""Model training, serialization, evaluation, and ticket prediction."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import prepare_text, validate_ticket_text
from .taxonomy import CATEGORIES, CATEGORY_TO_TEAM

MODEL_VERSION = 1


def build_model(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=100_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed)),
    ])


def train(texts: list[str], labels: list[str], seed: int = 42) -> Pipeline:
    model = build_model(seed)
    model.fit(texts, labels)
    return model


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"version": MODEL_VERSION, "categories": list(CATEGORIES), "model": model}, path)


def load_model(path: Path) -> Pipeline:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("version") != MODEL_VERSION:
        raise ValueError("Unsupported or invalid model artifact")
    model = artifact.get("model")
    if not hasattr(model, "predict_proba") or set(model.classes_) != set(artifact.get("categories", [])):
        raise ValueError("Model artifact has an invalid classifier or category list")
    return model


def predict(model: Pipeline, subject: str, description: str) -> dict:
    subject, description = validate_ticket_text(subject, description)
    probabilities = model.predict_proba([prepare_text(subject, description)])[0]
    scores = {str(label): float(score) for label, score in zip(model.classes_, probabilities)}
    category = max(scores, key=scores.get)
    return {
        "category": category,
        "team": CATEGORY_TO_TEAM[category],
        "confidence": scores[category],
        "category_scores": scores,
    }


def evaluate(model: Pipeline, texts: list[str], labels: list[str]) -> dict:
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "per_category": classification_report(labels, predictions, labels=CATEGORIES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=CATEGORIES).tolist(),
        "categories": list(CATEGORIES),
    }
