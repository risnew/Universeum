import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Result of an adjustment code prediction."""
    predicted_code: int
    predicted_category: str
    confidence: float
    top_predictions: list[tuple[str, float]]


# Category mapping
CATEGORY_NAMES = {
    1: "Audience size/character",
    2: "Language",
    3: "Age",
    4: "Technical",
    5: "Content",
    6: "Environment",
    8: "Time",
    9: "Execution",
}

SWEDISH_STOPWORDS = {
    "och", "i", "att", "en", "ett", "det", "som", "på", "är", "av", "för",
    "med", "till", "den", "har", "de", "inte", "om", "men", "var",
    "jag", "från", "vi", "kan", "så", "eller", "vid", "nu", "när", "alla",
    "mycket", "också", "efter", "bara", "där", "sin", "dem",
    "utan", "då", "över", "får", "två", "här", "under", "sig", "ska",
    "blev", "varit", "samt", "hos", "vad", "andra", "sedan", "mellan",
    "några", "inom", "dessa", "många", "genom", "finns", "själv",
    "innan", "vilka", "kunde", "deras", "medan", "samma", "detta", "hade",
    "inga", "något", "hur", "dels", "göra", "bra", "ha", "vara", "dom",
    "lite", "sen", "mer", "fick", "var", "blev", "hade", "skulle",
}


def preprocess_text(text: str) -> str:
    """Clean and preprocess Swedish text."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\såäö]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in SWEDISH_STOPWORDS and len(w) > 2]
    return ' '.join(words)


class AdjustmentClassifier:
    """Simple TF-IDF based classifier for adjustment categories."""

    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        self.classes_ = None

    def train(self, texts: list[str], labels: list[int]) -> dict:
        """Train the classifier on labeled data."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Preprocess texts
        processed_texts = [preprocess_text(t) for t in texts]

        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(processed_texts) if t.strip()]
        processed_texts = [processed_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]

        # Vectorize
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=2,
        )
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)

        # Train classifier
        self.classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
        )
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        self.is_trained = True

        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)

        return {
            "accuracy": cv_scores.mean(),
            "std": cv_scores.std(),
            "n_samples": len(labels),
            "n_features": X.shape[1],
        }

    def predict(self, text: str) -> PredictionResult:
        """Predict adjustment category for new text."""
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train() first.")

        processed = preprocess_text(text)
        if not processed.strip():
            return PredictionResult(
                predicted_code=0,
                predicted_category="Unknown",
                confidence=0.0,
                top_predictions=[],
            )

        X = self.vectorizer.transform([processed])
        probabilities = self.classifier.predict_proba(X)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_code = self.classes_[predicted_idx]

        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            (CATEGORY_NAMES.get(self.classes_[i], f"Code {self.classes_[i]}"), probabilities[i])
            for i in top_indices
        ]

        return PredictionResult(
            predicted_code=int(predicted_code),
            predicted_category=CATEGORY_NAMES.get(predicted_code, f"Code {predicted_code}"),
            confidence=float(probabilities[predicted_idx]),
            top_predictions=top_predictions,
        )

    def save(self, path: str):
        """Save trained model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'classes_': self.classes_,
            }, f)

    def load(self, path: str):
        """Load trained model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.classes_ = data['classes_']
            self.is_trained = True


def load_training_data(path: str) -> tuple[list[str], list[int]]:
    """Load and prepare training data from CSV."""
    df = pd.read_csv(path)

    # Filter rows with both text and code
    df = df[df['Anpassningstext'].notna() & df['Anpassningskod'].notna()]

    texts = df['Anpassningstext'].tolist()
    labels = df['Anpassningskod'].astype(int).tolist()

    return texts, labels


def get_category_distribution(path: str) -> pd.DataFrame:
    """Get distribution of adjustment categories."""
    df = pd.read_csv(path)
    df = df[df['Anpassningskod'].notna()]

    counts = df['Anpassningskod'].value_counts().sort_index()

    result = pd.DataFrame({
        'code': counts.index.astype(int),
        'category': [CATEGORY_NAMES.get(int(c), f"Code {int(c)}") for c in counts.index],
        'count': counts.values,
    })

    return result
