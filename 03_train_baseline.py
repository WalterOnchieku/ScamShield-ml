
"""
File: 03_train_baseline.py

Purpose:
Train and evaluate the first ScamShield KE text-classification model.

Pipeline:
1. Load the prepared training and testing datasets.
2. Convert message text into TF-IDF features.
3. Train a class-weighted Logistic Regression classifier.
4. Evaluate predictions on the held-out test set.
5. Save the model and a text evaluation report.

Outputs:
- models/scamshield_baseline.joblib
- reports/baseline_evaluation.txt

This is a baseline for SMS ham/spam classification, not a
validated detector for every type of scam.
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "scamshield_baseline.joblib"
REPORT_PATH = REPORT_DIR / "baseline_evaluation.txt"

LABELS = ["ham", "spam"]
RANDOM_STATE = 42


def load_split(path: Path) -> pd.DataFrame:
    """Load a prepared dataset and validate its required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run ml/01_dataset_audit.py and "
            "ml/02_prepare_dataset.py first."
        )

    df = pd.read_csv(
        path,
        dtype={"label": "string", "message": "string"},
    )

    if not {"label", "message"}.issubset(df.columns):
        raise ValueError(f"Missing required columns in {path}")

    if df[["label", "message"]].isna().any().any():
        raise ValueError(f"Missing values found in {path}")

    if not set(df["label"].unique()).issubset(set(LABELS)):
        raise ValueError(f"Unexpected labels found in {path}")

    if set(df["label"].unique()) != set(LABELS):
        raise ValueError(f"Both ham and spam must be present in {path}")

    if df["message"].str.strip().eq("").any():
        raise ValueError(f"Empty messages found in {path}")

    return df


def build_pipeline() -> Pipeline:
    """Create the TF-IDF and Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    """Train the baseline, evaluate it, and save its artifacts."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    test_df = load_split(TEST_PATH)

    # Verify that exact messages do not cross the train/test boundary.
    overlap = set(train_df["message"]) & set(test_df["message"])

    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} overlapping messages."
        )

    X_train = train_df["message"]
    y_train = train_df["label"]

    X_test = test_df["message"]
    y_test = test_df["label"]

    model = build_pipeline()

    print("Training TF-IDF + Logistic Regression...")
    print(f"Training messages: {len(train_df):,}")
    print(f"Testing messages:  {len(test_df):,}")

    model.fit(X_train, y_train)

    print("Training completed. Evaluating held-out test data...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Rows represent actual labels; columns represent predicted labels.
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    report_text = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    ham_correct = int(matrix[0, 0])
    ham_missed = int(matrix[0, 1])
    spam_missed = int(matrix[1, 0])
    spam_correct = int(matrix[1, 1])

    results = f"""
=================================================================
SCAMSHIELD KE — BASELINE MODEL EVALUATION
=================================================================
Generated (UTC): {datetime.now(timezone.utc).isoformat()}
Model: TF-IDF + Logistic Regression
Scikit-learn version: {sklearn.__version__}
Random state: {RANDOM_STATE}

1. DATA
-----------------------------------------------------------------
Training messages: {len(train_df):,}
Testing messages: {len(test_df):,}
Exact train/test message overlap: {len(overlap)}

2. MODEL CONFIGURATION
-----------------------------------------------------------------
TF-IDF n-grams: unigrams and bigrams
Minimum document frequency: 2
Sublinear term frequency: enabled
Classifier: Logistic Regression
Class weights: balanced
Maximum iterations: 1000

3. OVERALL METRIC
-----------------------------------------------------------------
Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)

4. CLASSIFICATION REPORT
-----------------------------------------------------------------
{report_text}

5. CONFUSION MATRIX
-----------------------------------------------------------------
Rows = actual labels; columns = predicted labels

                 Predicted ham   Predicted spam
Actual ham       {ham_correct:>13}   {ham_missed:>15}
Actual spam      {spam_missed:>13}   {spam_correct:>15}

6. INTERPRETATION
-----------------------------------------------------------------
Spam missed (false negatives): {spam_missed}
Ham flagged as spam (false positives): {ham_missed}

False negatives are spam messages incorrectly classified as ham.
False positives are legitimate messages incorrectly classified
as spam.

7. LIMITATIONS
-----------------------------------------------------------------
This model learns from the UCI SMS Spam Collection dataset.
It has not yet been validated on Kenyan scam messages.
A predicted spam label is a screening signal, not proof of fraud.
Test-set results should not be repeatedly used to tune the model.
""".strip()

    # Save the entire pipeline so inference uses the same vectorizer.
    joblib.dump(model, MODEL_PATH)

    REPORT_PATH.write_text(
        results + "\n",
        encoding="utf-8",
    )

    print("\n" + results)
    print(f"\nSaved model:  {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
