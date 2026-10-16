"""
File: 02_prepare_dataset.py

Purpose:
Prepare the audited UCI SMS Spam Collection dataset for baseline
machine-learning experiments.

Steps:
1. Validate the input dataset.
2. Detect messages assigned conflicting labels.
3. Remove duplicate messages.
4. Create a stratified 80/20 train-test split.
5. Verify that no exact messages overlap between the splits.
6. Save the prepared datasets and a split summary.

Outputs:
- data/processed/sms_spam_deduplicated.csv
- data/processed/train.csv
- data/processed/test.csv
- reports/split_summary.txt

The original raw dataset is never modified.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Resolve all paths relative to the ScamShield project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sms_spam_collection.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEDUPLICATED_PATH = PROCESSED_DIR / "sms_spam_deduplicated.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REPORT_PATH = REPORTS_DIR / "split_summary.txt"

EXPECTED_LABELS = {"ham", "spam"}
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_and_validate_dataset() -> pd.DataFrame:
    """Load the audited dataset and validate its structure and contents."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}\n"
            "Run ml/01_dataset_audit.py first."
        )

    df = pd.read_csv(
        INPUT_PATH,
        dtype={"label": "string", "message": "string"},
    )

    required_columns = {"label", "message"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain these columns: {required_columns}"
        )

    # Keep only the columns required for this experiment.
    df = df[["label", "message"]].copy()

    # Standardize label formatting without altering message content.
    df["label"] = df["label"].str.strip().str.lower()

    if df.isna().any().any():
        raise ValueError("The dataset contains missing labels or messages.")

    if df["message"].str.strip().eq("").any():
        raise ValueError("The dataset contains empty messages.")

    actual_labels = set(df["label"].unique())

    if actual_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Expected labels {EXPECTED_LABELS}, found {actual_labels}."
        )

    # The same exact message must not have conflicting labels.
    labels_per_message = df.groupby("message")["label"].nunique()
    conflicting_messages = labels_per_message[labels_per_message > 1]

    if not conflicting_messages.empty:
        examples = conflicting_messages.index[:5].tolist()
        raise ValueError(
            f"Found {len(conflicting_messages)} messages with conflicting "
            f"labels. Examples: {examples}"
        )

    return df


def format_distribution(df: pd.DataFrame) -> str:
    """Return label counts and percentages for a dataset split."""
    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True) * 100

    lines = []

    for label in ["ham", "spam"]:
        count = int(counts.get(label, 0))
        percentage = percentages.get(label, 0.0)
        lines.append(f"  {label}: {count:,} ({percentage:.2f}%)")

    return "\n".join(lines)


def main() -> None:
    """Prepare the data, create the split, and write a summary report."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_dataset()

    original_rows = len(df)
    exact_duplicate_rows = int(df.duplicated().sum())
    unique_message_count = int(df["message"].nunique())

    # Since conflicting labels have already been ruled out, retaining one
    # row per exact message is safe for this initial experiment.
    deduplicated_df = df.drop_duplicates(
        subset=["message"],
        keep="first",
    ).reset_index(drop=True)

    removed_rows = original_rows - len(deduplicated_df)

    # Stratification keeps the ham/spam proportions similar in both splits.
    train_df, test_df = train_test_split(
        deduplicated_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=deduplicated_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Confirm that exact messages cannot leak across the split.
    train_messages = set(train_df["message"])
    test_messages = set(test_df["message"])
    overlap = train_messages.intersection(test_messages)

    if overlap:
        raise RuntimeError(
            f"Data leakage detected: {len(overlap)} messages overlap."
        )

    # Save the prepared datasets.
    deduplicated_df.to_csv(DEDUPLICATED_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    report = f"""
=================================================================
SCAMSHIELD KE — DATA PREPARATION REPORT
=================================================================
Generated (UTC): {datetime.now(timezone.utc).isoformat()}
Input: {INPUT_PATH.relative_to(PROJECT_ROOT)}

1. VALIDATION
-----------------------------------------------------------------
Input rows: {original_rows:,}
Valid labels: {", ".join(sorted(EXPECTED_LABELS))}
Missing values: 0
Empty messages: 0
Messages with conflicting labels: 0

2. DEDUPLICATION
-----------------------------------------------------------------
Exact duplicate rows beyond first occurrences: {exact_duplicate_rows:,}
Unique messages before deduplication: {unique_message_count:,}
Rows removed: {removed_rows:,}
Rows after deduplication: {len(deduplicated_df):,}

3. TRAIN / TEST SPLIT
-----------------------------------------------------------------
Training rows: {len(train_df):,} ({len(train_df) / len(deduplicated_df) * 100:.2f}%)
Testing rows: {len(test_df):,} ({len(test_df) / len(deduplicated_df) * 100:.2f}%)
Requested test size: {TEST_SIZE:.0%}
Random seed: {RANDOM_STATE}
Stratified by: label

Training label distribution:
{format_distribution(train_df)}

Testing label distribution:
{format_distribution(test_df)}

4. LEAKAGE CHECK
-----------------------------------------------------------------
Exact message overlap between training and testing: {len(overlap)}
Result: {"PASS" if not overlap else "FAIL"}

5. OUTPUT FILES
-----------------------------------------------------------------
{DEDUPLICATED_PATH.relative_to(PROJECT_ROOT)}
{TRAIN_PATH.relative_to(PROJECT_ROOT)}
{TEST_PATH.relative_to(PROJECT_ROOT)}
reports/split_summary.txt

6. LIMITATIONS
-----------------------------------------------------------------
Deduplication removes exact repeated messages, not near-duplicates.
The dataset represents SMS ham/spam, not every type of scam.
It is not specifically representative of Kenyan messages.
The test set should be reserved for final evaluation, not repeated
tuning of model choices.
""".strip()

    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    print(report)
    print("\nPreparation completed successfully.")
    print(f"Training data: {TRAIN_PATH}")
    print(f"Testing data:  {TEST_PATH}")
    print(f"Report:        {REPORT_PATH}")


if __name__ == "__main__":
    main()
