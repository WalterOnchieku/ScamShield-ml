"""
File: 01_dataset_audit.py

Purpose:
Download and audit the official UCI SMS Spam Collection dataset.

Checks:
- Dataset size and structure
- Missing and empty values
- Label distribution
- Duplicate and conflicting messages
- Message lengths and representative examples

Outputs:
- data/raw/sms_spam_collection.zip
- data/raw/SMSSpamCollection
- data/raw/sms_spam_collection.csv
- reports/dataset_audit.txt
"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

ZIP_PATH = RAW_DIR / "sms_spam_collection.zip"
TEXT_PATH = RAW_DIR / "SMSSpamCollection"
CSV_PATH = RAW_DIR / "sms_spam_collection.csv"
REPORT_PATH = REPORTS_DIR / "dataset_audit.txt"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)


def download_dataset() -> None:
    """Download the official archive and extract the original text file."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        print("Downloading the official UCI dataset...")

        request = Request(
            DATASET_URL,
            headers={"User-Agent": "ScamShieldKE/0.1 dataset-audit"},
        )

        with urlopen(request, timeout=60) as response:
            ZIP_PATH.write_bytes(response.read())

    if not ZipFile(ZIP_PATH).testzip() is None:
        raise ValueError("The downloaded ZIP archive failed its integrity check.")

    with ZipFile(ZIP_PATH) as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name == "SMSSpamCollection"
        ]

        if not candidates:
            raise FileNotFoundError(
                "Could not find SMSSpamCollection inside the UCI archive."
            )

        TEXT_PATH.write_bytes(archive.read(candidates[0]))

    print(f"Archive: {ZIP_PATH}")
    print(f"Original dataset: {TEXT_PATH}")


def load_dataset() -> pd.DataFrame:
    """Parse the tab-separated UCI source file."""
    download_dataset()

    df = pd.read_csv(
        TEXT_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
    )

    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype("string")

    return df


def audit_dataset(df: pd.DataFrame) -> str:
    """Generate a human-readable data quality report."""
    lines = []

    def add(value=""):
        lines.append(str(value))

    add("=" * 65)
    add("SCAMSHIELD KE — DATASET AUDIT")
    add("=" * 65)
    add(f"Audit timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    add("Source: UCI SMS Spam Collection")
    add("Dataset ID: 228")
    add("Download: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
    add()

    add("1. DATASET OVERVIEW")
    add("-" * 65)
    add(f"Rows: {len(df):,}")
    add(f"Columns: {df.columns.tolist()}")
    add(f"Memory usage: {df.memory_usage(deep=True).sum():,} bytes")
    add()

    add("2. MISSING AND EMPTY VALUES")
    add("-" * 65)
    add(f"Missing messages: {df['message'].isna().sum():,}")
    add(f"Missing labels: {df['label'].isna().sum():,}")
    empty_count = int(df["message"].str.strip().eq("").sum())
    add(f"Empty messages: {empty_count:,}")
    add()

    add("3. LABEL DISTRIBUTION")
    add("-" * 65)
    counts = df["label"].value_counts(dropna=False)

    for label, count in counts.items():
        percentage = count / len(df) * 100
        add(f"{label}: {count:,} ({percentage:.2f}%)")

    add(f"Unique labels: {df['label'].nunique()}")
    add()

    add("4. DUPLICATE ANALYSIS")
    add("-" * 65)
    add(f"Exact duplicate rows: {int(df.duplicated().sum()):,}")
    add(
        "Repeated message rows: "
        f"{int(df.duplicated(subset=['message']).sum()):,}"
    )

    conflicting = (
        df.groupby("message", dropna=False)["label"]
        .nunique()
        .gt(1)
        .sum()
    )
    add(f"Messages with conflicting labels: {int(conflicting):,}")
    add()

    add("5. MESSAGE LENGTH")
    add("-" * 65)
    lengths = df["message"].str.len()
    add(f"Minimum: {lengths.min()} characters")
    add(f"Median: {lengths.median():.0f} characters")
    add(f"Maximum: {lengths.max()} characters")
    add()

    add("6. EXAMPLE MESSAGES")
    add("-" * 65)

    for label in ("ham", "spam"):
        add(f"\nLabel: {label}")

        examples = df.loc[df["label"] == label, "message"].head(3)

        for number, message in enumerate(examples, start=1):
            message = " ".join(str(message).split())
            if len(message) > 180:
                message = message[:177] + "..."
            add(f"{number}. {message}")

    add()
    add("7. INITIAL QUALITY FLAGS")
    add("-" * 65)

    if df["message"].isna().any() or df["label"].isna().any():
        add("WARNING: Missing values detected.")

    if empty_count:
        add("WARNING: Empty messages detected.")

    if not df["label"].isin(["ham", "spam"]).all():
        add("WARNING: Unexpected labels detected.")

    if df.duplicated().any():
        add("NOTE: Duplicate rows exist; review before splitting the data.")

    if conflicting:
        add("WARNING: Identical messages have conflicting labels.")

    if not any("WARNING:" in line for line in lines):
        add("No missing values, empty messages, or unexpected labels detected.")

    add()
    add("LIMITATION")
    add("-" * 65)
    add(
        "This dataset labels SMS as ham or spam. It does not represent "
        "all scam types or specifically represent Kenyan scam messages."
    )

    return "\n".join(lines)


def main() -> None:
    """Download the data, run the audit, and save outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    if df.empty:
        raise ValueError("The downloaded dataset contains no records.")

    df.to_csv(CSV_PATH, index=False)

    report = audit_dataset(df)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")

    print("\n" + report)
    print("\nAudit completed successfully.")
    print(f"CSV:    {CSV_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
