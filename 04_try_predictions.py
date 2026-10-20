
"""
File: 04_try_predictions.py

Purpose:
Load the trained ScamShield KE baseline and try predictions
on new example messages.

This is a smoke test for inference, not a model evaluation.
The examples are illustrative and are not labeled test-set samples.
"""

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "scamshield_baseline.joblib"


def main() -> None:
    """Load the model and display predictions for example messages."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run ml/03_train_baseline.py first."
        )

    model = joblib.load(MODEL_PATH)

    examples = [
        (
            "Ordinary message",
            "Hi, are we still meeting for lunch tomorrow?",
        ),
        (
            "Prize offer",
            "Congratulations! You have won a free prize. "
            "Claim now by sending your details.",
        ),
        (
            "Illustrative job offer",
            "URGENT JOB OFFER! Earn KSh 15000 daily from home. "
            "Pay a registration fee today to secure your position.",
        ),
    ]

    classifier = model.named_steps["classifier"]
    class_names = classifier.classes_

    print("SCAMSHIELD KE — INFERENCE SMOKE TEST")
    print("=" * 55)

    for description, message in examples:
        predicted_label = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]

        scores = dict(zip(class_names, probabilities))

        print(f"\nExample: {description}")
        print(f"Message: {message}")
        print(f"Predicted label: {predicted_label}")

        for label in class_names:
            print(f"  {label} score: {scores[label]:.4f}")

    print("\nNote: scores are model estimates, not proof of fraud.")
    print("This baseline was trained on general SMS ham/spam data.")


if __name__ == "__main__":
    main()
