"""
graders/classification_grader.py
Deterministic grader for the Email Classification task.

Scoring rules:
  +1.0  → exact label match (case-insensitive, stripped)
  -0.2  → wrong label (valid but incorrect)
  -0.5  → invalid response (not one of the three allowed labels)
"""

from email_env.models import Reward

VALID_LABELS = {"spam", "important", "normal"}


def grade_classification(predicted: str, ground_truth: str) -> Reward:
    """
    Grade a classification response.

    Args:
        predicted:    Raw string returned by the agent.
        ground_truth: The correct label ("spam", "important", or "normal").

    Returns:
        Reward with score in [-0.5, 1.0] and descriptive feedback.
    """
    # Normalise: lowercase, strip whitespace and punctuation
    cleaned = predicted.strip().lower().strip(".,!?\"'")

    # Handle multi-word responses: try to extract a valid label token
    if cleaned not in VALID_LABELS:
        for word in cleaned.split():
            word_clean = word.strip(".,!?\"'")
            if word_clean in VALID_LABELS:
                cleaned = word_clean
                break

    gt = ground_truth.strip().lower()

    if cleaned == gt:
        return Reward(
            score=1.0,
            feedback=f"Correct! The email is '{gt}'.",
            partial=False,
        )
    elif cleaned in VALID_LABELS:
        return Reward(
            score=-0.2,
            feedback=(
                f"Incorrect classification. Predicted '{cleaned}' but expected '{gt}'. "
                f"Penalty applied for wrong valid label."
            ),
            partial=False,
        )
    else:
        return Reward(
            score=-0.5,
            feedback=(
                f"Invalid response '{predicted[:80]}'. "
                f"Must be exactly one of: spam, important, normal. "
                f"Heavy penalty applied."
            ),
            partial=False,
        )
