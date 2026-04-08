"""
tasks/classification.py
Defines the Email Classification task (Easy difficulty).
Each sample includes an email and its ground-truth label.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationSample:
    """A single email classification example."""
    task_id: str
    email_text: str
    label: str          # "spam" | "important" | "normal"
    instructions: str = (
        "Classify the following email as exactly one of: spam, important, or normal. "
        "Reply with a single word only."
    )


# ── Curated dataset of 10 labelled emails ──────────────────────────────────

CLASSIFICATION_SAMPLES: List[ClassificationSample] = [
    ClassificationSample(
        task_id="cls_001",
        email_text=(
            "Congratulations! You've been selected to receive a FREE iPhone 15! "
            "Click the link below to claim your prize before it expires. "
            "Limited time offer — act now! https://totally-legit-prize.ru/claim"
        ),
        label="spam",
    ),
    ClassificationSample(
        task_id="cls_002",
        email_text=(
            "Hi Team,\n\n"
            "Reminder: the Q3 budget review meeting is scheduled for tomorrow at 10 AM "
            "in Conference Room B. Please bring your department's expense reports.\n\n"
            "Best,\nSarah (CFO)"
        ),
        label="important",
    ),
    ClassificationSample(
        task_id="cls_003",
        email_text=(
            "Hey,\n\n"
            "Just wanted to let you know that the lunch order for Friday is confirmed. "
            "We're going with sandwiches from the usual place. Let me know if you have "
            "any dietary preferences.\n\nCheers,\nMike"
        ),
        label="normal",
    ),
    ClassificationSample(
        task_id="cls_004",
        email_text=(
            "URGENT: Your bank account has been compromised! "
            "Verify your identity immediately by entering your SSN and password at "
            "http://secure-bank-verify.xyz — failure to act in 24 hours will result "
            "in account suspension!"
        ),
        label="spam",
    ),
    ClassificationSample(
        task_id="cls_005",
        email_text=(
            "Dear Mr. Johnson,\n\n"
            "We regret to inform you that your employment contract will not be renewed "
            "after December 31st. Please schedule a meeting with HR at your earliest "
            "convenience to discuss transition arrangements.\n\n"
            "Regards,\nHuman Resources Department"
        ),
        label="important",
    ),
    ClassificationSample(
        task_id="cls_006",
        email_text=(
            "Hi!\n\nJust sharing the photos from last weekend's team picnic. "
            "Had a great time! See the attached album.\n\nBest,\nAlex"
        ),
        label="normal",
    ),
    ClassificationSample(
        task_id="cls_007",
        email_text=(
            "You have WON $1,000,000 in the National Lottery draw! "
            "To claim your winnings, send us your full name, address, and a processing "
            "fee of $500 via wire transfer. Reply ASAP!"
        ),
        label="spam",
    ),
    ClassificationSample(
        task_id="cls_008",
        email_text=(
            "Dear All,\n\n"
            "Please be advised that the server maintenance window is scheduled for "
            "Saturday 2 AM–6 AM. All systems will be unavailable during this period. "
            "Please save your work before leaving Friday.\n\nIT Department"
        ),
        label="important",
    ),
    ClassificationSample(
        task_id="cls_009",
        email_text=(
            "Hey,\n\nDid you catch the game last night? "
            "Unbelievable finish! We should grab a beer and recap it sometime.\n\nCheers"
        ),
        label="normal",
    ),
    ClassificationSample(
        task_id="cls_010",
        email_text=(
            "Make $5000/week from home with NO experience needed! "
            "Our proven system has helped thousands. Click here to start earning TODAY. "
            "This offer expires at midnight — don't miss out!"
        ),
        label="spam",
    ),
]


def get_classification_samples() -> List[ClassificationSample]:
    """Return all classification samples."""
    return CLASSIFICATION_SAMPLES
