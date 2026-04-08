"""
tasks/summarization.py
Defines the Email Summarization task (Medium difficulty).
Each sample includes a long email and key reference keywords/phrases
that a good summary should contain.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SummarizationSample:
    """A single email summarization example."""
    task_id: str
    email_text: str
    reference_keywords: List[str]   # Words/phrases a good summary should include
    reference_summary: str          # Gold-standard summary for similarity scoring
    instructions: str = (
        "Summarize the following email in 1–2 concise sentences. "
        "Capture the main point and any critical action items."
    )


# ── Curated dataset ─────────────────────────────────────────────────────────

SUMMARIZATION_SAMPLES: List[SummarizationSample] = [
    SummarizationSample(
        task_id="sum_001",
        email_text=(
            "Dear Project Team,\n\n"
            "I hope this message finds you well. I wanted to reach out regarding the "
            "upcoming product launch scheduled for March 15th. As we approach the "
            "deadline, I want to ensure all departments are aligned on their "
            "deliverables.\n\n"
            "The marketing team needs to finalize the press release by March 10th. "
            "The engineering team must complete QA testing no later than March 8th. "
            "Customer support should have the FAQ documentation ready by March 12th.\n\n"
            "Please confirm your team's readiness in the all-hands meeting on Monday "
            "at 2 PM. Failure to meet these deadlines could delay the launch and affect "
            "our Q1 revenue targets significantly.\n\n"
            "Best regards,\nDavid Chen\nVP of Product"
        ),
        reference_keywords=["product launch", "March 15", "deadlines", "QA testing", "press release", "meeting"],
        reference_summary=(
            "VP of Product David Chen reminds the team of the March 15th product launch "
            "deadlines: press release by March 10th, QA testing by March 8th, and FAQ "
            "docs by March 12th, with a readiness check on Monday at 2 PM."
        ),
    ),
    SummarizationSample(
        task_id="sum_002",
        email_text=(
            "Hi Sarah,\n\n"
            "Following up on our conversation from last week regarding the budget "
            "allocation for the new office renovation project. After reviewing the "
            "contractor quotes, I believe we should go with BuildRight Construction "
            "as they offered the most competitive price at $240,000.\n\n"
            "Their proposal covers: full interior renovation of floors 3 and 4, "
            "new HVAC system installation, updated electrical wiring throughout, "
            "and modern open-plan office furniture.\n\n"
            "The timeline is estimated at 6 weeks starting in April, which means "
            "we'd need to arrange temporary workspace for about 45 employees during "
            "this period. I suggest we look into co-working spaces nearby.\n\n"
            "Could you please review the attached quote and approve it by Thursday "
            "so we can finalize the contract?\n\n"
            "Thanks,\nMarcus"
        ),
        reference_keywords=["BuildRight", "$240,000", "renovation", "approval", "Thursday", "6 weeks"],
        reference_summary=(
            "Marcus recommends approving BuildRight Construction's $240,000 quote for "
            "a 6-week office renovation on floors 3 and 4, starting in April, and "
            "requests Sarah's approval by Thursday."
        ),
    ),
    SummarizationSample(
        task_id="sum_003",
        email_text=(
            "Dear Valued Customer,\n\n"
            "We are writing to inform you of an important update to our Terms of "
            "Service and Privacy Policy, effective June 1st, 2024.\n\n"
            "Key changes include:\n"
            "1. Data retention period reduced from 5 years to 2 years.\n"
            "2. Third-party data sharing now requires explicit opt-in consent.\n"
            "3. Users can now request full data export within 48 hours.\n"
            "4. New two-factor authentication will be mandatory for all accounts.\n\n"
            "These changes are being made to comply with new GDPR regulations and to "
            "better protect your privacy. You do not need to take any action unless "
            "you wish to opt in to third-party data sharing or download your data.\n\n"
            "If you have any questions, please contact privacy@ourservice.com.\n\n"
            "Sincerely,\nThe Privacy Team"
        ),
        reference_keywords=["Terms of Service", "June 1st", "GDPR", "two-factor authentication", "data sharing", "privacy"],
        reference_summary=(
            "The company's Terms of Service and Privacy Policy are updating on June 1st "
            "with GDPR compliance changes, including shorter data retention, opt-in "
            "third-party sharing, and mandatory two-factor authentication."
        ),
    ),
    SummarizationSample(
        task_id="sum_004",
        email_text=(
            "Hi Team,\n\n"
            "Exciting news! Our annual performance reviews are coming up next month. "
            "This year we're switching to a new 360-degree feedback system called "
            "PerformanceHub, which will allow peers, managers, and direct reports to "
            "all provide structured feedback.\n\n"
            "Here's the schedule:\n"
            "- Self-assessments due: November 1st\n"
            "- Peer feedback window: November 1–10\n"
            "- Manager reviews: November 11–20\n"
            "- One-on-one discussions: November 21–30\n\n"
            "Please log into PerformanceHub at hub.company.com using your SSO credentials "
            "and complete your profile setup before October 28th. Training sessions will "
            "be held on October 25th and 26th at 3 PM in Room A.\n\n"
            "This is a great opportunity for professional growth. Please reach out to "
            "HR at hr@company.com with any questions.\n\n"
            "Thanks,\nPeople Operations"
        ),
        reference_keywords=["performance reviews", "PerformanceHub", "360-degree", "November", "self-assessment", "training"],
        reference_summary=(
            "Annual performance reviews are switching to PerformanceHub's 360-degree "
            "feedback system; employees must set up their profile by October 28th and "
            "complete self-assessments by November 1st."
        ),
    ),
    SummarizationSample(
        task_id="sum_005",
        email_text=(
            "Dear Dr. Patel,\n\n"
            "I am writing on behalf of the Academic Conference Committee to inform you "
            "that your paper, 'Machine Learning Applications in Early Cancer Detection', "
            "has been accepted for presentation at the International Medical AI Summit "
            "2024 in Zurich, Switzerland.\n\n"
            "The conference will take place from September 18–21, 2024. Your presentation "
            "is scheduled for September 19th at 10:30 AM in Hall C. You will have 20 "
            "minutes to present followed by a 10-minute Q&A session.\n\n"
            "Please submit your final presentation slides by September 5th via the "
            "conference portal at summit2024.org. Hotel accommodation has been arranged "
            "at the Zurich Grand Hotel — details are attached.\n\n"
            "We look forward to your contribution.\n\nBest regards,\nConference Committee"
        ),
        reference_keywords=["paper accepted", "Zurich", "September 19", "presentation", "slides", "Summit"],
        reference_summary=(
            "Dr. Patel's paper on ML in cancer detection was accepted for presentation "
            "at the International Medical AI Summit in Zurich on September 19th at "
            "10:30 AM; final slides are due September 5th."
        ),
    ),
]


def get_summarization_samples() -> List[SummarizationSample]:
    """Return all summarization samples."""
    return SUMMARIZATION_SAMPLES
