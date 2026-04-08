"""
tasks/reply.py
Defines the Email Reply Generation task (Hard difficulty).
Each sample provides a customer email and the expected reply
characteristics (tone, required points, forbidden content).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ReplySample:
    """A single email reply generation example."""
    task_id: str
    email_text: str                     # Customer's incoming email
    required_elements: List[str]        # Concepts the reply MUST address
    tone_keywords: List[str]            # Words indicating professional/empathetic tone
    forbidden_phrases: List[str]        # Phrases that indicate a bad reply
    reference_reply: str                # Gold-standard reply for comparison
    instructions: str = (
        "Write a professional, empathetic, and helpful reply to the following customer "
        "email. Address all concerns raised, maintain a polite tone, and provide a "
        "clear resolution or next step."
    )


# ── Curated dataset ─────────────────────────────────────────────────────────

REPLY_SAMPLES: List[ReplySample] = [
    ReplySample(
        task_id="rep_001",
        email_text=(
            "Subject: Order #45892 — Wrong Item Received\n\n"
            "Hello,\n\n"
            "I placed an order for a blue Nike running jacket (size L) three weeks ago, "
            "but I received a red Adidas hoodie instead. This is completely unacceptable! "
            "I needed the jacket for an event this weekend and now I have nothing to wear. "
            "I demand an immediate refund AND a replacement sent overnight shipping at no "
            "additional cost. I've been a loyal customer for 5 years and this is how you "
            "treat me?!\n\nVery frustrated,\nJennifer"
        ),
        required_elements=[
            "apology", "wrong item acknowledged", "replacement or refund offered",
            "expedited shipping", "appreciation for loyalty"
        ],
        tone_keywords=["sincerely", "apologize", "sorry", "understand", "resolve", "valued", "immediately"],
        forbidden_phrases=["not our fault", "nothing we can do", "read the policy", "you should have"],
        reference_reply=(
            "Dear Jennifer,\n\n"
            "I sincerely apologize for receiving the wrong item — that should never happen. "
            "We will immediately ship your blue Nike running jacket (size L) via overnight "
            "delivery at no cost, and you may keep or return the incorrect item at your "
            "convenience with a prepaid label. As a valued 5-year customer, we appreciate "
            "your loyalty and are sorry for the inconvenience caused to your event plans.\n\n"
            "Best regards,\nCustomer Support"
        ),
    ),
    ReplySample(
        task_id="rep_002",
        email_text=(
            "Subject: Billing Issue — Charged Twice\n\n"
            "Hi Support Team,\n\n"
            "I noticed I was charged $49.99 twice on my credit card on March 3rd for my "
            "monthly subscription. I only have one account and should only be billed once. "
            "My account email is john.doe@email.com. Please refund the duplicate charge as "
            "soon as possible. I've attached my bank statement as proof.\n\nThank you,\nJohn"
        ),
        required_elements=[
            "apology", "duplicate charge acknowledged", "refund promised",
            "timeline for refund", "investigation mentioned"
        ],
        tone_keywords=["apologize", "investigate", "refund", "confirm", "resolve", "shortly"],
        forbidden_phrases=["not possible", "no refunds", "contact your bank", "not our problem"],
        reference_reply=(
            "Dear John,\n\n"
            "Thank you for bringing this to our attention. I apologize for the duplicate "
            "charge of $49.99 on March 3rd. I've confirmed the error on your account "
            "(john.doe@email.com) and have initiated a full refund for the duplicate "
            "transaction, which should reflect in your account within 3–5 business days. "
            "You will receive a confirmation email once processed.\n\n"
            "Sincerely,\nBilling Support Team"
        ),
    ),
    ReplySample(
        task_id="rep_003",
        email_text=(
            "Subject: Request for Meeting to Discuss Partnership\n\n"
            "Dear Sales Team,\n\n"
            "My name is Priya Sharma, and I am the Business Development Director at "
            "TechGrow Solutions. We have been following your company's work in the "
            "AI automation space and believe there could be a mutually beneficial "
            "partnership opportunity between our organizations.\n\n"
            "We would love to schedule a 30-minute introductory call at your earliest "
            "convenience to explore synergies. Please let me know your availability "
            "for next week.\n\nBest regards,\nPriya Sharma\nBusiness Development Director"
        ),
        required_elements=[
            "enthusiastic response", "availability provided", "meeting scheduled or proposed",
            "interest in partnership expressed", "contact details requested or provided"
        ],
        tone_keywords=["delighted", "excited", "look forward", "opportunity", "pleasure", "schedule"],
        forbidden_phrases=["not interested", "too busy", "send a proposal first", "we don't do partnerships"],
        reference_reply=(
            "Dear Priya,\n\n"
            "Thank you for reaching out — we are excited to explore a potential partnership "
            "with TechGrow Solutions. I would be delighted to schedule an introductory call. "
            "I am available Tuesday, Wednesday, or Thursday next week between 10 AM and "
            "4 PM EST. Please let me know what works best for you, and I will send a "
            "calendar invite. Looking forward to the conversation!\n\n"
            "Best regards,\nSales Partnerships Team"
        ),
    ),
    ReplySample(
        task_id="rep_004",
        email_text=(
            "Subject: Product Not Working as Advertised\n\n"
            "To Whom It May Concern,\n\n"
            "I purchased your SmartHome Hub v2 last month based on the advertisement "
            "claiming it supports over 100 smart home devices. However, it only connects "
            "to about 20 devices and doesn't support my Philips Hue lights or Nest "
            "thermostat at all. The manual is also very confusing and unhelpful.\n\n"
            "I'm considering returning the product if this can't be resolved. I'd like "
            "either a solution to make it work with my devices or a full refund.\n\n"
            "Regards,\nRobert"
        ),
        required_elements=[
            "apology for inconvenience", "device compatibility acknowledged",
            "troubleshooting steps or solution", "return/refund option mentioned",
            "technical support offered"
        ],
        tone_keywords=["apologize", "understand", "help", "solution", "resolve", "support"],
        forbidden_phrases=["as advertised", "user error", "not our responsibility", "read the manual"],
        reference_reply=(
            "Dear Robert,\n\n"
            "I sincerely apologize for the frustration with your SmartHome Hub v2. "
            "Philips Hue and Nest integrations require a firmware update (v2.4.1) that "
            "may not have been included in your package — please visit our support page "
            "at support.smarthome.com/update to install it, which should resolve the "
            "compatibility issues. If the problem persists, our technical team is "
            "available at 1-800-SMART-HUB. Should the update not resolve your concerns, "
            "we are happy to offer a full refund.\n\n"
            "Best regards,\nTechnical Support Team"
        ),
    ),
    ReplySample(
        task_id="rep_005",
        email_text=(
            "Subject: Cancellation Request\n\n"
            "Hi,\n\n"
            "I would like to cancel my Premium subscription effective immediately. "
            "I've been paying $29.99/month but I'm no longer using the service enough "
            "to justify the cost. Please confirm the cancellation and let me know if "
            "I'll receive a prorated refund for the remaining days in my billing cycle.\n\n"
            "Thanks,\nEmma"
        ),
        required_elements=[
            "cancellation confirmed", "refund policy explained",
            "prorated refund addressed", "retention offer optional",
            "thank customer for past subscription"
        ],
        tone_keywords=["confirm", "understand", "thank", "process", "refund", "appreciate"],
        forbidden_phrases=["cannot cancel", "no refunds", "you signed a contract", "we will charge you"],
        reference_reply=(
            "Dear Emma,\n\n"
            "Your Premium subscription cancellation has been confirmed effective today. "
            "You will receive a prorated refund of $14.20 for the remaining 14 days of "
            "your billing cycle, which will be credited to your original payment method "
            "within 5–7 business days. Thank you for being a valued subscriber, and we "
            "hope to welcome you back in the future. If you change your mind, you can "
            "reactivate at any time.\n\nBest wishes,\nAccount Management Team"
        ),
    ),
]


def get_reply_samples() -> List[ReplySample]:
    """Return all reply generation samples."""
    return REPLY_SAMPLES
