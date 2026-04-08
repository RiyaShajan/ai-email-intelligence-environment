"""
graders/reply_grader.py
Deterministic grader for the Email Reply Generation task.

Scoring dimensions (each 0.0–1.0, then averaged):
  1. Relevance       — required_elements coverage
  2. Tone            — tone_keywords present, no forbidden_phrases
  3. Completeness    — unigram F1 against the reference reply
  4. Format          — reply has a greeting, body, and sign-off
"""

import re
from typing import List

from email_env.models import Reward


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _element_coverage(reply_lower: str, elements: List[str]) -> float:
    """Fraction of required semantic elements present (substring heuristic)."""
    if not elements:
        return 1.0
    hits = 0
    for el in elements:
        # Each element can be 1-3 words; check if any individual content word is present
        words = [w for w in el.lower().split() if len(w) > 3]
        if words and any(w in reply_lower for w in words):
            hits += 1
        elif not words and el.lower() in reply_lower:
            hits += 1
    return hits / len(elements)


def _tone_score(reply_lower: str, tone_keywords: List[str], forbidden_phrases: List[str]) -> float:
    """
    Score tone:
      +0.5 base
      +0.5 × fraction of tone_keywords present
      –0.3 per forbidden phrase found (min 0.0)
    """
    if not tone_keywords:
        tone_hit = 0.5
    else:
        hits = sum(1 for kw in tone_keywords if kw.lower() in reply_lower)
        tone_hit = 0.5 + 0.5 * (hits / len(tone_keywords))

    penalty = sum(0.3 for fp in forbidden_phrases if fp.lower() in reply_lower)
    return max(0.0, min(1.0, tone_hit - penalty))


def _unigram_f1(hyp: List[str], ref: List[str]) -> float:
    if not hyp or not ref:
        return 0.0
    hyp_set, ref_set = set(hyp), set(ref)
    overlap = len(hyp_set & ref_set)
    p = overlap / len(hyp_set) if hyp_set else 0.0
    r = overlap / len(ref_set) if ref_set else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _format_score(reply: str) -> float:
    """
    Check for basic email format:
      - Has a greeting (Dear / Hi / Hello)
      - Has a body (>30 words)
      - Has a sign-off (regards / sincerely / best / thank)
    """
    lower = reply.lower()
    has_greeting = bool(re.search(r"\b(dear|hi|hello|greetings)\b", lower))
    has_body = len(reply.split()) > 30
    has_signoff = bool(re.search(r"\b(regards|sincerely|best|thank you|thanks|yours)\b", lower))
    return (has_greeting + has_body + has_signoff) / 3.0


# ── Public API ────────────────────────────────────────────────────────────────

def grade_reply(
    predicted: str,
    required_elements: List[str],
    tone_keywords: List[str],
    forbidden_phrases: List[str],
    reference_reply: str,
) -> Reward:
    """
    Grade a reply generation response.

    Args:
        predicted:          The agent's generated reply.
        required_elements:  Concepts the reply must address.
        tone_keywords:      Words indicating professional/empathetic tone.
        forbidden_phrases:  Phrases that indicate a bad/rude reply.
        reference_reply:    Gold-standard reply for similarity scoring.

    Returns:
        Reward with score in [0.0, 1.0] and detailed feedback.
    """
    if not predicted or len(predicted.strip()) < 20:
        return Reward(
            score=0.0,
            feedback="Empty or trivially short reply. No credit awarded.",
            partial=False,
        )

    pred_lower = predicted.lower()
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference_reply)

    # Dimension scores
    relevance = _element_coverage(pred_lower, required_elements)
    tone = _tone_score(pred_lower, tone_keywords, forbidden_phrases)
    completeness = _unigram_f1(pred_tokens, ref_tokens)
    fmt = _format_score(predicted)

    # Weighted average: relevance 35%, tone 30%, completeness 20%, format 15%
    final_score = (
        relevance    * 0.35 +
        tone         * 0.30 +
        completeness * 0.20 +
        fmt          * 0.15
    )
    final_score = min(max(round(final_score, 4), 0.0), 1.0)

    # Check for forbidden phrases explicitly
    violations = [fp for fp in forbidden_phrases if fp.lower() in pred_lower]

    partial = 0.0 < final_score < 1.0
    feedback_parts = [
        f"Relevance (required elements): {relevance:.0%}.",
        f"Tone quality: {tone:.0%}.",
        f"Semantic completeness (F1 vs reference): {completeness:.0%}.",
        f"Email format score: {fmt:.0%}.",
        f"Final weighted score: {final_score:.4f}.",
    ]
    if violations:
        feedback_parts.append(
            f"WARNING: Forbidden phrase(s) detected: {violations}. Tone score penalised."
        )
    if final_score >= 0.75:
        feedback_parts.append("Excellent reply — professional, relevant, and complete.")
    elif final_score >= 0.40:
        feedback_parts.append("Partial credit — reply is relevant but has tone or completeness gaps.")
    else:
        feedback_parts.append("Low score — reply is missing key elements, unprofessional, or off-topic.")

    return Reward(
        score=final_score,
        feedback=" ".join(feedback_parts),
        partial=partial,
    )
