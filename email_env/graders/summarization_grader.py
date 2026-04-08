"""
graders/summarization_grader.py
Deterministic grader for the Email Summarization task.

Scoring strategy (partial scoring):
  1. Keyword coverage  (0.0–0.5): fraction of reference_keywords present in summary.
  2. Length penalty    (0.0–0.2): reward concise 1–3 sentence summaries.
  3. Similarity bonus  (0.0–0.3): simple unigram F1 vs the reference_summary.

Final score is clamped to [0.0, 1.0].
"""

import re
from typing import List

from email_env.models import Reward


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _keyword_coverage(summary_tokens: List[str], keywords: List[str]) -> float:
    """Fraction of reference keywords (case-insensitive) found in summary."""
    if not keywords:
        return 0.5
    summary_set = set(summary_tokens)
    hits = sum(
        1 for kw in keywords
        if any(part in summary_set for part in _tokenize(kw))
    )
    return hits / len(keywords)


def _unigram_f1(hypothesis: List[str], reference: List[str]) -> float:
    """Compute unigram F1 between hypothesis and reference token lists."""
    if not hypothesis or not reference:
        return 0.0
    hyp_set = set(hypothesis)
    ref_set = set(reference)
    overlap = len(hyp_set & ref_set)
    precision = overlap / len(hyp_set) if hyp_set else 0.0
    recall = overlap / len(ref_set) if ref_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _length_score(summary: str) -> float:
    """
    Reward summaries that are 1–3 sentences and at least 10 words.
    Penalise empty, trivially short, or extremely long outputs.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]", summary) if len(s.strip()) > 5]
    words = len(summary.split())

    if words < 5:
        return 0.0
    if 1 <= len(sentences) <= 3 and words <= 80:
        return 0.2
    if len(sentences) > 3 or words > 120:
        return 0.05   # too long
    return 0.1


# ── Public API ────────────────────────────────────────────────────────────────

def grade_summarization(
    predicted: str,
    reference_keywords: List[str],
    reference_summary: str,
) -> Reward:
    """
    Grade a summarization response.

    Args:
        predicted:           The agent's generated summary.
        reference_keywords:  Key concepts that should appear in a good summary.
        reference_summary:   Gold-standard summary for similarity scoring.

    Returns:
        Reward with score in [0.0, 1.0] and detailed feedback.
    """
    if not predicted or len(predicted.strip()) < 5:
        return Reward(
            score=0.0,
            feedback="Empty or trivially short summary. No credit awarded.",
            partial=False,
        )

    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(reference_summary)

    # Component scores
    kw_score = _keyword_coverage(pred_tokens, reference_keywords)    # 0–1
    len_score = _length_score(predicted)                              # 0–0.2
    sim_score = _unigram_f1(pred_tokens, ref_tokens)                 # 0–1

    # Weighted combination: keyword 50%, similarity 30%, length 20%
    weighted = (kw_score * 0.50) + (sim_score * 0.30) + (len_score / 0.2 * 0.20)
    final_score = min(max(round(weighted, 4), 0.0), 1.0)

    partial = 0.0 < final_score < 1.0
    feedback_parts = [
        f"Keyword coverage: {kw_score:.0%} ({int(kw_score * len(reference_keywords))}/{len(reference_keywords)} keywords found).",
        f"Semantic similarity (unigram F1 vs reference): {sim_score:.0%}.",
        f"Length appropriateness score: {len_score:.2f}/0.20.",
        f"Final weighted score: {final_score:.4f}.",
    ]
    if final_score >= 0.75:
        feedback_parts.append("Excellent summary — captures most key points concisely.")
    elif final_score >= 0.4:
        feedback_parts.append("Partial credit — summary is relevant but misses some key details.")
    else:
        feedback_parts.append("Low score — summary is too vague, too brief, or off-topic.")

    return Reward(
        score=final_score,
        feedback=" ".join(feedback_parts),
        partial=partial,
    )
