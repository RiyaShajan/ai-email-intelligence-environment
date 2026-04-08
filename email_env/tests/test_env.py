"""
tests/test_env.py
Unit tests for the AI Email Intelligence Environment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from email_env.env import EmailEnvironment
from email_env.models import Observation, Action, Reward
from email_env.graders import grade_classification, grade_summarization, grade_reply


# ── Environment API tests ─────────────────────────────────────────────────────

class TestEnvironmentAPI:
    def setup_method(self):
        self.env = EmailEnvironment(seed=42)

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, Observation)
        assert obs.task_type in ("classification", "summarization", "reply")
        assert len(obs.email_text) > 0

    def test_step_returns_tuple(self):
        self.env.reset()
        action = Action(response="spam")
        result = self.env.step(action)
        assert len(result) == 4
        obs, reward, done, info = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, Reward)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_state_returns_state(self):
        self.env.reset()
        state = self.env.state()
        assert state.current_task_index == 0
        assert state.total_tasks == 20
        assert state.done is False

    def test_episode_completes_after_all_tasks(self):
        obs = self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = self.env.step(Action(response="spam"))
            steps += 1
        assert steps == 20
        assert self.env.state().done is True

    def test_reset_clears_state(self):
        self.env.reset()
        self.env.step(Action(response="spam"))
        self.env.reset()
        state = self.env.state()
        assert state.current_task_index == 0
        assert state.cumulative_score == 0.0

    def test_total_tasks_is_20(self):
        assert self.env.total_tasks == 20

    def test_info_contains_required_keys(self):
        self.env.reset()
        _, _, _, info = self.env.step(Action(response="spam"))
        for key in ("task_id", "task_type", "step", "total_tasks",
                    "cumulative_score", "average_score"):
            assert key in info


# ── Classification grader tests ───────────────────────────────────────────────

class TestClassificationGrader:
    def test_correct_label_scores_one(self):
        r = grade_classification("spam", "spam")
        assert r.score == 1.0

    def test_correct_label_case_insensitive(self):
        r = grade_classification("SPAM", "spam")
        assert r.score == 1.0

    def test_wrong_label_penalty(self):
        r = grade_classification("normal", "spam")
        assert r.score == -0.2

    def test_invalid_response_heavy_penalty(self):
        r = grade_classification("I think it might be a spam email", "spam")
        # Should extract "spam" from the response
        assert r.score == 1.0  # extraction works

    def test_gibberish_heavy_penalty(self):
        r = grade_classification("asdfqwerty", "spam")
        assert r.score == -0.5

    def test_all_three_labels(self):
        for label in ("spam", "important", "normal"):
            r = grade_classification(label, label)
            assert r.score == 1.0


# ── Summarization grader tests ────────────────────────────────────────────────

class TestSummarizationGrader:
    KEYWORDS = ["product launch", "March 15", "deadlines", "QA testing"]
    REFERENCE = (
        "VP of Product reminds teams of March 15 product launch deadlines "
        "including QA testing by March 8."
    )

    def test_good_summary_high_score(self):
        summary = (
            "The VP of Product reminds teams of the March 15 product launch, "
            "with QA testing deadlines and press release due dates."
        )
        r = grade_summarization(summary, self.KEYWORDS, self.REFERENCE)
        assert r.score >= 0.5

    def test_empty_summary_zero_score(self):
        r = grade_summarization("", self.KEYWORDS, self.REFERENCE)
        assert r.score == 0.0

    def test_very_short_summary_low_score(self):
        r = grade_summarization("ok", self.KEYWORDS, self.REFERENCE)
        assert r.score == 0.0

    def test_partial_credit_possible(self):
        r = grade_summarization(
            "The team has some deadlines coming up next month.",
            self.KEYWORDS,
            self.REFERENCE,
        )
        assert 0.0 <= r.score <= 1.0

    def test_score_in_range(self):
        r = grade_summarization("random text", self.KEYWORDS, self.REFERENCE)
        assert 0.0 <= r.score <= 1.0


# ── Reply grader tests ────────────────────────────────────────────────────────

class TestReplyGrader:
    REQUIRED = ["apology", "replacement or refund offered", "appreciation for loyalty"]
    TONE_KWS = ["sincerely", "apologize", "sorry", "resolve", "valued"]
    FORBIDDEN = ["not our fault", "nothing we can do"]
    REFERENCE = (
        "Dear Jennifer,\n\nI sincerely apologize for receiving the wrong item. "
        "We will ship your replacement overnight. You are a valued customer.\n\n"
        "Best regards,\nSupport"
    )

    def test_good_reply_high_score(self):
        reply = (
            "Dear Jennifer,\n\n"
            "I sincerely apologize for receiving the wrong item. We will immediately "
            "ship your replacement overnight at no cost. You are a valued customer "
            "and we appreciate your loyalty.\n\n"
            "Best regards,\nSupport Team"
        )
        r = grade_reply(reply, self.REQUIRED, self.TONE_KWS, self.FORBIDDEN, self.REFERENCE)
        assert r.score >= 0.5

    def test_forbidden_phrase_penalised(self):
        reply = (
            "Dear Jennifer,\n\n"
            "This is not our fault but we apologize. We will resolve this.\n\n"
            "Regards"
        )
        r_bad = grade_reply(reply, self.REQUIRED, self.TONE_KWS, self.FORBIDDEN, self.REFERENCE)
        reply_good = (
            "Dear Jennifer,\n\nI sincerely apologize. We will resolve this immediately.\n\nBest regards"
        )
        r_good = grade_reply(reply_good, self.REQUIRED, self.TONE_KWS, self.FORBIDDEN, self.REFERENCE)
        assert r_bad.score <= r_good.score

    def test_empty_reply_zero_score(self):
        r = grade_reply("", self.REQUIRED, self.TONE_KWS, self.FORBIDDEN, self.REFERENCE)
        assert r.score == 0.0

    def test_score_in_range(self):
        r = grade_reply(
            "OK I will help.",
            self.REQUIRED, self.TONE_KWS, self.FORBIDDEN, self.REFERENCE,
        )
        assert 0.0 <= r.score <= 1.0


# ── Reproducibility tests ─────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_task_order(self):
        env1 = EmailEnvironment(seed=42)
        env2 = EmailEnvironment(seed=42)
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.task_id == obs2.task_id
        assert obs1.task_type == obs2.task_type

    def test_grader_deterministic(self):
        r1 = grade_classification("spam", "spam")
        r2 = grade_classification("spam", "spam")
        assert r1.score == r2.score
        assert r1.feedback == r2.feedback
