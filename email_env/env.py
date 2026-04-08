"""
env.py
AI Email Intelligence Environment — OpenEnv-compliant core.

Implements:
  reset()          → Observation
  step(action)     → (Observation, Reward, done: bool, info: dict)
  state()          → EnvironmentState

The environment runs through a mixed sequence of all three task types
in a fixed reproducible order.
"""

import random
from typing import Tuple, Dict, Any, List

from email_env.models import Observation, Action, Reward, StepResult, EnvironmentState
from email_env.tasks.classification import get_classification_samples
from email_env.tasks.summarization import get_summarization_samples
from email_env.tasks.reply import get_reply_samples
from email_env.graders import grade_classification, grade_summarization, grade_reply


class EmailEnvironment:
    """
    OpenEnv-compliant environment for email intelligence tasks.

    Episode structure:
      - All classification samples  (easy)
      - All summarization samples   (medium)
      - All reply samples           (hard)

    The task queue is deterministic (seeded). An agent iterates
    through every task; the episode ends when all tasks are complete.
    """

    def __init__(self, seed: int = 42, shuffle: bool = False):
        """
        Args:
            seed:    Random seed for reproducibility.
            shuffle: Whether to shuffle the task order (default: False).
        """
        self._seed = seed
        self._shuffle = shuffle
        self._rng = random.Random(seed)

        # Build the master task list once
        self._all_tasks: List[Dict[str, Any]] = self._build_task_list()

        # Runtime state
        self._current_index: int = 0
        self._cumulative_score: float = 0.0
        self._history: List[Dict[str, Any]] = []
        self._action_counts: Dict[str, int] = {}   # Track repeated actions per task

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_task_list(self) -> List[Dict[str, Any]]:
        """Assemble all tasks into a flat list."""
        tasks = []

        for s in get_classification_samples():
            tasks.append({
                "task_type": "classification",
                "task_id": s.task_id,
                "email_text": s.email_text,
                "instructions": s.instructions,
                "label": s.label,
            })

        for s in get_summarization_samples():
            tasks.append({
                "task_type": "summarization",
                "task_id": s.task_id,
                "email_text": s.email_text,
                "instructions": s.instructions,
                "reference_keywords": s.reference_keywords,
                "reference_summary": s.reference_summary,
            })

        for s in get_reply_samples():
            tasks.append({
                "task_type": "reply",
                "task_id": s.task_id,
                "email_text": s.email_text,
                "instructions": s.instructions,
                "required_elements": s.required_elements,
                "tone_keywords": s.tone_keywords,
                "forbidden_phrases": s.forbidden_phrases,
                "reference_reply": s.reference_reply,
            })

        if self._shuffle:
            self._rng.shuffle(tasks)

        return tasks

    def _current_task(self) -> Dict[str, Any]:
        return self._all_tasks[self._current_index]

    def _make_observation(self) -> Observation:
        task = self._current_task()
        return Observation(
            task_type=task["task_type"],
            email_text=task["email_text"],
            task_id=task["task_id"],
            instructions=task["instructions"],
        )

    def _detect_loop_penalty(self, task_id: str, response: str) -> float:
        """
        Return a penalty [-0.1] if the agent is repeating the exact same
        response for the same task (loop detection).
        """
        key = f"{task_id}::{response[:100]}"
        count = self._action_counts.get(key, 0)
        self._action_counts[key] = count + 1
        if count >= 1:
            return -0.1  # penalise repeated identical actions
        return 0.0

    # ── OpenEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """
        Reset the environment to its initial state.

        Returns:
            The first Observation in the episode.
        """
        self._current_index = 0
        self._cumulative_score = 0.0
        self._history = []
        self._action_counts = {}
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply an action and advance the environment.

        Args:
            action: Agent's Action containing a response string.

        Returns:
            (next_observation, reward, done, info)
              - next_observation: Observation for the next task (or current if done).
              - reward:           Reward object with score and feedback.
              - done:             True when all tasks are exhausted.
              - info:             Metadata dict with step details.
        """
        if self._current_index >= len(self._all_tasks):
            # Episode already over — return terminal state
            terminal_obs = Observation(
                task_type="classification",
                email_text="(Episode complete — call reset() to start a new episode.)",
                task_id="terminal",
            )
            return (
                terminal_obs,
                Reward(score=0.0, feedback="Episode is already complete."),
                True,
                {"status": "already_done"},
            )

        task = self._current_task()
        response = action.response

        # ── Grade the action ───────────────────────────────────────────────
        if task["task_type"] == "classification":
            reward = grade_classification(response, task["label"])

        elif task["task_type"] == "summarization":
            reward = grade_summarization(
                response,
                task["reference_keywords"],
                task["reference_summary"],
            )

        elif task["task_type"] == "reply":
            reward = grade_reply(
                response,
                task["required_elements"],
                task["tone_keywords"],
                task["forbidden_phrases"],
                task["reference_reply"],
            )

        else:
            reward = Reward(score=0.0, feedback="Unknown task type.")

        # ── Loop detection ─────────────────────────────────────────────────
        loop_penalty = self._detect_loop_penalty(task["task_id"], response)
        if loop_penalty < 0:
            new_score = max(-1.0, reward.score + loop_penalty)
            reward = Reward(
                score=new_score,
                feedback=reward.feedback + f" [Loop penalty applied: {loop_penalty}]",
                partial=reward.partial,
            )

        # ── Update state ───────────────────────────────────────────────────
        self._cumulative_score += reward.score
        self._history.append({
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "response_preview": response[:80],
            "score": reward.score,
        })

        self._current_index += 1
        done = self._current_index >= len(self._all_tasks)

        # ── Build next observation ─────────────────────────────────────────
        if done:
            next_obs = Observation(
                task_type="classification",
                email_text="(Episode complete.)",
                task_id="done",
            )
        else:
            next_obs = self._make_observation()

        info = {
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "step": self._current_index,
            "total_tasks": len(self._all_tasks),
            "cumulative_score": round(self._cumulative_score, 4),
            "average_score": round(
                self._cumulative_score / self._current_index, 4
            ),
            "done": done,
        }

        return next_obs, reward, done, info

    def state(self) -> EnvironmentState:
        """
        Return the current environment state snapshot.

        Returns:
            EnvironmentState with step counters, scores, and history.
        """
        return EnvironmentState(
            current_task_index=self._current_index,
            total_tasks=len(self._all_tasks),
            cumulative_score=round(self._cumulative_score, 4),
            task_history=list(self._history),
            done=self._current_index >= len(self._all_tasks),
        )

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def total_tasks(self) -> int:
        return len(self._all_tasks)

    @property
    def current_step(self) -> int:
        return self._current_index
