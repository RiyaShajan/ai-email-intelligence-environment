"""
models.py
Pydantic data models for the AI Email Intelligence Environment.
Defines the core data structures: Observation, Action, and Reward.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


class Observation(BaseModel):
    """
    Represents what the agent observes at each step.
    Contains the task type and the email text to process.
    """
    task_type: Literal["classification", "summarization", "reply"] = Field(
        ...,
        description="The type of task the agent must perform."
    )
    email_text: str = Field(
        ...,
        description="The raw email text the agent must analyze or respond to."
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for the current task."
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Optional human-readable instructions for the task."
    )


class Action(BaseModel):
    """
    Represents the agent's response to an observation.
    A single 'response' string covering classification label,
    summary text, or generated reply — depending on task type.
    """
    response: str = Field(
        ...,
        description=(
            "Agent's answer. For classification: 'spam', 'important', or 'normal'. "
            "For summarization: a 1–2 sentence summary. "
            "For reply: a professional email reply."
        )
    )


class Reward(BaseModel):
    """
    Represents the reward signal returned after an action.
    Includes a numeric score and human-readable feedback.
    """
    score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Reward score in range [-1.0, 1.0]. Higher is better."
    )
    feedback: str = Field(
        ...,
        description="Explanation of why this score was awarded."
    )
    partial: bool = Field(
        default=False,
        description="True if the reward is partial (not full credit)."
    )


class StepResult(BaseModel):
    """
    The full result returned by env.step().
    Mirrors the OpenEnv step contract.
    """
    observation: Observation
    reward: Reward
    done: bool = Field(
        default=False,
        description="True when all tasks in the episode are complete."
    )
    info: dict = Field(
        default_factory=dict,
        description="Extra metadata (e.g., task index, total score so far)."
    )


class EnvironmentState(BaseModel):
    """
    Internal environment state snapshot returned by env.state().
    """
    current_task_index: int
    total_tasks: int
    cumulative_score: float
    task_history: list = Field(default_factory=list)
    done: bool = False
