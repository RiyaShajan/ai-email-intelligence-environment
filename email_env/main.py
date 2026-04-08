"""
main.py
Interactive demo and CLI entry point for the AI Email Intelligence Environment.

Usage:
  python main.py              # Run demo with hard-coded answers
  python main.py --random     # Run demo with random (wrong) answers
"""

import argparse
import json
import sys

from email_env.env import EmailEnvironment
from email_env.models import Action


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def run_demo(use_random_answers: bool = False) -> None:
    """
    Step through the full episode, printing observations and rewards.
    Uses a set of demo answers (or random wrong ones) to illustrate the API.
    """
    env = EmailEnvironment(seed=42)

    # Pre-built demo answers (one per task, in order)
    demo_answers = (
        # ── Classification (10 tasks) ──────────────────────────────────────
        ["spam", "important", "normal", "spam", "important",
         "normal", "spam", "important", "normal", "spam"]
        +
        # ── Summarization (5 tasks) ────────────────────────────────────────
        [
            "VP of Product reminds teams of March 15 product launch deadlines including QA testing by March 8 and press release by March 10.",
            "Marcus recommends approving BuildRight Construction's $240,000 quote for a 6-week office renovation and asks Sarah to approve by Thursday.",
            "Service terms are updating June 1st with GDPR changes: shorter data retention, opt-in third-party sharing, and mandatory two-factor authentication.",
            "Annual performance reviews switch to PerformanceHub 360-degree feedback; profile setup due October 28th and self-assessments due November 1st.",
            "Dr. Patel's cancer detection paper was accepted at the Medical AI Summit in Zurich on September 19th; slides due September 5th.",
        ]
        +
        # ── Reply (5 tasks) ────────────────────────────────────────────────
        [
            "Dear Jennifer,\n\nI sincerely apologize for receiving the wrong item. We will immediately ship your blue Nike running jacket size L via overnight shipping at no cost. You are a valued customer and we appreciate your loyalty.\n\nBest regards,\nSupport Team",
            "Dear John,\n\nThank you for contacting us. I apologize for the duplicate charge of $49.99. I have initiated a refund which will appear within 3-5 business days. Thank you for your patience.\n\nSincerely,\nBilling Support",
            "Dear Priya,\n\nThank you for reaching out! We are excited about potential partnership opportunities with TechGrow Solutions. I am available next week on Tuesday and Wednesday between 10 AM and 3 PM EST. Looking forward to our conversation.\n\nBest regards,\nPartnerships Team",
            "Dear Robert,\n\nI apologize for the inconvenience. A firmware update (v2.4.1) should resolve the Philips Hue and Nest compatibility issues. Please visit our support page to install it. If issues persist, we are happy to offer a full refund.\n\nBest regards,\nTechnical Support",
            "Dear Emma,\n\nYour Premium subscription cancellation is confirmed effective today. You will receive a prorated refund within 5-7 business days. Thank you for being a valued subscriber and we hope to see you again.\n\nBest wishes,\nAccount Team",
        ]
    )

    random_wrong = {
        "classification": "important",  # always wrong-ish
        "summarization": "The email is about something.",
        "reply": "Ok.",
    }

    obs = env.reset()
    print("\n" + "═" * 70)
    print("  AI EMAIL INTELLIGENCE ENVIRONMENT — Demo Run")
    print("═" * 70)

    answer_idx = 0
    task_scores = {"classification": [], "summarization": [], "reply": []}

    while True:
        print(f"\nStep {env.current_step + 1}/{env.total_tasks}  "
              f"[Task: {obs.task_type.upper()}  |  ID: {obs.task_id}]")
        print_separator()
        print(f"EMAIL:\n{obs.email_text[:300]}{'...' if len(obs.email_text) > 300 else ''}")
        print(f"\nINSTRUCTIONS: {obs.instructions}")
        print_separator()

        # Choose answer
        if use_random_answers:
            answer = random_wrong[obs.task_type]
        else:
            answer = demo_answers[answer_idx] if answer_idx < len(demo_answers) else "N/A"
        answer_idx += 1

        print(f"AGENT RESPONSE: {answer[:120]}{'...' if len(answer) > 120 else ''}")

        action = Action(response=answer)
        next_obs, reward, done, info = env.step(action)

        print(f"\nREWARD  score={reward.score:+.4f}  partial={reward.partial}")
        print(f"FEEDBACK: {reward.feedback}")
        print(f"INFO: {json.dumps({k: v for k, v in info.items() if k not in ('done',)}, indent=2)}")

        task_scores[obs.task_type].append(reward.score)

        if done:
            break
        obs = next_obs

    # ── Final Summary ──────────────────────────────────────────────────────
    env_state = env.state()
    avg_score = env_state.cumulative_score / env_state.total_tasks

    print("\n" + "═" * 70)
    print("  EPISODE COMPLETE — Final Results")
    print("═" * 70)
    for task_type, scores in task_scores.items():
        if scores:
            print(f"  {task_type.capitalize():16s}  avg={sum(scores)/len(scores):+.4f}  "
                  f"tasks={len(scores)}")
    print(f"\n  TOTAL CUMULATIVE SCORE : {env_state.cumulative_score:+.4f}")
    print(f"  AVERAGE SCORE PER TASK : {avg_score:+.4f}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Email Intelligence Environment demo.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random (wrong) answers instead of demo answers.",
    )
    args = parser.parse_args()
    run_demo(use_random_answers=args.random)
