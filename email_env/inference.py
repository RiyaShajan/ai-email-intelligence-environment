"""
inference.py
Baseline inference script for the AI Email Intelligence Environment.

Uses the OpenAI-compatible API (via HF_TOKEN environment variable) to
run a language model agent through all tasks and report the final score.

Usage:
  export HF_TOKEN="your_api_key_here"
  python inference.py

  # Or use OpenAI key:
  export HF_TOKEN="sk-..."
  python inference.py --provider openai
"""

import os
import sys
import json
import time
import argparse
from typing import Optional

# ── Optional: use openai SDK if available, else fall back to requests ────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    import urllib.request

from email_env.env import EmailEnvironment
from email_env.models import Action, Observation


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL_OPENAI = "gpt-4o-mini"
DEFAULT_MODEL_HF = "mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = "https://api-inference.huggingface.co/models/{model}/v1/chat/completions"


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return (
        "You are an expert email analyst. You will receive emails along with task "
        "instructions. Follow the instructions exactly and produce only the requested "
        "output — no preamble, no explanations unless the task asks for a reply."
    )


def build_user_prompt(obs: Observation) -> str:
    return (
        f"Task: {obs.task_type.upper()}\n\n"
        f"Instructions:\n{obs.instructions}\n\n"
        f"Email:\n{obs.email_text}\n\n"
        "Your response:"
    )


# ── API call helpers ──────────────────────────────────────────────────────────

def call_openai(client: "OpenAI", model: str, system: str, user: str) -> str:
    """Call OpenAI-compatible chat completions."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def call_hf_api(api_key: str, model: str, system: str, user: str) -> str:
    """Call Hugging Face Inference API (OpenAI-compatible endpoint)."""
    url = HF_API_URL.format(model=model)
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


# ── Main inference loop ───────────────────────────────────────────────────────

def run_inference(
    provider: str = "openai",
    model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full episode using an LLM agent and return the results dict.

    Args:
        provider: "openai" or "hf" (Hugging Face).
        model:    Override default model name.
        verbose:  Print step-by-step progress.

    Returns:
        dict with per-task scores and overall average.
    """
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        print("  Set it with: export HF_TOKEN='your_api_key'", file=sys.stderr)
        sys.exit(1)

    # Set up client / model
    if provider == "openai":
        if not HAS_OPENAI:
            print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
            sys.exit(1)
        effective_model = model or DEFAULT_MODEL_OPENAI
        client = OpenAI(api_key=api_key)
        call_fn = lambda sys_p, usr_p: call_openai(client, effective_model, sys_p, usr_p)
    else:
        effective_model = model or DEFAULT_MODEL_HF
        call_fn = lambda sys_p, usr_p: call_hf_api(api_key, effective_model, sys_p, usr_p)

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  AI Email Intelligence — Baseline Inference")
        print(f"  Provider : {provider.upper()}  |  Model : {effective_model}")
        print(f"{'═'*70}\n")

    env = EmailEnvironment(seed=42)
    obs = env.reset()

    system_prompt = build_system_prompt()
    results = {"classification": [], "summarization": [], "reply": []}
    step_logs = []

    while True:
        user_prompt = build_user_prompt(obs)

        # ── LLM call with retry ────────────────────────────────────────────
        try:
            response_text = call_fn(system_prompt, user_prompt)
        except Exception as exc:
            print(f"  [Step {env.current_step+1}] API Error: {exc} — using empty response.")
            response_text = ""
            time.sleep(2)

        action = Action(response=response_text)
        next_obs, reward, done, info = env.step(action)

        step_logs.append({
            "task_id": obs.task_id,
            "task_type": obs.task_type,
            "response": response_text[:120],
            "score": reward.score,
            "feedback": reward.feedback,
        })
        results[obs.task_type].append(reward.score)

        if verbose:
            print(
                f"  [{info['step']:2d}/{info['total_tasks']}] "
                f"{obs.task_type.upper():<16} "
                f"score={reward.score:+.4f}  "
                f"avg={info['average_score']:+.4f}"
            )

        if done:
            break
        obs = next_obs
        time.sleep(0.3)   # rate-limit buffer

    # ── Compute summary ────────────────────────────────────────────────────
    all_scores = [s for scores in results.values() for s in scores]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0

    summary = {
        "model": effective_model,
        "provider": provider,
        "per_task_type": {
            k: {"count": len(v), "avg_score": round(sum(v)/len(v), 4) if v else 0.0}
            for k, v in results.items()
        },
        "overall_average_score": round(overall_avg, 4),
        "total_tasks": len(all_scores),
        "step_logs": step_logs,
    }

    if verbose:
        print(f"\n{'─'*70}")
        print("  RESULTS SUMMARY")
        print(f"{'─'*70}")
        for task_type, stats in summary["per_task_type"].items():
            print(f"  {task_type.capitalize():<16} avg={stats['avg_score']:+.4f}  "
                  f"(n={stats['count']})")
        print(f"\n  OVERALL AVERAGE SCORE : {overall_avg:+.4f}")
        print(f"{'═'*70}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AI Email Intelligence Environment baseline inference."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "hf"],
        default="openai",
        help="API provider: 'openai' (default) or 'hf' (Hugging Face).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default model name.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results JSON.",
    )
    args = parser.parse_args()

    results = run_inference(provider=args.provider, model=args.model)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
