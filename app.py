import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import uvicorn

from email_env.env import EmailEnvironment
from email_env.models import Action

# ---------- FastAPI ----------
api = FastAPI(title="AI Email Intelligence Environment")

env = EmailEnvironment(seed=42)
obs = env.reset()

class ActionRequest(BaseModel):
    response: str

@api.post("/reset")
def reset():
    global env, obs
    env = EmailEnvironment(seed=42)
    obs = env.reset()
    return {
        "task_type": obs.task_type,
        "email_text": obs.email_text,
        "task_id": obs.task_id,
        "instructions": obs.instructions
    }

@api.post("/step")
def step(action: ActionRequest):
    global env, obs
    next_obs, reward, done, info = env.step(Action(response=action.response))
    obs = next_obs
    return {
        "observation": {
            "task_type": next_obs.task_type,
            "email_text": next_obs.email_text,
            "task_id": next_obs.task_id,
            "instructions": next_obs.instructions
        },
        "reward": {
            "score": reward.score,
            "feedback": reward.feedback,
            "partial": reward.partial
        },
        "done": done,
        "info": info
    }

@api.get("/state")
def state():
    s = env.state()
    return {
        "current_task_index": s.current_task_index,
        "total_tasks": s.total_tasks,
        "cumulative_score": s.cumulative_score,
        "done": s.done
    }

# ---------- Gradio UI ----------
def ui_reset():
    result = reset()
    return f"Task: {result['task_type']}\n\nEmail:\n{result['email_text']}"

def ui_step(response):
    result = step(ActionRequest(response=response))
    return f"Score: {result['reward']['score']}\nFeedback: {result['reward']['feedback']}"

with gr.Blocks() as gradio_app:
    gr.Markdown("# 📧 AI Email Intelligence Environment")

    with gr.Tab("Reset"):
        btn = gr.Button("Reset")
        out = gr.Textbox()
        btn.click(ui_reset, outputs=out)

    with gr.Tab("Step"):
        inp = gr.Textbox(label="Your response")
        btn2 = gr.Button("Submit")
        out2 = gr.Textbox()
        btn2.click(ui_step, inputs=inp, outputs=out2)

# ---------- Mount ----------
app = FastAPI()
app.mount("/", api)
app = gr.mount_gradio_app(app, gradio_app, path="/gradio")

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
