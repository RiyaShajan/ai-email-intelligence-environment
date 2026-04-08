from email_env.env import EmailEnvironment
from email_env.models import Action

def run(task_type, email_text, response):
    env = EmailEnvironment(seed=42)
    obs = env.reset()
    
    while obs.task_type != task_type and env.current_step < env.total_tasks:
        _, _, done, _ = env.step(Action(response="spam"))
        if done:
            break
        obs = env._make_observation()

    action = Action(response=response)
    _, reward, _, info = env.step(action)

    return {
        "score": reward.score,
        "feedback": reward.feedback
    }
