from env import EmailEnv

env = EmailEnv()

state = env.reset()
done = False
total_reward = 0

while not done:
    action = "classify_meeting"  # baseline action
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("Final Score:", total_reward)
