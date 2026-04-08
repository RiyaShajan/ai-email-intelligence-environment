class EmailEnv:
    def __init__(self):
        self.current_step = 0
        self.max_steps = 5
        self.email = "Meeting request tomorrow"
        self.done = False

    def reset(self):
        self.current_step = 0
        self.done = False
        return self.state()

    def state(self):
        return {
            "email": self.email,
            "step": self.current_step
        }

    def step(self, action):
        self.current_step += 1

        # SIMPLE LOGIC
        if action == "classify_meeting":
            reward = 1.0
            self.done = True
        elif "meeting" in action:
            reward = 0.5
        else:
            reward = 0.0

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done, {}
