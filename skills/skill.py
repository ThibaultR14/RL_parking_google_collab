class Skill:
    def __init__(self, actions_sequence, name="skill"):
        self.actions_sequence = actions_sequence
        self.length = len(actions_sequence)
        self.current_step = 0
        self.name = name

    def reset(self):
        self.current_step = 0

    def step(self):
        action = self.actions_sequence[self.current_step]
        self.current_step += 1
        return action
