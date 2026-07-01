import random


class RandomAgent:
    def __init__(self, num_actions=4):
        self.num_actions = num_actions

    def select_action(self, state):
        action = random.randint(0, self.num_actions - 1)
        return action
    
    