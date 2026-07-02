import random
import torch

from models.q_network import QNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64):
        self.state_size = state_size
        self.action_size = action_size

        self.q_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )

    def select_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            action = random.randint(0, self.action_size - 1)
            return action

        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        action = torch.argmax(q_values, dim=1).item()

        return action
    
    def get_q_values(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.squeeze(0)