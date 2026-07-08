import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.q_network import QNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size

        self.q_network = QNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size,
        )

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
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
    
    def learn(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return None

        states, actions, rewards, next_states, dones = memory.sample_tensors(batch_size)

        q_values = self.q_network(states)
        selected_q_values = q_values.gather(1, actions)

        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            max_next_q_values = next_q_values.max(dim=1, keepdim=True)[0]
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(selected_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()