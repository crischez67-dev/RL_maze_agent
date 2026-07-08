import random
from collections import deque, namedtuple

import numpy as np
import torch

Experience = namedtuple(
    "Experience",
    ["state", "action", "reward", "next_state", "done"],
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        self.memory.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.memory, batch_size)
        return experiences

    def sample_tensors(self, batch_size):
        experiences = self.sample(batch_size)

        states = torch.tensor(
            np.array([experience.state for experience in experiences]),
            dtype=torch.float32,
        )

        actions = torch.tensor(
            [experience.action for experience in experiences],
            dtype=torch.long,
        ).unsqueeze(1)

        rewards = torch.tensor(
            [experience.reward for experience in experiences],
            dtype=torch.float32,
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array([experience.next_state for experience in experiences]),
            dtype=torch.float32,
        )

        dones = torch.tensor(
            [experience.done for experience in experiences],
            dtype=torch.float32,
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)