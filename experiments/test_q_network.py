import torch

from env.maps import SIMPLE_MAZE, SIMPLE_START, SIMPLE_GOAL
from env.maze_env import MazeEnv
from models.q_network import QNetwork


env = MazeEnv(
    maze=SIMPLE_MAZE,
    start_pos=SIMPLE_START,
    goal_pos=SIMPLE_GOAL,
    max_steps=50,
)

state = env.reset()

print("Estado original:", state)
print("Tamaño del estado:", len(state))

q_network = QNetwork(
    state_size=6,
    action_size=4,
)

state_tensor = torch.tensor(state, dtype=torch.float32)

print("Tensor del estado:", state_tensor)
print("Forma antes de unsqueeze:", state_tensor.shape)

state_tensor = state_tensor.unsqueeze(0)

print("Forma después de unsqueeze:", state_tensor.shape)

q_values = q_network(state_tensor)

print("Valores Q:", q_values)
print("Forma de salida:", q_values.shape)

best_action = torch.argmax(q_values, dim=1)

print("Mejor acción según la red:", best_action)
print("Mejor acción como número:", best_action.item())

action = best_action.item()

next_state, reward, done, info = env.step(action)

print("Acción ejecutada en el entorno:", action)
print("Nueva posición:", env.agent_pos)
print("Recompensa:", reward)
print("Done:", done)
print("Info:", info)